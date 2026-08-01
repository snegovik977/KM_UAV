# -*- coding: utf-8 -*-
"""Автомат миссии: от команды на взлёт до посадки на площадку.

    IDLE ──takeoff-пакет / --manual──> ARM ──> TAKEOFF ──> SURVEY ──┐
                                                                    │
    DISARM <── LAND <── RETURN <────────── DUST (пункты 2-4) <───────┘

Плюс аварийные переходы ИЗ ЛЮБОГО состояния: критический заряд, таймаут состояния,
срыв навигации, команда оператора. Они реализованы исключением AbortMission, которое
всплывает даже изнутри блокирующего перелёта.

Один поток на SDK. Все обращения к автопилоту идут отсюда, включая телеметрию:
во время перелёта Flight периодически зовёт _tick(), и в нём миссия читает позицию,
шлёт телеметрию и проверяет аварийные условия. Поэтому второго потока к SDK не нужно,
а поток перцепции (главный, в main.py) только читает снимок MissionState под замком.

Состояния DUST и recon_done оставлены точками расширения: пункты 2-4 плана
встраиваются в готовую цепочку, не переписывая автомат.
"""
from __future__ import annotations

import math
import threading
import time

from survey import plan_from_config
from tasks import TaskProfile

# Те же строки, что в протоколе (protocol/messages.py -> MISSION_STATES): состояние
# уходит в телеметрию, и расхождение здесь означает отвергнутый пакет.
IDLE, ARM, TAKEOFF, SURVEY, DUST, RETURN, LAND, DISARM, ERROR = (
    "IDLE", "ARM", "TAKEOFF", "SURVEY", "DUST", "RETURN", "LAND", "DISARM", "ERROR")


def to_world(x, y, origin):
    """Из СК дрона в общую СК роя.

    В подзадачах 2.3.x начало общей СК совпадает с точкой взлёта, и origin нулевой.
    В финале дрон стартует с автомобиля, и момент команды takeoff физически задаёт
    общее начало отсчёта обеим машинам — тогда сдвиг и поворот приходят в пакете.

    Функция общая для телеметрии и для координат станций намеренно: разойдись эти
    два пересчёта — и дрон сообщал бы своё положение в одной СК, а станции в другой.
    """
    ox, oy, oyaw = origin
    угол = math.radians(oyaw)
    return (ox + x * math.cos(угол) - y * math.sin(угол),
            oy + x * math.sin(угол) + y * math.cos(угол))


class AbortMission(Exception):
    """Аварийное завершение. severity: return — вернуться и сесть, land — сесть здесь."""

    def __init__(self, reason, severity="return"):
        Exception.__init__(self, reason)
        self.reason = reason
        self.severity = severity


class MissionState(object):
    """Снимок состояния для потока перцепции и телеметрии. Всё под одним замком."""

    def __init__(self):
        self._lock = threading.Lock()
        self.state = IDLE
        self.position = (0.0, 0.0, 0.0)
        # Все три угла, а не только курс: локализация станций считает луч камеры
        # в мировой СК, и крен с тангажом входят в неё наравне с рысканием
        # (~3.5 см ошибки на градус, docs/DRONE_PLAN.md §2.2).
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.battery = 0.0
        self.origin = (0.0, 0.0, 0.0)      # начало общей СК роя из пакета takeoff
        self.waypoint = 0
        self.waypoints_total = 0
        self.stations = 0                  # пункт 2: сколько станций подтверждено
        # Посадочный знак «H»: его точка на земле в СК дрона (pad) и время наблюдения
        # (pad_ts). Наполняет поток перцепции (perception/landing.py), читает полётный
        # поток при центрировании перед посадкой. None — знак ещё не найден.
        self.pad = None
        self.pad_ts = 0.0
        self.pad_score = 0.0
        self.abort_reason = None
        self.started = None
        self.finished = None

    def update(self, **поля):
        with self._lock:
            for имя, значение in поля.items():
                setattr(self, имя, значение)

    def snapshot(self):
        with self._lock:
            return {
                "state": self.state,
                "position": self.position,
                "roll": self.roll,
                "pitch": self.pitch,
                "yaw": self.yaw,
                "battery": self.battery,
                "origin": self.origin,
                "waypoint": self.waypoint,
                "waypoints_total": self.waypoints_total,
                "stations": self.stations,
                "pad": self.pad,
                "pad_ts": self.pad_ts,
                "pad_score": self.pad_score,
                "abort_reason": self.abort_reason,
                "elapsed": (time.time() - self.started) if self.started else 0.0,
            }

    def hud(self):
        """Одна строка для наложения на кадр отладочной трансляции."""
        s = self.snapshot()
        x, y, z = s["position"]
        return ("%s %d/%d  x=%+.2f y=%+.2f z=%.2f  yaw=%+.0f  batt=%.2fB  %ds"
                % (s["state"], s["waypoint"], s["waypoints_total"], x, y, z,
                   s["yaw"], s["battery"], int(s["elapsed"])))


class Mission(object):
    """Полётная логика. Запускается в отдельном потоке, главный поток занят перцепцией."""

    def __init__(self, flight, transport, cfg, state=None, log=None, factory=None,
                 task=None, registry=None):
        self.flight = flight
        self.transport = transport
        self.cfg = cfg
        self.state = state or MissionState()
        self._log = log or (lambda text: print(text))
        self.factory = factory              # protocol.MessageFactory, может быть None
        # Профиль подзадачи: он решает, уходят ли наружу пакеты разведки. Без него
        # (старые тесты, ручной запуск) берётся значение из конфига.
        self.task = task or TaskProfile(cfg["mission"].get("task", "2.3.1"))
        # Реестр наполняет поток перцепции; миссии он нужен ровно в одном месте —
        # чтобы дослать уточнённые координаты перед recon_done. None, если детекции нет.
        self.registry = registry

        mission = cfg["mission"]
        self.manual_start = bool(mission["manual_start"])
        self.telemetry_period = 1.0 / max(float(mission["telemetry_hz"]), 0.1)
        self.timeouts = {
            SURVEY: float(mission["survey_timeout"]),
            RETURN: float(mission["return_timeout"]),
            LAND: float(mission["land_timeout"]),
        }
        self.batt_low = float(cfg["flight"]["batt_low"])
        self.batt_critical = float(cfg["flight"]["batt_critical"])
        self.h_survey = float(cfg["flight"]["h_survey"])

        # Центрирование по посадочному знаку перед спуском. Точку знака кладёт в
        # состояние поток перцепции (perception/landing.py); здесь только сведение над
        # ней. Секция может отсутствовать (старые конфиги) — тогда садимся вслепую.
        landing = cfg.get("landing", {}) or {}
        self.landing_enabled = bool(landing.get("enabled", False))
        self.landing = {
            "center_height": float(landing.get("center_height", self.h_survey)),
            "tol": float(landing.get("tol", 0.10)),
            "gain": float(landing.get("gain", 0.9)),
            "max_steps": int(landing.get("max_steps", 6)),
            "step_timeout": float(landing.get("step_timeout", 6.0)),
            "acquire_timeout": float(landing.get("acquire_timeout", 4.0)),
            "fresh": float(landing.get("fresh", 1.5)),
        }

        self._stop = threading.Event()
        self._state_started = time.time()
        self._last_telemetry = 0.0
        self._zone_override = None
        self._nav_lost_since = None
        self.plan = None

    # ------------------------------------------------------------------- служебное

    def _set_state(self, новое):
        self._log("[миссия] %s -> %s" % (self.state.state, новое))
        self.state.update(state=новое)
        self._state_started = time.time()
        self._tick(force_telemetry=True)

    def stop(self, reason="команда оператора"):
        """Мягкая остановка снаружи: миссия прервётся на ближайшем такте."""
        self.state.update(abort_reason=reason)
        self._stop.set()

    def _tick(self, force_telemetry=False):
        """Один такт наблюдения: телеметрия наружу, аварийные условия внутрь.

        Зовётся и из шагов автомата, и из Flight во время перелёта — иначе аварийная
        ситуация, возникшая в середине блокирующего перелёта, была бы замечена только
        по его окончании.
        """
        position = self.flight.position()
        orientation = self.flight.orientation()
        battery = self.flight.battery()
        поля = {}
        if position is not None:
            поля["position"] = position
        if orientation is not None:
            поля["roll"], поля["pitch"], поля["yaw"] = orientation
        if battery is not None:
            поля["battery"] = battery
        if поля:
            self.state.update(**поля)

        self._publish_telemetry(force=force_telemetry)
        self._check_aborts(battery)

    def _publish_telemetry(self, force=False):
        """Телеметрия баллов не даёт, но без неё не отладить полёт и не защитить решение."""
        if self.transport is None or self.factory is None:
            return
        now = time.time()
        if not force and now - self._last_telemetry < self.telemetry_period:
            return
        self._last_telemetry = now
        s = self.state.snapshot()
        x, y = self._to_world(s["position"][0], s["position"][1])
        try:
            self.transport.send(self.factory.telemetry(
                x=x, y=y, z=s["position"][2],
                yaw=s["yaw"] + s["origin"][2], batt=s["battery"], state=s["state"]))
        except Exception as e:
            # Транспорт не имеет права ронять полёт.
            self._log("[миссия] телеметрия не ушла: %s: %s" % (type(e).__name__, e))

    def _to_world(self, x, y):
        return to_world(x, y, self.state.snapshot()["origin"])

    def _check_aborts(self, battery):
        if self._stop.is_set():
            raise AbortMission(self.state.snapshot()["abort_reason"] or "остановлено снаружи")

        if battery is not None and battery > 0:
            if battery < self.batt_critical:
                raise AbortMission("критический заряд %.2f В" % battery, severity="land")
            if battery < self.batt_low:
                raise AbortMission("низкий заряд %.2f В" % battery)

        текущее = self.state.snapshot()["state"]
        лимит = self.timeouts.get(текущее)
        if лимит and time.time() - self._state_started > лимит:
            raise AbortMission("таймаут состояния %s (%.0f с)" % (текущее, лимит))

        # Срыв оптопотока: одиночный сбой бывает и на исправном полёте, поэтому
        # реагируем не на первое же False, а на устойчивую потерю.
        if not self.flight.nav_ok():
            if self._nav_lost_since is None:
                self._nav_lost_since = time.time()
            elif time.time() - self._nav_lost_since > 2.0:
                raise AbortMission("потеряна навигация (оптопоток)")
        else:
            self._nav_lost_since = None

    # ------------------------------------------------------------------ шаги миссии

    def _wait_command(self):
        """IDLE: ждём пакет takeoff от автомобиля или передатчика."""
        self._set_state(IDLE)
        if self.manual_start:
            self._log("[миссия] manual_start: стартуем без пакета от хаба")
            return
        self._log("[миссия] жду команду takeoff от передатчика/автомобиля")
        while not self._stop.is_set():
            if self.transport is not None:
                for msg in self.transport.recv():
                    if msg["type"] != "takeoff":
                        continue
                    data = msg["data"]
                    self.state.update(origin=(data["origin_x"], data["origin_y"],
                                              data["origin_yaw"]))
                    if data.get("zone_w") or data.get("zone_h"):
                        self._zone_override = data
                        self._log("[миссия] зона из пакета: %.1f x %.1f м"
                                  % (data.get("zone_w", 0.0), data.get("zone_h", 0.0)))
                    self._log("[миссия] команда на взлёт принята от %s, origin=(%.2f, %.2f, %.0f)"
                              % (msg["src"], data["origin_x"], data["origin_y"],
                                 data["origin_yaw"]))
                    return
            time.sleep(0.1)
        raise AbortMission("прервано до взлёта", severity="none")

    def _arm(self):
        self._set_state(ARM)
        self.state.update(started=time.time())
        self.flight.arm()

    def _takeoff(self):
        self._set_state(TAKEOFF)
        self.flight.takeoff()
        # Подъём на рабочую высоту отдельной командой: takeoff() поднимает на свою,
        # а змейка считалась под h_survey.
        self.flight.goto(0.0, 0.0, self.h_survey, on_tick=self._tick)

    def _survey(self):
        """Облёт территории — те самые 3 балла подзадачи 2.3.1.

        Форма маршрута (обход трассы по контуру или змейка) выбирается в config.yaml
        -> route.mode; автомат их не различает.
        """
        self._set_state(SURVEY)
        self.plan = plan_from_config(self.cfg, self._zone_override)
        for предупреждение in self.plan.warnings:
            self._log("[миссия] ВНИМАНИЕ: %s" % предупреждение)
        self._log("[миссия] %s" % self.plan.summary(speed=self.cfg["flight"]["speed"]))
        self.state.update(waypoints_total=len(self.plan.waypoints), waypoint=0)

        for номер, точка in enumerate(self.plan, 1):
            self.state.update(waypoint=номер)
            достигнута = self.flight.goto(точка.x, точка.y, точка.z, точка.yaw,
                                          on_tick=self._tick)
            if not достигнута:
                # Одна не достигнутая точка — не повод бросать облёт: остальные полосы
                # ещё дадут покрытие. А вот молчать об этом нельзя.
                self._log("[миссия] точка %d/%d не достигнута, иду дальше"
                          % (номер, len(self.plan.waypoints)))
        self._log("[миссия] облёт завершён")
        self._after_survey()

    def _after_survey(self):
        """Конец разведки: наружу уходит recon_done с числом найденных станций.

        Пакет обязан уйти строго после всей змейки, иначе автомобиль начнёт строить
        маршрут по неполной карте.

        Шлём и при нуле станций: для автомобиля recon_done — сигнал «разведка кончилась,
        можно ехать», и молчание оставило бы его ждать бесконечно. Пустая карта — тоже
        результат разведки.
        """
        # Момент самый выгодный для уточнения координат: все наблюдения собраны,
        # медиана максимально устойчива, а автомобиль ещё не начал строить маршрут.
        if self.registry is not None:
            try:
                уточнено = self.registry.flush()
                if уточнено:
                    self._log("[миссия] дослано уточнённых координат: %d" % уточнено)
            except Exception as e:
                self._log("[миссия] уточнение координат не удалось: %s: %s"
                          % (type(e).__name__, e))

        if not self.task.sends_recon_done:
            return
        if self.transport is None or self.factory is None:
            return
        количество = self.state.snapshot()["stations"]
        self.transport.send(self.factory.recon_done(count=количество))
        self._log("[миссия] recon_done: %d станций" % количество)

    def _dust(self):
        """Точка расширения пункта 4: подлёт к запылённой станции и обдув.

        Пока реестра станций нет, состояние проходится насквозь. Автомат уже знает
        про него, поэтому пункт 4 не потребует переписывать цепочку.
        """
        return

    def _return(self):
        """Возврат на посадочную площадку «H» — она же точка взлёта, то есть (0, 0)."""
        self._set_state(RETURN)
        self.flight.goto(0.0, 0.0, self.h_survey, on_tick=self._tick)

    def _land(self):
        """Посадка на площадку — 2 балла подзадачи 2.3.1.

        Перед вертикальным спуском дрон сводит себя над знаком «H» по зрению
        (_center_over_pad): возврат приводит его в точку взлёта, но оптопоток за облёт
        успевает увести локальную СК, и без доводки «сел в пределах площадки» держится
        на волоске. Не увидели знак — садимся по координатам возврата, как раньше.
        """
        self._set_state(LAND)
        self._center_over_pad()
        self.flight.land()
        self._set_state(DISARM)
        self.flight.disarm()

    def _center_over_pad(self):
        """Свести дрон над посадочным знаком «H» до вертикального спуска.

        Точку знака на земле (в СК дрона) кладёт в состояние поток перцепции. Здесь —
        П-регулятор: подвинуться на долю gain текущей ошибки, взять свежее наблюдение,
        повторить. Знак в СК дрона неподвижен, поэтому сходимость быстрая; gain<1 и
        предел шагов гасят раскачку от дрожания центра между кадрами.

        Ветвь целиком необязательная: выключенное центрирование, отсутствие камеры или
        незахваченный знак означают возврат к посадке вслепую, а не аварию.
        """
        if not self.landing_enabled:
            return
        параметры = self.landing
        # Спуск на высоту центрирования: знак крупнее в кадре и точнее ловится.
        # Держим x, y — их-то и будем доводить.
        позиция = self.flight.position()
        if позиция is not None:
            self.flight.goto(позиция[0], позиция[1], параметры["center_height"],
                             on_tick=self._tick)

        if not self._await_pad(параметры["acquire_timeout"]):
            self._log("[миссия] посадочный знак не найден за %.1f с — сажусь по "
                      "координатам возврата" % параметры["acquire_timeout"])
            return

        for шаг in range(1, параметры["max_steps"] + 1):
            цель = self._fresh_pad(параметры["fresh"])
            позиция = self.flight.position()
            if цель is None or позиция is None:
                self._log("[миссия] знак потерян на шаге %d — заканчиваю центрирование"
                          % шаг)
                break
            ошибка = math.hypot(цель[0] - позиция[0], цель[1] - позиция[1])
            if ошибка <= параметры["tol"]:
                self._log("[миссия] над знаком: ошибка %.2f м <= %.2f м, спускаюсь"
                          % (ошибка, параметры["tol"]))
                return
            # Шаг П-регулятора: доводим на долю ошибки, а не сразу в цель, — центр знака
            # между кадрами дрожит, и полный шаг раскачивал бы дрон вокруг площадки.
            цель_x = позиция[0] + параметры["gain"] * (цель[0] - позиция[0])
            цель_y = позиция[1] + параметры["gain"] * (цель[1] - позиция[1])
            self._log("[миссия] центрирование %d/%d: ошибка %.2f м -> правлю"
                      % (шаг, параметры["max_steps"], ошибка))
            self.flight.goto(цель_x, цель_y, параметры["center_height"],
                             timeout=параметры["step_timeout"], on_tick=self._tick)
        else:
            self._log("[миссия] предел шагов центрирования исчерпан — спускаюсь как есть")

    def _await_pad(self, timeout):
        """Дождаться первого свежего наблюдения знака. Во время ожидания продолжаем
        такты миссии: телеметрия идёт, аварийные условия проверяются."""
        дедлайн = time.time() + timeout
        while time.time() < дедлайн:
            if self._fresh_pad(self.landing["fresh"]) is not None:
                return True
            self._tick()
            time.sleep(0.1)
        return False

    def _fresh_pad(self, fresh):
        """Точка знака (x, y) в СК дрона, если наблюдение не старше fresh секунд, иначе
        None: устаревшее наблюдение неактуально — дрон с тех пор сместился."""
        s = self.state.snapshot()
        if s["pad"] is None:
            return None
        if time.time() - s["pad_ts"] > fresh:
            return None
        return s["pad"]

    # ------------------------------------------------------------------ выполнение

    def run(self):
        """Полный проход миссии. Блокирующий — запускать в отдельном потоке."""
        try:
            self._wait_command()
            self._arm()
            self._takeoff()
            self._survey()
            self._dust()
            self._return()
            self._land()
            self._log("[миссия] завершена штатно")
        except AbortMission as e:
            self._handle_abort(e)
        except Exception as e:
            self._log("[миссия] НЕОЖИДАННАЯ ОШИБКА: %s: %s" % (type(e).__name__, e))
            import traceback
            traceback.print_exc()
            self.state.update(state=ERROR, abort_reason="%s: %s" % (type(e).__name__, e))
            self._emergency_land()
        finally:
            self.state.update(finished=time.time())

    def _handle_abort(self, abort):
        self._log("[миссия] АВАРИЙНОЕ ЗАВЕРШЕНИЕ: %s (%s)" % (abort.reason, abort.severity))
        self.state.update(abort_reason=abort.reason)
        if abort.severity == "none":
            self.state.update(state=IDLE)
            return
        # Со «стоп-краном» больше не проверяем аварийные условия: повторный AbortMission
        # внутри аварийной посадки оставил бы дрон в воздухе.
        self._stop.set()
        if abort.severity == "return":
            try:
                self._set_state(RETURN)
                self.flight.goto(0.0, 0.0, self.h_survey, wait=True, on_tick=None)
            except Exception as e:
                self._log("[миссия] возврат не удался (%s: %s) — сажусь на месте"
                          % (type(e).__name__, e))
        self._emergency_land()

    def _emergency_land(self):
        try:
            self.state.update(state=LAND)
            self.flight.land()
            self.state.update(state=DISARM)
            self.flight.disarm()
        except Exception as e:
            self._log("[миссия] посадка не удалась: %s: %s" % (type(e).__name__, e))
