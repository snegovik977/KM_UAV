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

# Те же строки, что в протоколе (protocol/messages.py -> MISSION_STATES): состояние
# уходит в телеметрию, и расхождение здесь означает отвергнутый пакет.
IDLE, ARM, TAKEOFF, SURVEY, DUST, RETURN, LAND, DISARM, ERROR = (
    "IDLE", "ARM", "TAKEOFF", "SURVEY", "DUST", "RETURN", "LAND", "DISARM", "ERROR")


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
        self.yaw = 0.0
        self.battery = 0.0
        self.origin = (0.0, 0.0, 0.0)      # начало общей СК роя из пакета takeoff
        self.waypoint = 0
        self.waypoints_total = 0
        self.stations = 0                  # пункт 2: сколько станций подтверждено
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
                "yaw": self.yaw,
                "battery": self.battery,
                "origin": self.origin,
                "waypoint": self.waypoint,
                "waypoints_total": self.waypoints_total,
                "stations": self.stations,
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

    def __init__(self, flight, transport, cfg, state=None, log=None, factory=None):
        self.flight = flight
        self.transport = transport
        self.cfg = cfg
        self.state = state or MissionState()
        self._log = log or (lambda text: print(text))
        self.factory = factory              # protocol.MessageFactory, может быть None

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
            поля["yaw"] = orientation[2]
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
        """Из СК дрона в общую СК роя.

        В подзадачах 2.3.x начало общей СК совпадает с точкой взлёта, и origin нулевой.
        В финале дрон стартует с автомобиля, и момент команды takeoff физически задаёт
        общее начало отсчёта обеим машинам — тогда сдвиг и поворот приходят в пакете.
        """
        ox, oy, oyaw = self.state.snapshot()["origin"]
        угол = math.radians(oyaw)
        return (ox + x * math.cos(угол) - y * math.sin(угол),
                oy + x * math.sin(угол) + y * math.cos(угол))

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
        """Точка расширения пункта 2: здесь уходит recon_done с числом станций.

        Пакет обязан уйти строго после всей змейки, иначе автомобиль начнёт строить
        маршрут по неполной карте.
        """
        if self.transport is None or self.factory is None:
            return
        количество = self.state.snapshot()["stations"]
        if количество:
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

        Пока садимся по координатам точки взлёта. Доводка по зрению (детекция
        площадки «H» + П-регулятор) — пункт 1.5 плана, встраивается сюда же, когда
        появится перцепция.
        """
        self._set_state(LAND)
        self.flight.land()
        self._set_state(DISARM)
        self.flight.disarm()

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
