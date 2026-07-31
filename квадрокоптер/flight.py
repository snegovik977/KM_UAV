# -*- coding: utf-8 -*-
"""Обёртка над pioneer_sdk2.Pioneer: единственное место, где живут оси автопилота.

Три причины, по которым весь остальной код не имеет права звать SDK напрямую:

1. СИСТЕМА КООРДИНАТ. Раздатка организаторов: «ось y направлена вперёд, ось x — вправо»
   (docs/organizer_handouts.md §3.1), сайт Geoscan подразумевает обратное. Ошибка
   не падает с исключением — она разворачивает змейку на 90°, и это видно только
   в воздухе. Поэтому наружу выставляется наша правая СК (X вперёд, Y влево, Z вверх),
   а пересчёт делает AxisAdapter по параметрам config.yaml -> axes. После tools/check_axes.py
   разворот траектории правится одной строкой конфига, а не переписыванием кода.

2. ЕДИНИЦЫ YAW. PDF организаторов противоречит сам себе: «в радианах» в комментарии
   и «в градусах» в карточке метода. Наружу — всегда градусы.

3. МИНИМАЛЬНАЯ ВЫСОТА. h_min проверяется здесь, а не в dust.py: защита от касания
   панели должна работать для любой команды, откуда бы она ни пришла. Сломанное
   оборудование по регламенту не заменяется, а время попытки при поломке не
   останавливается.

Плюс два расхождения источников по именам методов прячутся здесь же (range_down,
position): пробуем оба варианта, запоминаем работающий.
"""
from __future__ import annotations

import math
import os
import time


def wrap_deg(angle):
    """Угол в (-180, 180]. Без этого разница курсов около ±180° даёт скачок на 360°."""
    angle = math.fmod(angle + 180.0, 360.0)
    if angle <= 0:
        angle += 360.0
    return angle - 180.0


class AxisAdapter:
    """Пересчёт «наша СК <-> СК автопилота».

    Наша СК: X вперёд, Y влево, Z вверх (правая, человеческая).
    СК автопилота: определяется тремя флагами конфига, потому что достоверно она
    неизвестна до полёта.

        swap_xy: true   ->  sdk_x = sign_x * our_y,  sdk_y = sign_y * our_x
        swap_xy: false  ->  sdk_x = sign_x * our_x,  sdk_y = sign_y * our_y

    Вариант из раздатки организаторов (SDK: Y вперёд, X вправо) — это
    swap_xy=true, sign_x=-1, sign_y=1.
    """

    def __init__(self, swap_xy=True, sign_x=-1, sign_y=1, yaw_units="deg", yaw_sign=1):
        if sign_x not in (1, -1) or sign_y not in (1, -1):
            raise ValueError("sign_x/sign_y должны быть 1 или -1, получено %r/%r"
                             % (sign_x, sign_y))
        if yaw_units not in ("deg", "rad"):
            raise ValueError("yaw_units должно быть deg или rad, получено %r" % (yaw_units,))
        self.swap_xy = bool(swap_xy)
        self.sign_x = int(sign_x)
        self.sign_y = int(sign_y)
        self.yaw_units = yaw_units
        self.yaw_sign = int(yaw_sign)

    @classmethod
    def from_config(cls, cfg):
        axes = cfg["axes"]
        return cls(swap_xy=axes["swap_xy"], sign_x=axes["sign_x"], sign_y=axes["sign_y"],
                   yaw_units=axes["yaw_units"], yaw_sign=axes.get("yaw_sign", 1))

    def to_sdk(self, x, y, z):
        if self.swap_xy:
            return self.sign_x * y, self.sign_y * x, z
        return self.sign_x * x, self.sign_y * y, z

    def from_sdk(self, sdk_x, sdk_y, sdk_z):
        # Знаки — только ±1, поэтому обратное преобразование это то же умножение.
        if self.swap_xy:
            return self.sign_y * sdk_y, self.sign_x * sdk_x, sdk_z
        return self.sign_x * sdk_x, self.sign_y * sdk_y, sdk_z

    def yaw_to_sdk(self, yaw_deg):
        yaw = self.yaw_sign * yaw_deg
        return math.radians(yaw) if self.yaw_units == "rad" else yaw

    def yaw_from_sdk(self, yaw_sdk):
        yaw = math.degrees(yaw_sdk) if self.yaw_units == "rad" else yaw_sdk
        return self.yaw_sign * yaw


class Flight:
    """Полётные команды в нашей СК, в градусах, с жёстким полом по высоте."""

    def __init__(self, pioneer, cfg, log=None):
        self.pioneer = pioneer
        self.cfg = cfg
        self.axes = AxisAdapter.from_config(cfg)
        self._log = log or (lambda text: print(text))

        flight = cfg["flight"]
        self.h_min = float(flight["h_min"])
        self.h_max = float(flight["h_max"])
        self.speed = float(flight["speed"])
        self.point_timeout = float(flight["point_timeout"])

        self.yaw0 = 0.0                 # курс в момент arm(): начало нашей СК по углу
        self.clamped = 0                # сколько раз сработал пол по высоте
        self._position_call = None      # какой вариант get_local_position_lps работает
        self._range_call = None         # get_dist_sensor_data или get_ranger_data
        self._last_target = None

    # ------------------------------------------------------------- жизненный цикл

    def arm(self):
        """Запуск моторов. Здесь автопилот фиксирует ноль своей локальной СК —
        не в takeoff(), как можно было бы подумать (docs/organizer_handouts.md §3.1)."""
        self.pioneer.arm()
        time.sleep(float(self.cfg["flight"]["arm_settle"]))
        # Курс в момент arm() — это направление нашей оси X. Дальше все углы
        # отдаются относительно него, иначе телеметрия дрона и карта автомобиля
        # окажутся повёрнуты друг относительно друга.
        raw = self._raw_orientation()
        self.yaw0 = raw[2] if raw else 0.0
        self._log("[flight] arm выполнен, курс старта yaw0=%.1f" % self.yaw0)
        return True

    def takeoff(self):
        self.pioneer.takeoff()
        time.sleep(float(self.cfg["flight"]["takeoff_settle"]))
        self._log("[flight] takeoff выполнен")
        return True

    def land(self):
        self._log("[flight] посадка")
        return self.pioneer.land()

    def rtl(self):
        self._log("[flight] возврат в точку взлёта (rtl)")
        return self.pioneer.rtl()

    def disarm(self):
        return self.pioneer.disarm()

    # Высота, ниже которой считаем дрон стоящим на земле, если ничего лучше нет.
    LANDED_HEIGHT = 0.25

    def is_landed(self):
        """Стоит ли дрон на земле.

        ⚠️ На нашем борту метода `is_landed()` у Pioneer **НЕТ** — проверено
        tools/check_sdk.py 31.07.2026, хотя он есть в примерах документации. Поэтому
        цепочка: сам метод -> состояние полёта -> высота по дальномеру -> высота по LPS.

        Ошибаться безопаснее в сторону «не на земле»: лишняя команда land() ничего
        не ломает, а пропущенная означает disarm в воздухе, то есть падение.
        """
        fn = getattr(self.pioneer, "is_landed", None)
        if fn is not None:
            try:
                return bool(fn())
            except Exception:
                pass

        fly_state = getattr(self.pioneer, "get_fly_state", None)
        if fly_state is not None:
            try:
                name = str(getattr(fly_state(), "name", fly_state())).upper()
                if "LAND" in name or "GROUND" in name:
                    return True
                if "SKY" in name or "FLY" in name or "FLIGHT" in name or "AIR" in name:
                    return False
            except Exception:
                pass

        height = self.range_down()
        if height is None:
            position = self.position()
            height = position[2] if position else None
        if height is None:
            return False
        return height <= self.LANDED_HEIGHT

    def close(self):
        try:
            self.pioneer.close_connection()
        except Exception as e:
            self._log("[flight] close_connection: %s: %s" % (type(e).__name__, e))

    # -------------------------------------------------------------- перемещение

    def _guard_z(self, z):
        """Пол и потолок по высоте.

        Команда подрезается, а не отбрасывается: отброшенная команда оставит дрон
        на прежней высоте, и пошаговый спуск в подзадаче 2.3.4 зациклится, ничего
        не сообщив. Подрезанная — доводит дрон ровно до безопасной границы.
        """
        if z < self.h_min:
            self.clamped += 1
            self._log("[flight] ПОЛ: запрошено z=%.2f, подрезано до h_min=%.2f (случай %d)"
                      % (z, self.h_min, self.clamped))
            return self.h_min
        if z > self.h_max:
            self._log("[flight] ПОТОЛОК: запрошено z=%.2f, подрезано до h_max=%.2f"
                      % (z, self.h_max))
            return self.h_max
        return z

    def goto(self, x, y, z, yaw=0.0, wait=True, timeout=None, on_tick=None):
        """Перелёт в точку нашей СК. Возвращает True, если точка достигнута.

        on_tick вызывается пока дрон летит (~10 Гц) — через него миссия обновляет
        телеметрию и проверяет аварийные условия, не заводя второго потока к SDK.
        """
        z = self._guard_z(z)
        sdk_x, sdk_y, sdk_z = self.axes.to_sdk(x, y, z)
        sdk_yaw = self.axes.yaw_to_sdk(yaw)
        self._last_target = (x, y, z)
        try:
            self.pioneer.go_to_local_point(sdk_x, sdk_y, sdk_z, sdk_yaw)
        except Exception as e:
            self._log("[flight] go_to_local_point не принят: %s: %s" % (type(e).__name__, e))
            return False
        if not wait:
            return True
        return self.wait_point(timeout if timeout is not None else self.point_timeout,
                               on_tick=on_tick)

    def goto_body(self, dx, dy, dz, dyaw=0.0, wait=True, timeout=None, on_tick=None):
        """Смещение относительно текущего положения дрона — основа П-регуляторов
        центрирования (доводка посадки, наведение на панель)."""
        dz_guarded = dz
        position = self.position()
        if position is not None:
            # Пол считаем по абсолютной высоте: относительная команда сама по себе
            # безопасной высоты не знает.
            target_z = position[2] + dz
            dz_guarded = self._guard_z(target_z) - position[2]
        sdk_dx, sdk_dy, sdk_dz = self.axes.to_sdk(dx, dy, dz_guarded)
        sdk_yaw = self.axes.yaw_to_sdk(dyaw)
        try:
            self.pioneer.go_to_local_point_body_fixed(sdk_dx, sdk_dy, sdk_dz, sdk_yaw)
        except Exception as e:
            self._log("[flight] go_to_local_point_body_fixed не принят: %s: %s"
                      % (type(e).__name__, e))
            return False
        if not wait:
            return True
        return self.wait_point(timeout if timeout is not None else self.point_timeout,
                               on_tick=on_tick)

    def wait_point(self, timeout, on_tick=None):
        """Ожидание прихода в точку опросом.

        point_reached() есть в примере организаторов, но не значился в выжимке с сайта.
        Если метода нет — ждём по оценке времени из расстояния и скорости. Это хуже,
        но не «пролететь 3 секунды вперёд»: оценка считается от фактической геометрии,
        а не подгоняется под расстановку (регламент 2.5).
        """
        fn = getattr(self.pioneer, "point_reached", None)
        deadline = time.time() + timeout
        if fn is None:
            return self._sleep_ticking(min(timeout, self._estimate_time()), on_tick)
        while time.time() < deadline:
            try:
                if fn():
                    self._sleep_ticking(float(self.cfg["flight"]["settle"]), on_tick)
                    return True
            except Exception as e:
                self._log("[flight] point_reached: %s: %s — жду по оценке времени"
                          % (type(e).__name__, e))
                self._sleep_ticking(
                    min(max(0.0, deadline - time.time()), self._estimate_time()), on_tick)
                return True
            if on_tick is not None:
                on_tick()
            time.sleep(0.05)
        self._log("[flight] ТАЙМАУТ прихода в точку %r за %.1f с" % (self._last_target, timeout))
        return False

    def _sleep_ticking(self, duration, on_tick):
        """Пауза, во время которой миссия продолжает следить за аварийными условиями."""
        deadline = time.time() + duration
        while time.time() < deadline:
            if on_tick is not None:
                on_tick()
            time.sleep(min(0.1, max(0.0, deadline - time.time())))
        return True

    def _estimate_time(self):
        position = self.position()
        if position is None or self._last_target is None:
            return 2.0
        dx = self._last_target[0] - position[0]
        dy = self._last_target[1] - position[1]
        dz = self._last_target[2] - position[2]
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        return min(self.point_timeout, distance / max(self.speed, 0.05) + 1.0)

    # --------------------------------------------------------------- телеметрия

    def position(self):
        """Позиция в нашей СК или None.

        get_local_position_lps в выжимке с сайта — без аргументов, а пример организаторов
        зовёт его с позиционным True (принудительный опрос вместо кэша). Пробуем оба
        варианта один раз и запоминаем работающий.
        """
        raw = self._raw_position()
        if raw is None:
            return None
        try:
            return self.axes.from_sdk(float(raw[0]), float(raw[1]), float(raw[2]))
        except (TypeError, ValueError, IndexError):
            return None

    def _raw_position(self):
        variants = [lambda: self.pioneer.get_local_position_lps(),
                    lambda: self.pioneer.get_local_position_lps(True)]
        if self._position_call is not None:
            variants = [self._position_call]
        for call in variants:
            try:
                value = call()
            except TypeError:
                continue
            except Exception:
                return None
            if value is not None:
                self._position_call = call
                return value
        return None

    def orientation(self):
        """(roll, pitch, yaw) в градусах; yaw — относительно курса на момент arm()."""
        raw = self._raw_orientation()
        if raw is None:
            return None
        roll, pitch, yaw = (self.axes.yaw_from_sdk(v) for v in raw[:3])
        yaw0 = self.axes.yaw_from_sdk(self.yaw0)
        return roll, pitch, wrap_deg(yaw - yaw0)

    def _raw_orientation(self):
        try:
            value = self.pioneer.get_orientation()
        except Exception:
            return None
        if value is None or len(value) < 3:
            return None
        return value

    def battery(self):
        """Напряжение батареи в вольтах или None. get_battery_status() отдаёт
        кортеж (напряжение, температура)."""
        try:
            value = self.pioneer.get_battery_status()
        except Exception:
            return None
        if value is None:
            return None
        try:
            return float(value[0]) if isinstance(value, (tuple, list)) else float(value)
        except (TypeError, ValueError):
            return None

    def range_down(self):
        """Высота по нижнему дальномеру — независимый от оптопотока контроль.

        Имя метода расходится: раздатка организаторов даёт get_dist_sensor_data()
        (одно число), сайт Geoscan — get_ranger_data() (пять дальностей, вертикальная
        последняя). Пробуем оба, запоминаем работающий.
        """
        def from_dist():
            value = self.pioneer.get_dist_sensor_data()
            return None if value is None else float(value)

        def from_ranger():
            value = self.pioneer.get_ranger_data()
            if value is None:
                return None
            return float(value[-1]) if isinstance(value, (tuple, list)) else float(value)

        variants = [from_dist, from_ranger]
        if self._range_call is not None:
            variants = [self._range_call]
        for call in variants:
            try:
                value = call()
            except (AttributeError, TypeError):
                continue
            except Exception:
                return None
            if value is not None:
                self._range_call = call
                return value
        return None

    # Значения NavStatus, при которых навигация считается исправной. На нашем борту
    # в покое отдаётся NavStatus.OK (проверено check_sdk.py 31.07.2026). Остальные
    # имена этого перечисления нам пока не встречались — незнакомое значение считается
    # отказом, но реакция наступает только на устойчивой потере (см. mission.py).
    NAV_OK_NAMES = ("OK", "GOOD", "ACTIVE", "READY")

    def nav_ok(self):
        """Жив ли оптопоток. Срыв над однотонным полом уводит позицию, и это надо
        поймать раньше, чем координаты станций уедут за порог 30 см."""
        try:
            status = self.pioneer.get_nav_status_lps()
        except Exception:
            return True     # метод не работает — не повод прерывать миссию
        if status is None:
            return True
        if isinstance(status, bool):
            return status

        # get_nav_status_lps отдаёт перечисление NavStatus. Сравниваем по имени,
        # а не по числу: числовые значения могут поменяться между версиями SDK.
        name = getattr(status, "name", None)
        if name is not None:
            ok = name.upper() in self.NAV_OK_NAMES
            if not ok:
                self._log("[flight] навигация: %s" % (status,))
            return ok

        if isinstance(status, (int, float)):
            return status > 0
        return True


def create_pioneer(log=None):
    """Настоящий SDK или заглушка — по переменной окружения PIONEER_MOCK.

    Ветвление живёт здесь, чтобы main.py не знал о существовании мока: на борту
    и на ноутбуке запускается один и тот же код.
    """
    log = log or (lambda text: print(text))
    if os.environ.get("PIONEER_MOCK"):
        from mock_pioneer import MockPioneer
        log("[flight] PIONEER_MOCK=1 — работаем на заглушке, полёта не будет")
        return MockPioneer()
    from pioneer_sdk2 import Pioneer
    return Pioneer()
