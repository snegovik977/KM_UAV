# -*- coding: utf-8 -*-
"""Заглушка pioneer_sdk2: весь автомат миссии крутится без дрона и без полигона.

Это не удобство, а экономия расходуемого ресурса. По регламенту 1.3 лимит времени
утекает, даже когда полигон простаивает, — значит отладка обязана происходить
оффлайн, а полигон тратится только на попытки.

Что заглушка умеет и зачем:
  - кинематика: дрон реально «летит» к точке с заданной скоростью, point_reached()
    становится истинным по прибытии. Так проверяются тайминги и порядок состояний;
  - разряд батареи: напряжение падает со временем, и аварийный возврат по низкому
    заряду можно прогнать не дожидаясь настоящей посадки на 15-й минуте;
  - отказы по требованию: зависший перелёт, срыв навигации, молчащая камера —
    ветки, которые в воздухе воспроизводить дорого и опасно.

Чего заглушка принципиально НЕ ловит: ошибку в осях. Она рапортует позицию в той же
системе координат, в которой приняла команду, поэтому любая конфигурация axes выглядит
исправной. Оси проверяются только полётом — tools/check_axes.py.

Сигнатуры собраны пересказом документации и раздатки организаторов. После
tools/check_sdk.py на борту этот файл нужно править по факту.

Управление через переменные окружения (чтобы не менять код ради одного прогона):
    PIONEER_MOCK=1                  включить заглушку (читает flight.create_pioneer)
    PIONEER_MOCK_BATT=6.5           стартовое напряжение, В
    PIONEER_MOCK_DRAIN=0.02         падение напряжения, В/с
    PIONEER_MOCK_STUCK=1            перелёты никогда не завершаются (проверка таймаутов)
    PIONEER_MOCK_NAV_FAIL=25        через N секунд полёта срывается оптопоток
    PIONEER_MOCK_VIDEO=путь.mp4     кадры из видеофайла вместо синтетики
    PIONEER_MOCK_STATIONS="1,1.5,ok; -1.2,.5,dust"   расстановка станций, метры, наша СК
    PIONEER_MOCK_STATION_SIZE=0.35x0.5               габариты станции, м
"""
from __future__ import annotations

import math
import os
import threading
import time

try:
    import numpy as np
except ImportError:                  # numpy есть и на борту, и локально, но падать не будем
    np = None

try:
    import cv2
except ImportError:
    cv2 = None


def _env_float(name, default):
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


class CameraType(object):
    MAIN = "main"
    # OPT намеренно объявлен, но открывать его нельзя: это камера оптического потока
    # позиционирования, второй потребитель сажает навигацию
    # (docs/lessons_from_archipelago.md §1.2).
    OPT = "opt"


class ServoPriority(object):
    LOW = 0
    HIGH = 1


class Event(object):
    ALL = "ALL"
    COPTER_LANDED = "COPTER_LANDED"
    TAKEOFF_COMPLETE = "TAKEOFF_COMPLETE"
    ENGINES_STARTED = "ENGINES_STARTED"
    POINT_REACHED = "POINT_REACHED"
    POINT_DECELERATION = "POINT_DECELERATION"
    LOW_CHARGE = "LOW_CHARGE"
    LOW_VOLTAGE1 = "LOW_VOLTAGE1"
    LOW_VOLTAGE2 = "LOW_VOLTAGE2"
    SHOCK = "SHOCK"


class _Перечисление(object):
    """Подобие enum из SDK: у настоящих значений есть .name, и код сравнивает по нему."""

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return "<%s: %d>" % (self.name, self.value)


class NavStatus(object):
    OK = _Перечисление("OK", 3)
    UNDEFINED = _Перечисление("UNDEFINED", 0)


class FlyState(object):
    LANDED = _Перечисление("LANDED", 0)
    IN_SKY = _Перечисление("IN_SKY", 2)


# Камере нужна поза дрона, чтобы рисовать правдоподобный кадр. Ссылка на последний
# созданный дрон — сокращение, допустимое только в заглушке: в настоящем SDK камера
# и автопилот друг о друге не знают.
_последний_дрон = None


class MockPioneer(object):
    """Кинематика по командам + телеметрия + отказы по требованию."""

    def __init__(self, tcp=None, wait_callback=True, safety_command=True, logger=True,
                 speed=0.45, batt=None, drain=None, stuck=None, nav_fail=None):
        global _последний_дрон
        self.wait_callback = wait_callback
        self.speed = float(speed)

        self.batt = _env_float("PIONEER_MOCK_BATT", 8.0) if batt is None else float(batt)
        self.drain = _env_float("PIONEER_MOCK_DRAIN", 0.004) if drain is None else float(drain)
        self.stuck = bool(os.environ.get("PIONEER_MOCK_STUCK")) if stuck is None else bool(stuck)
        self.nav_fail_after = (_env_float("PIONEER_MOCK_NAV_FAIL", 0.0)
                               if nav_fail is None else float(nav_fail))

        # Позиция в СК автопилота — той же, в которой приходят команды.
        self.position = [0.0, 0.0, 0.0]
        self.target = [0.0, 0.0, 0.0]
        self.yaw = 0.0
        self.target_yaw = 0.0

        self.state = "ON_LAND"           # ON_LAND -> ARMED -> IN_SKY -> ON_LAND
        self.armed = False
        self.journal = []                # что и в каком порядке вызывали — для тестов

        self._started = time.time()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._loop = threading.Thread(target=self._integrate, name="mock-kinematics",
                                      daemon=True)
        self._loop.start()
        _последний_дрон = self

    # ------------------------------------------------------------------ кинематика

    def _integrate(self, dt=0.05):
        while not self._stop.wait(dt):
            with self._lock:
                if self.state != "IN_SKY":
                    continue
                for i in range(3):
                    шаг = self.speed * dt
                    разница = self.target[i] - self.position[i]
                    if abs(разница) <= шаг:
                        self.position[i] = self.target[i]
                    else:
                        self.position[i] += math.copysign(шаг, разница)
                разница_курса = self.target_yaw - self.yaw
                шаг_курса = 90.0 * dt
                if abs(разница_курса) <= шаг_курса:
                    self.yaw = self.target_yaw
                else:
                    self.yaw += math.copysign(шаг_курса, разница_курса)

    def _журнал(self, событие):
        self.journal.append((round(time.time() - self._started, 2), событие))

    # ---------------------------------------------------------------- жизненный цикл

    def arm(self, timeout=5, retries=0):
        if self.state != "ON_LAND":
            raise RuntimeError("arm() из состояния %s" % self.state)
        self.state = "ARMED"
        self.armed = True
        # Ноль СК фиксируется именно здесь, а не в takeoff().
        with self._lock:
            self.position = [0.0, 0.0, 0.0]
            self.target = [0.0, 0.0, 0.0]
        self._журнал("arm")
        return True

    def takeoff(self):
        if self.state != "ARMED":
            raise RuntimeError("takeoff() из состояния %s" % self.state)
        self.state = "IN_SKY"
        with self._lock:
            self.target = [0.0, 0.0, 1.0]
        self._журнал("takeoff")
        return True

    def land(self):
        with self._lock:
            self.target = [self.position[0], self.position[1], 0.0]
        # Ждём фактического снижения, но не бесконечно: настоящая land() тоже
        # возвращает управление.
        deadline = time.time() + 8.0
        while time.time() < deadline and self.position[2] > 0.02:
            time.sleep(0.05)
        with self._lock:
            self.position[2] = 0.0
        self.state = "ON_LAND"
        self._журнал("land")
        return True

    def rtl(self):
        with self._lock:
            self.target = [0.0, 0.0, max(1.0, self.position[2])]
        self._журнал("rtl")
        return True

    def disarm(self):
        self.armed = False
        self.state = "ON_LAND"
        self._журнал("disarm")
        return True

    # ⚠️ Метода is_landed() здесь намеренно НЕТ: на реальном борту его тоже нет
    # (проверено tools/check_sdk.py 31.07.2026), хотя он встречается в примерах
    # документации. Заглушка обязана врать так же, как железо, иначе тесты зелёные,
    # а на дроне AttributeError в аварийной посадке. Вместо него — get_fly_state().

    def get_fly_state(self):
        return FlyState.LANDED if self.state == "ON_LAND" else FlyState.IN_SKY

    def close_connection(self):
        self._stop.set()
        self._журнал("close_connection")
        return True

    # ---------------------------------------------------------------- перемещение

    def go_to_local_point(self, x, y, z, yaw=0.0, time_=0):
        if self.state != "IN_SKY":
            raise RuntimeError("go_to_local_point() из состояния %s" % self.state)
        with self._lock:
            self.target = [float(x), float(y), float(z)]
            self.target_yaw = float(yaw)
        self._журнал("goto(%.2f, %.2f, %.2f)" % (x, y, z))
        return True

    def go_to_local_point_body_fixed(self, x, y, z, yaw=0.0, time_=0):
        if self.state != "IN_SKY":
            raise RuntimeError("go_to_local_point_body_fixed() из состояния %s" % self.state)
        with self._lock:
            self.target = [self.position[0] + float(x),
                           self.position[1] + float(y),
                           self.position[2] + float(z)]
            self.target_yaw = self.yaw + float(yaw)
        self._журнал("goto_body(%.2f, %.2f, %.2f)" % (x, y, z))
        return True

    def point_reached(self):
        if self.stuck:
            return False        # проверка таймаута перелёта
        with self._lock:
            расстояние = max(abs(a - b) for a, b in zip(self.position, self.target))
        return расстояние < 0.05

    def set_yaw(self, yaw):
        with self._lock:
            self.target_yaw = float(yaw)
        return True

    # ---------------------------------------------------------------- телеметрия

    def get_local_position_lps(self):
        with self._lock:
            return tuple(self.position)

    def get_orientation(self):
        return (0.0, 0.0, self.yaw)

    def get_altitude(self):
        return self.position[2]

    def get_battery_status(self):
        # Разряд идёт всегда, а не только в полёте: батарея садится и на земле,
        # пока идёт настройка перед попыткой.
        напряжение = self.batt - self.drain * (time.time() - self._started)
        return (round(max(напряжение, 5.0), 3), 25.0)

    def get_dist_sensor_data(self):
        """Нижний дальномер. На нашем борту работает именно он."""
        return round(self.position[2], 3)

    def get_ranger_data(self):
        # ⚠️ На нашем борту модуль Ranger не установлен и метод отдаёт пять None
        # (проверено check_sdk.py 31.07.2026). Заглушка повторяет это поведение,
        # чтобы код был обязан уметь падать обратно на get_dist_sensor_data().
        return (None, None, None, None, None)

    def get_nav_status_lps(self):
        if self.nav_fail_after and (time.time() - self._started) > self.nav_fail_after:
            return NavStatus.UNDEFINED      # срыв оптопотока над однотонным полом
        return NavStatus.OK

    def get_nav_system(self, update=False):
        return "LPS"

    def subscribe(self, callback, event):
        return True

    def unsubscribe(self, callback, event):
        return True


# Расстановка станций по умолчанию: (x, y, статус) в НАШЕЙ СК, метры от точки взлёта.
# Это декорация тестового стенда, а не константа алгоритма: в боевом коде таких
# координат быть не может (регламент 2.5). Точки лежат вдоль контура из config.yaml,
# чтобы облёт по умолчанию реально проносил их под камерой.
СТАНЦИИ_ПО_УМОЛЧАНИЮ = ((1.2, 0.8, "ok"), (1.2, -0.8, "dust"),
                        (-1.4, 0.9, "broken"), (-1.4, -0.7, "ok"))

# Габариты станции. ✅ ИЗМЕРЕНЫ рулеткой на площадке 01.08.2026: сверху 29 x 21 см.
# Длинная сторона положена вдоль Y — ориентация станции на полигоне произвольная,
# а заглушке нужен известный длинный бок, по нему проверяется угол в детекторе.
# Пороги детектора в config.yaml обязаны бракетировать это значение, а не совпадать
# с ним: заглушка рисует идеальный прямоугольник, настоящий контур будет грязнее.
РАЗМЕР_СТАНЦИИ = (0.21, 0.29)      # (вдоль X, вдоль Y), м = 0.061 м²

ЯРКОСТЬ_ПОЛА = 205                 # светлый пол, панель на нём — тёмное пятно

# Как выглядит каждый статус. Признаки те же, по которым его определяет
# perception/status.py (docs/DRONE_PLAN.md §3.2): пыль поднимает яркость и убивает
# видимость линий ячеек, поломка рвёт контур.
ВИД_СТАНЦИИ = {
    "ok":     {"цвет": (62, 54, 46), "ячейки": True, "разлом": False},
    "dust":   {"цвет": (150, 152, 150), "ячейки": False, "разлом": False},
    "broken": {"цвет": (58, 52, 48), "ячейки": True, "разлом": True},
}


def _разобрать_станции(текст):
    """"x,y,статус; x,y,статус" -> список кортежей. Кривую запись пропускаем с шумом."""
    станции = []
    for кусок in текст.replace("\n", ";").split(";"):
        кусок = кусок.strip()
        if not кусок:
            continue
        части = [ч.strip() for ч in кусок.split(",")]
        try:
            x, y = float(части[0]), float(части[1])
        except (IndexError, ValueError):
            print("[mock] не понял описание станции %r — пропускаю" % кусок)
            continue
        статус = части[2] if len(части) > 2 and части[2] in ВИД_СТАНЦИИ else "ok"
        станции.append((x, y, статус))
    return станции


class MockCamera(object):
    """Кадры без камеры: видеофайл, если он задан, иначе синтетика по позе дрона.

    Синтетика рисуется ТОЙ ЖЕ дырочной моделью, которой локализация считает координаты
    (localization.ground_to_pixel), только обратной, и с интринсиками из config.yaml.
    Это принципиально: так сквозной прогон «мок -> детектор -> локализация -> реестр»
    проверяет прямую формулу против обратной и меряет фактическую ошибку в метрах —
    то есть критерий «<30 см» подзадачи 2.3.2 проверяется без полигона.

    Чего заглушка по-прежнему НЕ ловит: неверные интринсики и R_mount. Она рисует теми же
    числами, которыми потом считает, поэтому любая калибровка выглядит идеальной.
    Настоящую проверку даёт только рулетка под дроном.
    """

    # Фактическое разрешение борта: get_cv_frame() отдаёт (720, 1080, 3), uint8, BGR
    # (проверено check_sdk.py 31.07.2026).
    def __init__(self, camera_type=CameraType.MAIN, width=1080, height=720, fps=15.0,
                 stations=None, cfg=None):
        if camera_type == CameraType.OPT:
            raise RuntimeError("CameraType.OPT открывать нельзя: это камера оптического "
                               "потока позиционирования")
        self.width = width
        self.height = height
        self.fps = fps
        self.stopped = False
        self.frames = 0
        self._video = None

        описание = os.environ.get("PIONEER_MOCK_STATIONS")
        if stations is not None:
            self.stations = list(stations)
        elif описание:
            self.stations = _разобрать_станции(описание)
        else:
            self.stations = list(СТАНЦИИ_ПО_УМОЛЧАНИЮ)
        self.station_size = self._размер_станции()

        self._калибровка = None
        self._оси = None
        self._настроить_оптику(cfg)

        путь = os.environ.get("PIONEER_MOCK_VIDEO")
        if путь and cv2 is not None:
            self._video = cv2.VideoCapture(путь)
            if not self._video.isOpened():
                print("[mock] видео %s не открылось — работаю на синтетике" % путь)
                self._video = None

    @staticmethod
    def _размер_станции():
        текст = os.environ.get("PIONEER_MOCK_STATION_SIZE")
        if not текст:
            return РАЗМЕР_СТАНЦИИ
        try:
            вдоль_x, вдоль_y = (float(ч) for ч in текст.lower().split("x"))
            return (вдоль_x, вдоль_y)
        except ValueError:
            print("[mock] не понял PIONEER_MOCK_STATION_SIZE=%r, беру %s"
                  % (текст, РАЗМЕР_СТАНЦИИ))
            return РАЗМЕР_СТАНЦИИ

    def _настроить_оптику(self, cfg):
        """Калибровка и адаптер осей из config.yaml. Не вышло — рисуем без станций:
        заглушка не имеет права ронять прогон, ради которого она существует."""
        try:
            import config as config_module
            from flight import AxisAdapter
            from perception.calib import from_config as калибровка_из_конфига

            cfg = cfg if cfg is not None else config_module.load()
            self._калибровка = калибровка_из_конфига(cfg, log=lambda _: None)
            self._оси = AxisAdapter.from_config(cfg)
        except Exception as e:
            print("[mock] оптика не настроена (%s: %s) — станции рисоваться не будут"
                  % (type(e).__name__, e))

    def get_cv_frame(self, timeout=5.0):
        if self.stopped:
            raise RuntimeError("камера остановлена")
        time.sleep(1.0 / max(self.fps, 1.0))
        self.frames += 1
        if self._video is not None:
            ok, frame = self._video.read()
            if not ok:
                self._video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._video.read()
            if ok:
                return frame
        return self._синтетика()

    # ------------------------------------------------------------------ синтетика

    def pose(self):
        """Поза дрона в НАШЕЙ СК. Заглушка автопилота хранит её в СК автопилота —
        в той же, в которой приняла команду, — поэтому пересчёт обязателен."""
        from localization import Pose

        дрон = _последний_дрон
        if дрон is None or self._оси is None:
            return None
        x, y, z = self._оси.from_sdk(*дрон.get_local_position_lps())
        yaw = self._оси.yaw_from_sdk(дрон.get_orientation()[2])
        return Pose(x, y, max(z, 0.05), 0.0, 0.0, yaw)

    def _синтетика(self):
        if np is None:
            return None
        frame = np.full((self.height, self.width, 3), ЯРКОСТЬ_ПОЛА, dtype=np.uint8)
        поза = self.pose()
        if поза is None or cv2 is None or self._калибровка is None:
            return frame

        for x, y, статус in self.stations:
            self._нарисовать_станцию(frame, поза, x, y, статус)

        # Подписи на кадре намеренно НЕТ: у настоящей камеры её тоже нет, а детектор
        # честно ловил бы тёмные буквы как объект. Поза и состояние миссии наносятся
        # поверх позже, в main.py, уже после инференса.
        return frame

    def _в_пиксели(self, поза, точки):
        """Точки земли -> пиксели. None, если хоть одна не проецируется."""
        from localization import ground_to_pixel

        калибровка = self._калибровка
        готово = []
        for x, y in точки:
            пиксель = ground_to_pixel(x, y, поза, калибровка.intrinsics,
                                      калибровка.r_mount)
            if пиксель is None:
                return None
            готово.append((int(round(пиксель[0])), int(round(пиксель[1]))))
        return готово

    def _нарисовать_станцию(self, frame, поза, cx, cy, статус):
        вид = ВИД_СТАНЦИИ.get(статус, ВИД_СТАНЦИИ["ok"])
        половина_x, половина_y = self.station_size[0] / 2.0, self.station_size[1] / 2.0
        углы = self._в_пиксели(поза, [
            (cx - половина_x, cy - половина_y), (cx + половина_x, cy - половина_y),
            (cx + половина_x, cy + половина_y), (cx - половина_x, cy + половина_y)])
        if углы is None:
            return
        # Целиком за кадром — не тратим время на отрисовку.
        запас = max(self.width, self.height)
        if all(u < -запас or u > self.width + запас or
               v < -запас or v > self.height + запас for u, v in углы):
            return

        cv2.fillPoly(frame, [np.array(углы, dtype=np.int32)], вид["цвет"])

        if вид["ячейки"]:
            # Линии между фотоэлементами.
            # ⚠️ Заглушка рисует «пыль» как посветление всей панели. По фотографиям
            # с площадки (01.08.2026, `фото станции/`) это НЕВЕРНО: пыль изображается
            # цветными телами, положенными на панель, см. docs/DRONE_PLAN.md §3.2.
            # Для подзадачи 2.3.2 разницы нет — станцией считаются и чистые, и грязные,
            # а классов классический детектор не различает. Переделывать отрисовку
            # придётся под 2.3.3, вместе с perception/status.py.
            светлее = tuple(min(255, c + 45) for c in вид["цвет"])
            for доля in (0.25, 0.5, 0.75):
                y_линии = cy - половина_y + 2 * половина_y * доля
                отрезок = self._в_пиксели(поза, [(cx - половина_x, y_линии),
                                                 (cx + половина_x, y_линии)])
                if отрезок:
                    cv2.line(frame, отрезок[0], отрезок[1], светлее, 1, cv2.LINE_AA)

        if вид["разлом"]:
            # Неисправная станция: угол выломан, контур перестаёт быть прямоугольным.
            выломанный = self._в_пиксели(поза, [
                (cx + половина_x, cy + половина_y),
                (cx + половина_x - половина_x, cy + половина_y),
                (cx + половина_x, cy + половина_y - половина_y)])
            if выломанный:
                cv2.fillPoly(frame, [np.array(выломанный, dtype=np.int32)],
                             (ЯРКОСТЬ_ПОЛА, ЯРКОСТЬ_ПОЛА, ЯРКОСТЬ_ПОЛА))

    def stop(self):
        self.stopped = True
        if self._video is not None:
            self._video.release()
        return True


class MockImageViewer(object):
    """Вместо HTTP-трансляции — счётчик кадров: логика main.py от этого не зависит."""

    def __init__(self):
        self.shown = 0
        self.last = None
        self.closed = False

    def imshow(self, name=None, frame=None, fps=30):
        self.shown += 1
        self.last = frame
        return True

    def close(self):
        self.closed = True
        return True


class MockServoCamera(object):
    """Диапазон углов взят по полевому опыту чужой команды: -80…+30, надира нет.

    Заглушка отвергает -90 намеренно: если код где-то полагается на надир, это должно
    всплыть здесь, а не на полигоне при разборе, почему координаты уехали на 35 см.
    """

    MIN_ANGLE = -80
    MAX_ANGLE = 30

    def __init__(self):
        self.angle = 0

    def set_angle(self, angle=-45, priority=None):
        if not self.MIN_ANGLE <= angle <= self.MAX_ANGLE:
            raise ValueError("угол %s вне диапазона %d…%d"
                             % (angle, self.MIN_ANGLE, self.MAX_ANGLE))
        self.angle = angle
        return True


# Имена как в настоящем SDK: main.py импортирует одно и то же, не зная, где он работает.
Pioneer = MockPioneer
Camera = MockCamera
ImageViewer = MockImageViewer
ServoCamera = MockServoCamera
