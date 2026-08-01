# -*- coding: utf-8 -*-
"""Калибровка камеры: интринсики, дисторсия и матрица установки R_mount.

Три величины, без которых координаты станций не сойдутся с рулеткой
(docs/DRONE_PLAN.md §2.2). Все три ИЗМЕРЯЮТСЯ, а не берутся из документации,
и лежат в config.yaml — в коде их быть не должно (регламент 2.5).

  fx, fy, cx, cy + дисторсия   tools/calibrate_camera.py, шахматка, 20-30 кадров.
                               15 минут и полигон не нужен.
  угол сервопривода            config.yaml -> camera.servo_angle. У нас -80:
                               ✅ проверено check_sdk.py, SDK отвергает -85 и -90.
  R_mount при этом угле        поправки camera.mount.* — механический перекос
                               крепления, измеряется ВМЕСТЕ с углом сервы.

⚠️ Надира не существует. Максимум сервопривода -80°, значит у камеры всегда остаётся
~10° наклона: на высоте 2 м это уводит центр кадра на ~35 см от точки под дроном —
больше всего порога «<30 см» (docs/lessons_from_archipelago.md §1.1). Само по себе
не страшно: постоянный наклон входит в R_mount и вычитается формулой. Страшно забыть
его туда положить, поэтому угол сервы и поправки лежат в конфиге рядом.

Модуль на стандартной библиотеке: его импортирует и заглушка камеры, которая должна
работать там, где нет ни cv2, ни NPU.
"""
from __future__ import annotations

import math

from localization import matmul, rot_x, rot_y, rot_z

# Сколько шагов делает обращение дисторсии. Пять хватает с запасом: модель Брауна-Конради
# на наших коэффициентах сходится за два-три.
_ШАГОВ_ОБРАЩЕНИЯ = 5

# Ориентация камеры при угле сервопривода 0: смотрит вперёд по курсу.
#   x камеры (вправо по кадру) -> -Y корпуса (вправо, ведь Y влево)
#   y камеры (вниз по кадру)   -> -Z корпуса (вниз)
#   z камеры (оптическая ось)  -> +X корпуса (вперёд)
_КАМЕРА_ВПЕРЁД = ((0.0, 0.0, 1.0),
                  (-1.0, 0.0, 0.0),
                  (0.0, -1.0, 0.0))


def mount_matrix(servo_angle, roll=0.0, pitch=0.0, yaw=0.0):
    """Поворот «камера -> корпус» при заданном угле сервопривода.

    servo_angle — то, что уходит в ServoCamera.set_angle: 0 — камера вперёд,
    отрицательные значения опускают её к земле, -90 был бы надиром (недостижим).

    roll/pitch/yaw — поправки на механический перекос крепления, в градусах, в осях
    корпуса. Их измеряют по кадру над известной точкой ИМЕННО при рабочем угле сервы.

    Проверка себя: при servo_angle=-90 и нулевых поправках получается надирная матрица
    ((0,-1,0), (-1,0,0), (0,0,-1)) — оптическая ось строго вниз, верх кадра вперёд.
    """
    # Опускание камеры на угол |servo| — это поворот вокруг оси Y корпуса (влево).
    # Знак минус: отрицательный servo_angle означает «вниз», а правый поворот
    # вокруг +Y уводит оптическую ось из +X в -Z при положительном угле.
    поворот_сервы = rot_y(-float(servo_angle))
    перекос = matmul(rot_z(yaw), matmul(rot_y(pitch), rot_x(roll)))
    return matmul(перекос, matmul(поворот_сервы, _КАМЕРА_ВПЕРЁД))


class Intrinsics(object):
    """Дырочная модель камеры плюс дисторсия Брауна-Конради.

    normalize() и project() — строго обратные друг другу с точностью обращения
    дисторсии. На этом держится сквозной тест «мок нарисовал -> детектор нашёл ->
    локализация вернула ту же точку».
    """

    def __init__(self, fx, fy, cx, cy, k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0,
                 measured=True):
        if fx <= 0 or fy <= 0:
            raise ValueError("фокусные расстояния должны быть положительными: %r, %r"
                             % (fx, fy))
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.p1 = float(p1)
        self.p2 = float(p2)
        self.k3 = float(k3)
        # False, если интринсики не измерены, а выведены из угла обзора. Флаг нужен,
        # чтобы «<30 см» не проверяли по оценке, приняв её за калибровку.
        self.measured = bool(measured)

    # ------------------------------------------------------------- дисторсия

    @property
    def has_distortion(self):
        return any(abs(k) > 1e-12 for k in (self.k1, self.k2, self.p1, self.p2, self.k3))

    def _distort(self, xn, yn):
        """Идеальные нормированные координаты -> искажённые объективом."""
        r2 = xn * xn + yn * yn
        радиальная = 1.0 + self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 * r2 * r2
        x = xn * радиальная + 2.0 * self.p1 * xn * yn + self.p2 * (r2 + 2.0 * xn * xn)
        y = yn * радиальная + self.p1 * (r2 + 2.0 * yn * yn) + 2.0 * self.p2 * xn * yn
        return x, y

    def _undistort(self, xd, yd):
        """Обращение дисторсии простой итерацией — тем же способом, что в OpenCV."""
        if not self.has_distortion:
            return xd, yd
        x, y = xd, yd
        for _ in range(_ШАГОВ_ОБРАЩЕНИЯ):
            вперёд_x, вперёд_y = self._distort(x, y)
            x += xd - вперёд_x
            y += yd - вперёд_y
        return x, y

    # --------------------------------------------------------------- перевод

    def normalize(self, u, v):
        """Пиксель -> идеальные нормированные координаты (x/z, y/z) в СК камеры."""
        return self._undistort((u - self.cx) / self.fx, (v - self.cy) / self.fy)

    def project(self, xn, yn):
        """Идеальные нормированные координаты -> пиксель (с дисторсией)."""
        xd, yd = self._distort(xn, yn)
        return self.cx + self.fx * xd, self.cy + self.fy * yd

    # ------------------------------------------------------------ производные

    def fov_deg(self, width, height):
        """Углы обзора (горизонтальный, вертикальный) в градусах.

        Закрывает «НЕ ПРОВЕРЕНО» у zone.fov_h_deg: после калибровки шаг змейки
        считается из измеренной оптики, а не из числа, взятого наугад.
        """
        return (2.0 * math.degrees(math.atan(width / (2.0 * self.fx))),
                2.0 * math.degrees(math.atan(height / (2.0 * self.fy))))

    @classmethod
    def from_fov(cls, fov_h_deg, width, height):
        """Грубая оценка по углу обзора: центр в середине кадра, дисторсии нет,
        пиксели квадратные. Только чтобы прогон на заглушке работал ДО калибровки."""
        if not 0.0 < fov_h_deg < 180.0:
            raise ValueError("угол обзора вне (0, 180): %r" % (fov_h_deg,))
        f = width / (2.0 * math.tan(math.radians(fov_h_deg) / 2.0))
        return cls(f, f, width / 2.0, height / 2.0, measured=False)

    def __repr__(self):
        return ("Intrinsics(fx=%.1f, fy=%.1f, cx=%.1f, cy=%.1f, %s)"
                % (self.fx, self.fy, self.cx, self.cy,
                   "измерены" if self.measured else "ОЦЕНКА по углу обзора"))


class Calibration(object):
    """Всё, что нужно локализации: интринсики, R_mount и размер кадра."""

    def __init__(self, intrinsics, r_mount, width, height, servo_angle,
                 attitude_roll_sign=1, attitude_pitch_sign=1):
        self.intrinsics = intrinsics
        self.r_mount = r_mount
        self.width = int(width)
        self.height = int(height)
        self.servo_angle = float(servo_angle)
        # Знаки углов get_orientation() на нашем борту НЕ ПРОВЕРЕНЫ (см. localization.py).
        self.attitude_roll_sign = int(attitude_roll_sign)
        self.attitude_pitch_sign = int(attitude_pitch_sign)

    def fix_attitude(self, roll, pitch, yaw):
        """Углы от Flight.orientation() -> углы в конвенции localization.py."""
        return (self.attitude_roll_sign * roll, self.attitude_pitch_sign * pitch, yaw)

    @property
    def measured(self):
        return self.intrinsics.measured

    def __repr__(self):
        return ("Calibration(%r, серва %.0f°, кадр %dx%d)"
                % (self.intrinsics, self.servo_angle, self.width, self.height))


def _число(раздел, ключ, по_умолчанию=None):
    """Значение из конфига или None. Пустой ключ и «НЕ ИЗМЕРЕНО» читаются как «нет»."""
    if not isinstance(раздел, dict):
        return по_умолчанию
    значение = раздел.get(ключ, по_умолчанию)
    if значение is None or isinstance(значение, bool):
        return по_умолчанию
    try:
        return float(значение)
    except (TypeError, ValueError):
        return по_умолчанию


def from_config(cfg, log=None):
    """Калибровка по config.yaml.

    Если интринсики не заполнены — работаем по оценке из zone.fov_h_deg и ГРОМКО
    об этом говорим. Молчаливая подстановка оценки означала бы, что на разборе попытки
    «координаты уехали» будут искать в детекторе, а причина в незаполненном конфиге.
    """
    log = log or (lambda text: print(text))
    camera = cfg.get("camera", {})
    width = int(_число(camera, "frame_width", 1080))
    height = int(_число(camera, "frame_height", 720))
    servo = _число(camera, "servo_angle", -80.0)

    intr_cfg = camera.get("intrinsics", {})
    fx = _число(intr_cfg, "fx")
    fy = _число(intr_cfg, "fy")
    if fx and fy:
        intrinsics = Intrinsics(
            fx, fy,
            _число(intr_cfg, "cx", width / 2.0),
            _число(intr_cfg, "cy", height / 2.0),
            k1=_число(intr_cfg, "k1", 0.0), k2=_число(intr_cfg, "k2", 0.0),
            p1=_число(intr_cfg, "p1", 0.0), p2=_число(intr_cfg, "p2", 0.0),
            k3=_число(intr_cfg, "k3", 0.0))
    else:
        fov = _число(cfg.get("zone", {}), "fov_h_deg", 60.0)
        intrinsics = Intrinsics.from_fov(fov, width, height)
        log("[калибровка] ВНИМАНИЕ: camera.intrinsics не заполнены, работаю по оценке "
            "из zone.fov_h_deg=%.0f°. Координаты станций будут смещены — прогнать "
            "tools/calibrate_camera.py до попытки на точность" % fov)

    mount = camera.get("mount", {})
    r_mount = mount_matrix(servo,
                           roll=_число(mount, "roll", 0.0),
                           pitch=_число(mount, "pitch", 0.0),
                           yaw=_число(mount, "yaw", 0.0))

    attitude = camera.get("attitude", {})
    return Calibration(intrinsics, r_mount, width, height, servo,
                       attitude_roll_sign=int(_число(attitude, "roll_sign", 1.0)),
                       attitude_pitch_sign=int(_число(attitude, "pitch_sign", 1.0)))
