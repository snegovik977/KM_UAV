# -*- coding: utf-8 -*-
"""Конвейер перцепции: кадр -> детекции -> координаты -> реестр -> пакеты.

Живёт в ГЛАВНОМ потоке (main.py), пока полётная логика крутится в отдельном.
Причина в docs/DRONE_PLAN.md §1.3: Pioneer(wait_callback=True) делает полётные команды
блокирующими, и гони мы миссию из главного потока — инференс вставал бы на каждом
перелёте между точками, а станции, пролетевшие под дроном, просто не попадали бы в кадр.

Позу дрона конвейер берёт из снимка MissionState под замком и в SDK не лезет вовсе:
к автопилоту обращается только полётный поток.

Три правила из чужого полевого опыта (docs/lessons_from_archipelago.md §3), которые
здесь реализованы буквально:
  - перцепция не имеет права ронять полёт: каждый шаг в try/except, ошибки считаются
    и печатаются с прореживанием, а не глушатся;
  - дешёвое и дорогое разделены по частоте: инференс раз в every_n кадров, отрисовка
    и HUD каждый кадр — трансляция обязана оставаться плавной;
  - зависание — со sleep, а не busy-loop (это уже в main.py).
"""
from __future__ import annotations

from localization import (Pose, ground_area, in_central_region, offset_from_nadir,
                          pixel_to_ground)
from mission import SURVEY, to_world

try:
    import cv2
except ImportError:                      # pragma: no cover — на борту cv2 есть
    cv2 = None

# Цвет рамки по статусу, BGR. Те же цвета у визуализатора: оператор смотрит на два
# экрана сразу, и «жёлтое» на одном не должно означать другое на втором.
ЦВЕТ_СТАТУСА = {
    "ok": (80, 220, 80),
    "dust": (0, 165, 255),
    "broken": (60, 60, 235),
    None: (200, 200, 200),
}


class Perception(object):
    """Один кадр -> наблюдения в реестре. Ничего не знает ни про SDK, ни про сеть."""

    def __init__(self, cfg, state, registry, calibration, detector, log=None,
                 only_in_survey=True):
        self.state = state
        self.registry = registry
        self.calibration = calibration
        self.detector = detector
        self._log = log or (lambda text: print(text))

        раздел = cfg.get("detector", {})
        self.every_n = max(1, int(раздел.get("every_n", 3)))
        self.central_frac = float(раздел.get("central_frac", 0.5))
        # Наблюдения принимаются только во время облёта. На взлёте, возврате и посадке
        # дрон видит те же станции под другими углами и с разгоном — это лишний шум
        # в медиане; а на площадке в кадре ещё и посадочный знак.
        self.only_in_survey = bool(only_in_survey)

        self.frames = 0
        self.inferences = 0
        self.detections = 0
        self.errors = 0
        self.last = []                   # детекции последнего инференса — для отрисовки

    # ------------------------------------------------------------------- один кадр

    def pose(self, snapshot=None):
        """Поза дрона в его собственной СК. None, если высота ещё нулевая."""
        s = snapshot if snapshot is not None else self.state.snapshot()
        x, y, z = s["position"]
        roll, pitch, yaw = self.calibration.fix_attitude(
            s.get("roll", 0.0), s.get("pitch", 0.0), s.get("yaw", 0.0))
        if z <= 0.0:
            return None
        return Pose(x, y, z, roll, pitch, yaw)

    def should_infer(self, snapshot):
        if self.only_in_survey and snapshot["state"] != SURVEY:
            return False
        return self.frames % self.every_n == 0

    def process(self, frame):
        """Обработать кадр. Возвращает список детекций (пустой, если инференс пропущен).

        Никогда не бросает исключений: перцепция не имеет права ронять полёт.
        """
        self.frames += 1
        if frame is None:
            return []
        snapshot = self.state.snapshot()
        if not self.should_infer(snapshot):
            return self.last

        try:
            поза = self.pose(snapshot)
            if поза is None:
                return []
            детекции = self.detector.detect(frame, area_m2=self._площадь(поза))
            self.inferences += 1
            self.last = детекции
            self._записать(детекции, поза, snapshot, frame.shape)
            return детекции
        except Exception as e:
            self.errors += 1
            if self.errors % 30 == 1:
                self._log("[перцепция] инференс упал (%d-я ошибка): %s: %s"
                          % (self.errors, type(e).__name__, e))
            return []

    def _площадь(self, поза):
        def считать(углы):
            return ground_area(углы, поза, self.calibration.intrinsics,
                               self.calibration.r_mount)
        return считать

    def _записать(self, детекции, поза, snapshot, shape):
        """Детекции -> мировые координаты -> реестр."""
        высота, ширина = shape[:2]
        origin = snapshot["origin"]
        for детекция in детекции:
            точка = pixel_to_ground(детекция.cx, детекция.cy, поза,
                                    self.calibration.intrinsics,
                                    self.calibration.r_mount)
            if точка is None:
                continue                 # луч не встретил землю — не станция
            self.detections += 1
            # В реестр координаты идут уже в ОБЩЕЙ СК роя, тем же пересчётом, что
            # и телеметрия: иначе дрон сообщал бы своё место в одной СК, а станции
            # в другой, и автомобиль строил бы маршрут не туда.
            x, y = to_world(точка[0], точка[1], origin)
            self.registry.observe(
                x, y, детекция.label, детекция.score,
                precise=in_central_region(детекция.cx, детекция.cy, ширина, высота,
                                          self.central_frac))

    # -------------------------------------------------------------------- отрисовка

    def draw(self, frame, детекции=None):
        """Рамки станций и граница центральной области поверх кадра трансляции.

        Дешёвая операция, поэтому идёт каждый кадр, а не раз в every_n: оператор
        смотрит на живую картинку, и мигающие рамки читаются хуже, чем застывшие.
        """
        if frame is None or cv2 is None:
            return frame
        try:
            высота, ширина = frame.shape[:2]
            поле = int(ширина * (1.0 - self.central_frac) / 2.0)
            верх = int(высота * (1.0 - self.central_frac) / 2.0)
            cv2.rectangle(frame, (поле, верх), (ширина - поле, высота - верх),
                          (120, 120, 120), 1)

            for детекция in (детекции if детекции is not None else self.last):
                x1, y1, x2, y2 = детекция.bbox
                цвет = ЦВЕТ_СТАТУСА.get(детекция.label, ЦВЕТ_СТАТУСА[None])
                cv2.rectangle(frame, (x1, y1), (x2, y2), цвет, 2)
                подпись = "%s %.2f" % (детекция.label or "?", детекция.score)
                cv2.putText(frame, подпись, (x1, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, цвет, 1, cv2.LINE_AA)
        except Exception:
            pass                         # отрисовка тем более не имеет права мешать
        return frame

    def hud(self):
        """Хвост строки состояния: что видит перцепция прямо сейчас."""
        return "det=%d станций=%d" % (len(self.last), self.registry.confirmed_count())

    def summary(self):
        return ("[перцепция] кадров %d, инференсов %d, детекций %d, ошибок %d"
                % (self.frames, self.inferences, self.detections, self.errors))

    def check(self, h_survey):
        """Разовая проверка при старте: что скажет геометрия на рабочей высоте.

        Смысл — увидеть забытую калибровку НА ЗЕМЛЕ, до вылета. Смещение центра кадра
        от надира при серве -80 составляет ~35 см на 2 м, и если оно вдруг нулевое,
        значит R_mount не тот, что нужен.
        """
        поза = Pose(0.0, 0.0, float(h_survey), 0.0, 0.0, 0.0)
        снос = offset_from_nadir(поза, self.calibration.intrinsics,
                                 self.calibration.r_mount)
        self._log("[перцепция] на высоте %.1f м центр кадра смещён от надира на %s"
                  % (h_survey, "?" if снос is None else "%.2f м" % снос))
        if not self.calibration.measured:
            self._log("[перцепция] ВНИМАНИЕ: интринсики не измерены — координаты станций "
                      "будут смещены, «<30 см» по ним проверять нельзя")
