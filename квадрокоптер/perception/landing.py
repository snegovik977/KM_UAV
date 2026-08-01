# -*- coding: utf-8 -*-
"""Наведение на посадочную площадку «H»: детектор знака и мост в полётную логику.

Зачем отдельно от detector.py. Детектор станций ищет пятно, НЕПОХОЖЕЕ НА ФОН, — его
порог адаптивен и намеренно нестрог, потому что заранее неизвестно, как станция
выглядит на покрытии полигона. Посадочный знак известен точно: насыщенно-жёлтый круг
с чёрно-жёлтой каймой и буквой «H» (см. landing_site.jpg). Его надёжнее искать по цвету,
а не по контрасту, и мешать эти два поиска в одном классе — значит подгонять пороги
станции под знак и наоборот.

Работает для ЛЮБОЙ подзадачи, включая 2.3.1, где детектора станций нет вовсе:
центрирование перед посадкой — часть базового полёта (2 балла за «сел в пределах
площадки»), а не разведки. Поэтому калибровку наведение при необходимости поднимает
своей копией, не завися от того, создан ли конвейер станций.

Разделение обязанностей:
  LandingPadDetector  кадр -> PadDetection(центр в пикселях, радиус, уверенность) | None.
                      Чистый CV, пороги из config.yaml -> landing.*.
  LandingGuide        детектор + калибровка. update() берёт кадр и позу дрона, находит
                      знак, проецирует его центр на землю (localization.pixel_to_ground)
                      и кладёт точку в MissionState. Полётный поток читает её оттуда и
                      сводит дрон над знаком — так же, как поток перцепции и полётный
                      поток обмениваются позой, без второго обращения к SDK.

cv2/numpy импортируются мягко: перцепция не имеет права ронять полёт, а состав пакетов
выданной машины заранее неизвестен (регламент 1.4). Нет их — наведение просто не
поднимается, и дрон садится вслепую по координатам возврата.
"""
from __future__ import annotations

import math
import time
from collections import namedtuple

from localization import Pose, pixel_to_ground

try:
    import cv2
except ImportError:                      # pragma: no cover — на борту cv2 есть
    cv2 = None

try:
    import numpy as np
except ImportError:                      # pragma: no cover
    np = None

# cx, cy — центр знака в пикселях кадра, идёт в локализацию;
# radius — радиус описанной окружности жёлтого пятна, пиксели (для отрисовки и оценки);
# score  — заполненность описанной окружности жёлтым: круг знака её заполняет, редкие
#          жёлтые крапины — нет. Служит и порогом отсева, и уверенностью для оператора.
PadDetection = namedtuple("PadDetection", "cx cy radius score")


class LandingPadDetector(object):
    """Жёлтый круг посадочного знака в кадре. Один знак на площадке — берём крупнейший.

    Порядок: BGR -> HSV -> маска жёлтого -> морфологическая чистка -> контуры ->
    отсев мелочи -> описанная окружность по всем значимым точкам. Центр окружности и
    есть центр знака: кайма, кольцо и буква «H» симметричны, и минимальная описанная
    окружность их множества центрируется на знаке даже когда буква и кольцо в маске
    разорваны (между ними серая заливка).
    """

    def __init__(self, hue_lo=18, hue_hi=40, sat_min=70, val_min=70, close_px=7,
                 min_area_px=300, min_fill=0.12, log=None):
        if cv2 is None or np is None:
            raise RuntimeError("детектору посадочного знака нужны cv2 и numpy")
        self.hue_lo = int(hue_lo)
        self.hue_hi = int(hue_hi)
        self.sat_min = int(sat_min)
        self.val_min = int(val_min)
        self.close_px = int(close_px)
        self.min_area_px = int(min_area_px)
        self.min_fill = float(min_fill)
        self._log = log or (lambda text: print(text))

    def _маска(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        нижняя = np.array([self.hue_lo, self.sat_min, self.val_min], dtype=np.uint8)
        верхняя = np.array([self.hue_hi, 255, 255], dtype=np.uint8)
        маска = cv2.inRange(hsv, нижняя, верхняя)
        if self.close_px > 0:
            ядро = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.close_px, self.close_px))
            маска = cv2.morphologyEx(маска, cv2.MORPH_CLOSE, ядро)
        return маска

    def detect(self, frame):
        """Кадр BGR -> PadDetection или None. Исключений не бросает: наведение не
        имеет права ронять полёт."""
        if frame is None or cv2 is None or np is None:
            return None
        try:
            маска = self._маска(frame)
            контуры = cv2.findContours(маска, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)[-2]
            значимые = [c for c in контуры if cv2.contourArea(c) >= self.min_area_px]
            if not значимые:
                return None
            точки = np.vstack(значимые)
            площадь = float(sum(cv2.contourArea(c) for c in значимые))
            (cx, cy), радиус = cv2.minEnclosingCircle(точки)
            if радиус <= 0.0:
                return None
            # Заполненность описанной окружности: круг знака (с каймой и «H») закрывает
            # заметную её долю, а разбросанные жёлтые точки — нет, даже уложившись в неё.
            заполнение = площадь / (math.pi * радиус * радиус)
            if заполнение < self.min_fill:
                return None
            return PadDetection(float(cx), float(cy), float(радиус), float(заполнение))
        except Exception as e:
            self._log("[наведение] детектор знака упал: %s: %s" % (type(e).__name__, e))
            return None


class LandingGuide(object):
    """Мост «кадр -> точка знака на земле в MissionState».

    Живёт в главном (перцепционном) потоке рядом с конвейером станций и вызывается
    из главного цикла на фазах возврата и посадки. Центр знака проецируется на землю
    в СОБСТВЕННОЙ СК дрона (без to_world): полётный поток командует goto в той же СК,
    и лишний перевод в общую СК роя только внёс бы ошибку origin.
    """

    def __init__(self, detector, calibration, state, phases, log=None):
        self.detector = detector
        self.calibration = calibration
        self.state = state
        # На каких состояниях миссии искать знак. RETURN — чтобы к моменту LAND точка
        # уже была захвачена и центрирование не ждало первого кадра.
        self.phases = tuple(phases)
        self._log = log or (lambda text: print(text))
        self.last = None                 # последняя детекция — для отрисовки HUD
        self.hits = 0
        self.misses = 0

    def _pose(self, snapshot):
        x, y, z = snapshot["position"]
        if z <= 0.0:
            return None
        roll, pitch, yaw = self.calibration.fix_attitude(
            snapshot.get("roll", 0.0), snapshot.get("pitch", 0.0),
            snapshot.get("yaw", 0.0))
        return Pose(x, y, z, roll, pitch, yaw)

    def active(self, snapshot):
        return snapshot["state"] in self.phases

    def update(self, frame):
        """Найти знак на кадре и, если найден, записать его точку на земле в состояние.

        Возвращает PadDetection или None. Никогда не бросает исключений.
        """
        snapshot = self.state.snapshot()
        if not self.active(snapshot):
            return None
        поза = self._pose(snapshot)
        if поза is None:
            return None
        детекция = self.detector.detect(frame)
        self.last = детекция
        if детекция is None:
            self.misses += 1
            return None
        точка = pixel_to_ground(детекция.cx, детекция.cy, поза,
                                self.calibration.intrinsics, self.calibration.r_mount)
        if точка is None:
            self.misses += 1               # луч на знак не встретил землю — не доверяем
            return None
        self.hits += 1
        self.state.update(pad=(точка[0], точка[1]), pad_ts=time.time(),
                          pad_score=детекция.score)
        return детекция

    def draw(self, frame):
        """Кружок вокруг знака поверх кадра трансляции. Дёшево, поэтому каждый кадр."""
        if frame is None or cv2 is None or self.last is None:
            return frame
        try:
            центр = (int(self.last.cx), int(self.last.cy))
            cv2.circle(frame, центр, int(self.last.radius), (0, 255, 255), 2)
            cv2.drawMarker(frame, центр, (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
        except Exception:
            pass
        return frame

    def hud(self):
        if self.last is None:
            return "знак:—"
        return "знак %.0f%%" % (100.0 * self.last.score)


def from_config(cfg, state, calibration=None, log=None):
    """Собрать LandingGuide по config.yaml. None, если центрирование выключено или
    не на чем работать (нет cv2/numpy, не поднялась калибровка).

    calibration передаётся, если конвейер станций её уже загрузил, — чтобы не читать
    интринсики и не печатать предупреждение о некалиброванной камере дважды.
    """
    log = log or (lambda text: print(text))
    раздел = cfg.get("landing", {})
    if not раздел or not раздел.get("enabled", False):
        return None
    if cv2 is None or np is None:
        log("[наведение] нет cv2/numpy — центрирование по площадке отключено, "
            "посадка пойдёт по координатам возврата")
        return None
    try:
        детектор = LandingPadDetector(
            hue_lo=раздел.get("hue_lo", 18), hue_hi=раздел.get("hue_hi", 40),
            sat_min=раздел.get("sat_min", 70), val_min=раздел.get("val_min", 70),
            close_px=раздел.get("close_px", 7),
            min_area_px=раздел.get("min_area_px", 300),
            min_fill=раздел.get("min_fill", 0.12), log=log)
        if calibration is None:
            from perception.calib import from_config as калибровка_из_конфига
            calibration = калибровка_из_конфига(cfg, log=log)
        from mission import LAND, RETURN
        return LandingGuide(детектор, calibration, state, phases=(RETURN, LAND), log=log)
    except Exception as e:
        log("[наведение] не поднялось (%s: %s) — посадка по координатам возврата"
            % (type(e).__name__, e))
        return None
