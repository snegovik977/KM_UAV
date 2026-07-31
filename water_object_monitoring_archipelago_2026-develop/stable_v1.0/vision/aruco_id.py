"""
Чтение ArUco-метки на носу кораблика -> уникальный ID (+5 за каждый, всего +30).

Метки крепят организаторы позже (~3x3 см), словарь ⚠️ не подтверждён (предположительно 4x4).
Модуль готов заранее: детектим маркеры на кадре и сопоставляем каждый ближайшему боксу.

ВНИМАНИЕ (риск из pipeline.md): метка 3 см с высоты облёта, вероятно, НЕ читается —
ID придётся снимать снижением/зависанием над лодкой. Этот модуль — только чтение,
логику снижения решает управление полётом.
"""
from __future__ import annotations
import cv2
import numpy as np

# ✅ подтверждено регламентом (обновление 26.07.2026): ArUco-метки 4x4 - 1000
DEFAULT_DICT = cv2.aruco.DICT_4X4_1000


class ArucoReader:
    def __init__(self, dictionary: int = DEFAULT_DICT):
        self._dict = cv2.aruco.getPredefinedDictionary(dictionary)
        self._params = cv2.aruco.DetectorParameters()
        # Послабления под МЕЛКИЕ метки (с высоты метка 3 см — считанные пиксели).
        # Проверено на фото меток оргов: ловит 4x4 от ~16 px на сторону.
        self._params.minMarkerPerimeterRate = 0.01
        self._params.adaptiveThreshWinSizeMin = 3
        self._params.adaptiveThreshWinSizeMax = 53
        self._params.adaptiveThreshWinSizeStep = 6
        self._detector = cv2.aruco.ArucoDetector(self._dict, self._params)

    def detect(self, frame_bgr: np.ndarray):
        """-> list[(marker_id:int, center:(x,y))]"""
        corners, ids, _ = self._detector.detectMarkers(frame_bgr)
        out = []
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                cx, cy = c[0].mean(axis=0)
                out.append((int(i), (float(cx), float(cy))))
        return out

    @staticmethod
    def match_to_bbox(markers, bbox_xyxy):
        """Вернуть marker_id, чей центр лежит внутри bbox (или None)."""
        x1, y1, x2, y2 = bbox_xyxy
        for mid, (cx, cy) in markers:
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return mid
        return None
