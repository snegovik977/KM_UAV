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

# Регламент называет 4x4-1000, НО он часто мисдекодит мелкие/смазанные метки в
# случайные высокие id (542 и т.п.) -> фантомный переучёт. Метки оргов (id<=36)
# декодятся ОДИНАКОВО в 4x4_50 и 4x4_1000, а 4x4_50 (50 кодов, большое расстояние)
# почти не мисдекодит. Поэтому берём 4x4_50. Если реальные метки не читаются —
# вернуть DICT_4X4_1000 и полагаться на whitelist id + голосование в pipeline.
DEFAULT_DICT = cv2.aruco.DICT_4X4_50


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
        """-> list[(marker_id:int, center:(x,y))]  (детект по всему кадру — дорого)."""
        corners, ids, _ = self._detector.detectMarkers(frame_bgr)
        out = []
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                cx, cy = c[0].mean(axis=0)
                out.append((int(i), (float(cx), float(cy))))
        return out

    def read_bbox(self, frame_bgr: np.ndarray, bbox_xyxy, pad: int = 8):
        """Читает ArUco ВНУТРИ бокса лодки (маленький кроп). -> marker_id | None.

        Дёшево по CPU (детект на кропе, а не на полном кадре) и без фантомных меток
        из сетки/плитки — смотрим только там, где судно.
        """
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        _, ids, _ = self._detector.detectMarkers(crop)
        if ids is not None and len(ids) > 0:
            return int(ids[0][0])
        return None

    @staticmethod
    def match_to_bbox(markers, bbox_xyxy):
        """Вернуть marker_id, чей центр лежит внутри bbox (или None)."""
        x1, y1, x2, y2 = bbox_xyxy
        for mid, (cx, cy) in markers:
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return mid
        return None
