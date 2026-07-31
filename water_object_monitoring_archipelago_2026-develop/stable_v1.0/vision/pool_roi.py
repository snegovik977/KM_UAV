"""
Сегментация зоны бассейна (ROI) для подавления ложных срабатываний.

Идея: настоящие кораблики всегда ВНУТРИ бассейна. Оранжевая метка старт/стоп,
бамбуковые жерди, белые стыки каркаса, пол — вне воды. Отсекая детекции, чей центр
не попадает в маску бассейна, убираем главные источники ложных срабатываний
(критерий стабильности +20, штраф -10 за каждое ложное).

Вода бассейна бирюзовая: B - R > ~22 при B > ~80 (замерено). Маску заливаем,
берём крупнейшую компоненту и заполняем дыры (лодки/блики внутри — часть ROI).
"""
from __future__ import annotations
import cv2
import numpy as np


def pool_mask(frame_bgr: np.ndarray, margin_px: int = -12) -> np.ndarray:
    """Бинарная маска зоны бассейна (uint8 0/255).

    margin_px>0 расширяет маску, <0 сужает ВНУТРЬ воды. По умолчанию сужаем: так
    отсекаются бежевые углы/стыки каркаса на кромке (частый источник ложных).
    """
    b = frame_bgr[..., 0].astype(np.int16)
    r = frame_bgr[..., 2].astype(np.int16)
    water = ((b - r > 22) & (b > 80)).astype(np.uint8) * 255

    # Сомкнуть разрывы (сетка/блики дробят воду), затем крупнейшая компонента
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    water = cv2.morphologyEx(water, cv2.MORPH_CLOSE, k, iterations=2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(water, connectivity=8)
    if n <= 1:
        return np.zeros(frame_bgr.shape[:2], np.uint8)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == biggest).astype(np.uint8) * 255

    # Заполнить дыры (лодки/блики/пятна внутри бассейна = часть ROI)
    ff = mask.copy()
    h, w = mask.shape
    m2 = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, m2, (0, 0), 255)
    mask = mask | cv2.bitwise_not(ff)

    if margin_px > 0:
        kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin_px * 2 + 1,) * 2)
        mask = cv2.dilate(mask, kk)
    return mask


def center_in_mask(bbox_xyxy, mask: np.ndarray) -> bool:
    """True, если центр bbox попадает в маску бассейна."""
    x1, y1, x2, y2 = bbox_xyxy
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    h, w = mask.shape
    if not (0 <= cx < w and 0 <= cy < h):
        return False
    return mask[cy, cx] > 0


def pool_coverage(frame_bgr: np.ndarray, mask: np.ndarray) -> float:
    """Доля кадра, занятая бассейном — полезно как гейт «бассейн в кадре»."""
    return float((mask > 0).mean())
