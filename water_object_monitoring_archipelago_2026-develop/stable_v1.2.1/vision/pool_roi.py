"""
Сегментация зоны бассейна (ROI) для подавления ложных срабатываний.

Настоящие кораблики всегда ВНУТРИ бассейна. Оранжевая метка старт/стоп, жерди,
стыки каркаса, пол — вне воды. Отсекая детекции, чей центр не в маске бассейна,
убираем главные источники ложных (критерий стабильности +20, штраф -10).

Оптимизация: маска считается на УМЕНЬШЕННОМ кадре (downscale) — в разы дешевле по CPU
(важно для борта). center_in_mask/pool_coverage работают в координатах маски.
"""
from __future__ import annotations
import cv2
import numpy as np


def pool_mask(frame_bgr: np.ndarray, margin_px: int = -12, downscale: int = 3) -> np.ndarray:
    """Бинарная маска бассейна (uint8 0/255) на разрешении frame/downscale.

    margin_px (в координатах ПОЛНОГО кадра) >0 расширяет, <0 сужает внутрь воды.
    """
    h, w = frame_bgr.shape[:2]
    small = cv2.resize(frame_bgr, (max(1, w // downscale), max(1, h // downscale)))
    b = small[..., 0].astype(np.int16)
    r = small[..., 2].astype(np.int16)
    water = ((b - r > 22) & (b > 80)).astype(np.uint8) * 255

    ks = max(3, 15 // downscale) | 1                      # нечётный размер ядра
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    water = cv2.morphologyEx(water, cv2.MORPH_CLOSE, k, iterations=2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(water, connectivity=8)
    if n <= 1:
        return np.zeros(small.shape[:2], np.uint8)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == biggest).astype(np.uint8) * 255

    ff = mask.copy()
    mh, mw = mask.shape
    m2 = np.zeros((mh + 2, mw + 2), np.uint8)
    cv2.floodFill(ff, m2, (0, 0), 255)
    mask = mask | cv2.bitwise_not(ff)

    m = margin_px // downscale
    if m != 0:
        kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(m) * 2 + 1,) * 2)
        mask = cv2.dilate(mask, kk) if m > 0 else cv2.erode(mask, kk)
    return mask


def center_in_mask(bbox_xyxy, mask: np.ndarray, downscale: int = 3) -> bool:
    """True, если центр bbox (координаты полного кадра) попадает в маску (уменьшенную)."""
    x1, y1, x2, y2 = bbox_xyxy
    cx = int((x1 + x2) / 2 / downscale)
    cy = int((y1 + y2) / 2 / downscale)
    h, w = mask.shape
    if not (0 <= cx < w and 0 <= cy < h):
        return False
    return mask[cy, cx] > 0


def pool_coverage(mask: np.ndarray) -> float:
    """Доля кадра, занятая бассейном (маска уже уменьшена — доля та же)."""
    return float((mask > 0).mean())
