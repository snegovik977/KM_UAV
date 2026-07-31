"""
Классификация цвета кораблика по кропу bbox: registered (зелёный) / unregistered (оранжевый).

ВАЖНО про домен (замерено на реальных кадрах дрона):
  - Вода бассейна бирюзовая: RGB ~ [70,129,139] -> G > R, но B > G (b-g ~ +10).
  - Пол серо-бирюзовый, тоже G > R.
  - Голубой КОРПУС лодки: B >= G > R (самый яркий канал — синий).
  - Оранжевый АКЦЕНТ: R доминирует (R > G > B).
  - Зелёный АКЦЕНТ: G доминирует И G > B  (именно G>B отличает его от воды/пола, где B>=G).

Поэтому классифицируем по АКЦЕНТНЫМ пикселям, а не по всему корпусу.
Работаем в BGR (соглашение OpenCV).
"""
from __future__ import annotations
import numpy as np

# Пороги подобраны по замерам на кадрах; вынесены для калибровки.
ORANGE_MIN_R = 60      # оранжевый достаточно яркий
ORANGE_R_OVER_G = 12   # R заметно больше G
ORANGE_R_OVER_B = 22   # R заметно больше B
GREEN_MIN_G = 45
GREEN_G_OVER_R = 8     # G больше R (у воды тоже, поэтому мало)
GREEN_G_OVER_B = 4     # G больше B  <-- ключевой признак против воды (у воды B>G)
GREEN_MAX_G = 205      # отсечь пересвеченные блики
MIN_ACCENT_FRAC = 0.010  # минимум акцентных пикселей от площади кропа, иначе "цвет не определён"


def color_scores(crop_bgr: np.ndarray) -> tuple[int, int]:
    """Возвращает (orange_px, green_px) — число оранжевых и зелёных акцентных пикселей."""
    if crop_bgr is None or crop_bgr.size == 0:
        return 0, 0
    b = crop_bgr[..., 0].astype(np.int16)
    g = crop_bgr[..., 1].astype(np.int16)
    r = crop_bgr[..., 2].astype(np.int16)

    orange = (r > ORANGE_MIN_R) & (r - g > ORANGE_R_OVER_G) & (r - b > ORANGE_R_OVER_B)
    green = (
        (g > GREEN_MIN_G) & (g < GREEN_MAX_G)
        & (g - r > GREEN_G_OVER_R) & (g - b > GREEN_G_OVER_B)
    )
    return int(orange.sum()), int(green.sum())


def classify_color(crop_bgr: np.ndarray) -> tuple[str | None, float]:
    """
    Классифицирует кроп -> ('registered'|'unregistered'|None, confidence 0..1).
    None — цвет не определён уверенно (мало акцентных пикселей): кандидат в hard-negative.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None, 0.0
    area = crop_bgr.shape[0] * crop_bgr.shape[1]
    o, gr = color_scores(crop_bgr)
    total = o + gr
    if area == 0 or total < MIN_ACCENT_FRAC * area or total == 0:
        return None, 0.0
    if o >= gr:
        return "unregistered", o / total
    return "registered", gr / total
