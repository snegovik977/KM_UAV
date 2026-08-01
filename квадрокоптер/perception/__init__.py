# -*- coding: utf-8 -*-
"""Перцепция: калибровка камеры и детекция станций.

Пакет собран так, чтобы его импорт никогда не ронял полёт: cv2 и numpy тянет только
detector.py, и делает это мягко. calib.py — чистая математика на стандартной библиотеке,
он нужен ещё и заглушке камеры, которая работает там, где ни модели, ни NPU нет.
"""
from __future__ import annotations

from .calib import Calibration, Intrinsics, mount_matrix   # noqa: F401
from .detector import Detection, create_detector           # noqa: F401

__all__ = ["Calibration", "Intrinsics", "mount_matrix", "Detection", "create_detector"]
