# -*- coding: utf-8 -*-
"""Чтение и запись картинок по путям с кириллицей.

`cv2.imread` и `cv2.imwrite` на Windows отдают путь в C-runtime в кодировке ANSI.
Русские буквы в неё не переводятся, поэтому файл существует, а imread молча возвращает
None; imwrite так же молча ничего не пишет и возвращает False, который обычно никто
не проверяет. Исключения не бросается ни в одном из случаев.

Для нас это не редкий случай, а норма: русские имена директорий требует регламент 2.9
(`распределительный хаб`, `квадрокоптер`), а инструкции в RUN.md по-русски же называют
рабочие папки (`кадры/`, `разметка/`, `разбор/`). Плюс tools/replay.py пишет кадры
с русским ИМЕНЕМ файла — там ломается даже ASCII-папка.

На Linux (борт) проблемы нет, но код общий, поэтому путь один на обе системы.
Копия этих же четырёх строк живёт в `распределительный хаб/visualizer.py`: визуализатор
обязан оставаться самодостаточным.
"""
from __future__ import annotations

import os

try:
    import cv2
except ImportError:                      # pragma: no cover
    cv2 = None

try:
    import numpy as np
except ImportError:                      # pragma: no cover
    np = None


def imread(path, flags=None):
    """cv2.imread, понимающий кириллицу. None, если прочитать не вышло."""
    if cv2 is None:
        return None
    if np is None:                       # без numpy остаётся только штатный путь
        return cv2.imread(path) if flags is None else cv2.imread(path, flags)
    try:
        буфер = np.fromfile(path, dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if буфер.size == 0:
        return None
    return cv2.imdecode(буфер, cv2.IMREAD_COLOR if flags is None else flags)


def imwrite(path, image):
    """cv2.imwrite, понимающий кириллицу. True, если файл действительно записан.

    Расширение берётся из имени файла — оно же выбирает кодек, как и у imwrite.
    """
    if cv2 is None:
        return False
    if np is None:
        return bool(cv2.imwrite(path, image))
    расширение = os.path.splitext(path)[1] or ".png"
    ok, буфер = cv2.imencode(расширение, image)
    if not ok:
        return False
    try:
        буфер.tofile(path)
    except OSError:
        return False
    return True
