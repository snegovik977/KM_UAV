# -*- coding: utf-8 -*-
"""Пути к модулям проекта.

Директории названы по регламенту 2.9 («распределительный хаб», «квадрокоптер») —
пакетами Python они быть не могут, поэтому путь до них добавляется здесь, а не
через setup.py. Ровно то же делает квадрокоптер/main.py на борту.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for sub in ("распределительный хаб", "квадрокоптер", "tools"):
    path = os.path.join(ROOT, sub)
    if path not in sys.path:
        sys.path.insert(0, path)
