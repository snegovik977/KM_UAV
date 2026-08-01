# -*- coding: utf-8 -*-
"""Вызовы RecordWriter из инструментов существуют на самом деле.

Родилось из настоящей поломки: `tools/handheld.py --record` звал `recorder.event(...)`,
которого у RecordWriter нет, и падал с AttributeError на первом же кадре. Заметить это
можно было только с дроном в руках — а именно этим инструментом снимается датасет
для обучения модели (docs/YOLO_TRAINING.md §2), то есть отказ приходился ровно
на съёмку, ради которой едут на площадку.

Проверка статическая (разбор AST), потому что запустить handheld.py в тесте нельзя:
он держит камеру, поднимает трансляцию и работает до Ctrl+C.
"""
from __future__ import annotations

import ast
import os

import pytest

from record import RecordWriter

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Файл -> имя переменной, в которой лежит RecordWriter.
ИНСТРУМЕНТЫ = (
    (os.path.join(КОРЕНЬ, "tools", "handheld.py"), "recorder"),
    (os.path.join(КОРЕНЬ, "квадрокоптер", "main.py"), "recorder"),
)


def вызовы(путь, переменная):
    """Имена методов, вызванных у этой переменной: recorder.frame(...) -> 'frame'."""
    with open(путь, "rb") as f:
        дерево = ast.parse(f.read().decode("utf-8"))
    имена = set()
    for узел in ast.walk(дерево):
        if (isinstance(узел, ast.Call) and isinstance(узел.func, ast.Attribute)
                and isinstance(узел.func.value, ast.Name)
                and узел.func.value.id == переменная):
            имена.add(узел.func.attr)
    return имена


@pytest.mark.parametrize("путь,переменная", ИНСТРУМЕНТЫ,
                         ids=[os.path.basename(п) for п, _ in ИНСТРУМЕНТЫ])
def test_методы_записи_существуют(путь, переменная):
    if not os.path.exists(путь):
        pytest.skip("нет %s" % путь)
    for имя in вызовы(путь, переменная):
        assert hasattr(RecordWriter, имя), (
            "%s зовёт recorder.%s(), которого у RecordWriter нет"
            % (os.path.basename(путь), имя))


def test_в_запись_идёт_кадр_до_отрисовки():
    """Ключ mission.record_raw читают оба инструмента, снимающих датасет.

    Записанный кадр с рамками детектора и полосой HUD — это отравленный датасет:
    сеть выучит собственную разметку как признак станции, и заметить это по метрике
    нельзя (docs/YOLO_TRAINING.md §2).
    """
    for путь, _ in ИНСТРУМЕНТЫ:
        if not os.path.exists(путь):
            continue
        with open(путь, "rb") as f:
            текст = f.read().decode("utf-8")
        assert "mission.record_raw" in текст, (
            "%s не спрашивает mission.record_raw" % os.path.basename(путь))
