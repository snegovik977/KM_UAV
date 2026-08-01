# -*- coding: utf-8 -*-
"""Загрузка config.yaml без обязательной зависимости от PyYAML.

PyYAML нет ни на этой машине, ни, скорее всего, на борту — а поставить его на борт
означает перевести дрон в режим клиента Wi-Fi, после чего он перестаёт быть
на 172.17.49.2 (docs/organizer_handouts.md §2.4). Ради одного файла настроек эта возня
не окупается, поэтому здесь лежит разбор того подмножества YAML, которым написан
config.yaml: вложенные словари, скаляры, комментарии. Если PyYAML в системе есть,
используется он.

Подмножество осознанно узкое. Всё, что в него не входит (списки, многострочные блоки,
якоря), в config.yaml не используется — и не должно появиться.

    from config import load
    cfg = load()                 # рядом с этим файлом
    cfg["flight"]["h_min"]       # или cfg.get_path("flight.h_min")
"""
from __future__ import annotations

import io
import os

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


class ConfigError(ValueError):
    pass


class Config(dict):
    """Словарь с доступом по пути: cfg.get_path('flight.h_min')."""

    def __init__(self, data, path=None):
        dict.__init__(self, data)
        self.path = path

    def get_path(self, dotted, default=None):
        node = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is not None:
                    return default
                raise ConfigError("нет параметра %r в %s" % (dotted, self.path or "config"))
            node = node[part]
        return node


def _scalar(text):
    """YAML-скаляр -> число, bool, None или строка."""
    text = text.strip()
    if not text:
        return None
    if text[0] in "\"'" and text[-1] == text[0] and len(text) >= 2:
        return text[1:-1]
    low = text.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "~", "none"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _strip_comment(line):
    """Убрать хвостовой комментарий, не тронув решётку внутри кавычек."""
    quote = None
    for i, char in enumerate(line):
        if quote:
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char == "#":
            return line[:i]
    return line


def loads(text, path=None):
    """Разобрать текст в словарь. Только вложенные словари и скаляры."""
    # BOM в начале файла: его дописывает почти любой Windows-редактор и PowerShell
    # (Set-Content -Encoding utf8 в PS 5.1). Невидимый символ прилипает к первому
    # ключу, и разбор падает на первой же строке с сообщением про «строку без ключа»,
    # в котором не видно причины. На площадке config.yaml правят руками и в чужом
    # редакторе (регламент 1.4 — машина выдаётся), так что случай не гипотетический.
    if text[:1] == "﻿":
        text = text[1:]
    root = {}
    # Стек (отступ, словарь): по отступу определяем, в какой словарь класть ключ.
    stack = [(-1, root)]
    for number, raw in enumerate(text.splitlines(), 1):
        line = _strip_comment(raw).rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("-"):
            raise ConfigError("%s:%d: списки не поддерживаются, а в config.yaml их быть "
                              "не должно: %r" % (path or "config", number, raw.strip()))
        indent = len(line) - len(line.lstrip())
        if ":" not in line:
            raise ConfigError("%s:%d: строка без ключа: %r" % (path or "config", number, raw))
        key, _, value = line.strip().partition(":")
        key = key.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ConfigError("%s:%d: сбит отступ" % (path or "config", number))
        parent = stack[-1][1]

        if value.strip() == "":
            child = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _scalar(value)
    return root


def load(path=None):
    """Прочитать config.yaml. PyYAML, если он есть, иначе разбор выше."""
    path = path or DEFAULT_PATH
    with io.open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        import yaml
        data = yaml.safe_load(text)
    except ImportError:
        data = loads(text, path)
    if not isinstance(data, dict):
        raise ConfigError("%s: ожидался словарь на верхнем уровне" % path)
    return Config(data, path)
