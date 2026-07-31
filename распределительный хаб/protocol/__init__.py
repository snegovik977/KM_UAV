# -*- coding: utf-8 -*-
"""Общий протокол роя: дрон, автомобиль, визуализатор и хаб импортируют ОДИН этот пакет.

Дублировать формат пакетов в трёх местах нельзя: в финале дрон обязан подставиться
вместо программы-передатчика без изменений в коде автомобиля (docs/NOTES.md §3).

    import sys; sys.path.insert(0, "<репозиторий>/распределительный хаб")
    from protocol import MessageFactory, parse, HttpTransport
"""
from __future__ import annotations

from .messages import (  # noqa: F401
    MISSION_STATES,
    PROTOCOL_VERSION,
    SOURCES,
    STATUSES,
    TYPES,
    MessageFactory,
    ProtocolError,
    SeqDedup,
    dumps,
    make,
    make_recon_done,
    make_station_new,
    make_status_update,
    make_takeoff,
    make_telemetry,
    parse,
    validate,
)
from .transport import HttpTransport  # noqa: F401

__all__ = [
    "MISSION_STATES", "PROTOCOL_VERSION", "SOURCES", "STATUSES", "TYPES",
    "MessageFactory", "ProtocolError", "SeqDedup", "HttpTransport",
    "dumps", "make", "make_recon_done", "make_station_new", "make_status_update",
    "make_takeoff", "make_telemetry", "parse", "validate",
]
