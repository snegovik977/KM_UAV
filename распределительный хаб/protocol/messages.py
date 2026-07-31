# -*- coding: utf-8 -*-
"""Формат пакетов — единый источник правды для дрона, автомобиля и визуализатора.

Главное архитектурное требование задания (docs/NOTES.md §3): в подзадачах 2.2.4/2.2.6
пакеты шлёт программа-передатчик, а в финале — настоящий дрон, и код автомобиля не должен
заметить разницы. Поэтому формат фиксируется здесь один раз, и никто — ни дрон, ни
передатчик, ни хаб — не собирает JSON руками.

Оболочка любого пакета:

    {"ver": 1, "src": "drone", "seq": 17, "ts": 1234.567, "type": "telemetry", "data": {...}}

Пять типов, из них пункту 1 плана нужны только takeoff и telemetry. Остальные три
заведены сразу, потому что трек автомобиля начинает писать под этот формат уже сейчас,
а менять контракт задним числом дороже, чем описать его целиком:

    takeoff        автомобиль/передатчик -> дрон    команда на взлёт, несёт origin
    telemetry      дрон -> всем                     поза и состояние миссии
    station_new    обнаружитель -> всем             пакет 1 регламента 2.2.4
    status_update  обнаружитель -> всем             пакет 2 регламента 2.2.4
    recon_done     дрон -> всем                     пакет 3 регламента 2.2.4

Координаты везде в НАШЕЙ системе координат: X вперёд, Y влево, Z вверх, начало — точка
взлёта дрона (docs/DRONE_PLAN.md §1.2). Пересчёт в СК автопилота живёт только
в квадрокоптер/flight.py и наружу не выходит.

Модуль совместим с Python 3.8 и не тянет зависимостей: он импортируется и на борту,
и на компьютере оператора.
"""
from __future__ import annotations

import json
import time

PROTOCOL_VERSION = 1

# Кто имеет право быть отправителем. Строгий список, потому что по src ведётся
# дедупликация и по нему же различаются независимые нумерации station_new.
SOURCES = ("drone", "car", "transmitter", "hub", "visualizer")

# Три класса состояния станции из регламента 2.1. Те же строки — метки классов YOLO.
STATUSES = ("ok", "dust", "broken")

# Состояния автомата миссии, допустимые в поле telemetry.state (см. квадрокоптер/mission.py).
MISSION_STATES = ("IDLE", "ARM", "TAKEOFF", "SURVEY", "DUST", "RETURN", "LAND", "DISARM", "ERROR")


class ProtocolError(ValueError):
    """Пакет не соответствует контракту: чужая версия, неизвестный тип, нет поля."""


# Схема полезной нагрузки: имя поля -> (тип, обязательное). Тип используется и для
# приведения: "1.5" от чужой реализации станет 1.5, а не уронит разбор.
_REQUIRED = True
_OPTIONAL = False

_SCHEMA = {
    "takeoff": {
        # origin — начало общей СК роя. В финале его задаёт автомобиль в момент команды,
        # в подзадачах 2.3.x его шлёт передатчик нулевым.
        "origin_x": (float, _REQUIRED),
        "origin_y": (float, _REQUIRED),
        "origin_yaw": (float, _REQUIRED),
        # Размеры зоны облёта: параметры попытки, а не константа кода (регламент 2.5).
        # Если автомобиль их не знает — дрон возьмёт свои из config.yaml.
        "zone_w": (float, _OPTIONAL),
        "zone_h": (float, _OPTIONAL),
    },
    "telemetry": {
        "x": (float, _REQUIRED),
        "y": (float, _REQUIRED),
        "z": (float, _REQUIRED),
        "yaw": (float, _REQUIRED),          # градусы, наша СК (против часовой от оси X)
        "batt": (float, _REQUIRED),         # вольты, как отдаёт get_battery_status()
        "state": (str, _REQUIRED),          # одно из MISSION_STATES
    },
    "station_new": {
        "id": (str, _REQUIRED),             # префикс по источнику: d1, d2… / c1, c2…
        "x": (float, _REQUIRED),
        "y": (float, _REQUIRED),
        "status": (str, _REQUIRED),         # одно из STATUSES
        "conf": (float, _REQUIRED),
    },
    "status_update": {
        "id": (str, _REQUIRED),
        "status": (str, _REQUIRED),
        "conf": (float, _REQUIRED),
    },
    "recon_done": {
        "count": (int, _REQUIRED),          # сколько станций найдено за весь облёт
    },
}

TYPES = tuple(sorted(_SCHEMA))


# --------------------------------------------------------------------- сборка

def make(src, seq, msg_type, data, ts=None):
    """Собрать и проверить пакет. Валидация та же, что при разборе, — отправитель
    не может создать пакет, который получатель отвергнет."""
    msg = {
        "ver": PROTOCOL_VERSION,
        "src": src,
        "seq": int(seq),
        "ts": float(time.time() if ts is None else ts),
        "type": msg_type,
        "data": dict(data),
    }
    return validate(msg)


def make_takeoff(src, seq, origin_x=0.0, origin_y=0.0, origin_yaw=0.0,
                 zone_w=None, zone_h=None, ts=None):
    data = {"origin_x": origin_x, "origin_y": origin_y, "origin_yaw": origin_yaw}
    if zone_w is not None:
        data["zone_w"] = zone_w
    if zone_h is not None:
        data["zone_h"] = zone_h
    return make(src, seq, "takeoff", data, ts)


def make_telemetry(src, seq, x, y, z, yaw, batt, state, ts=None):
    return make(src, seq, "telemetry",
                {"x": x, "y": y, "z": z, "yaw": yaw, "batt": batt, "state": state}, ts)


def make_station_new(src, seq, station_id, x, y, status, conf, ts=None):
    return make(src, seq, "station_new",
                {"id": station_id, "x": x, "y": y, "status": status, "conf": conf}, ts)


def make_status_update(src, seq, station_id, status, conf, ts=None):
    return make(src, seq, "status_update",
                {"id": station_id, "status": status, "conf": conf}, ts)


def make_recon_done(src, seq, count, ts=None):
    return make(src, seq, "recon_done", {"count": count}, ts)


class MessageFactory:
    """Нумерует пакеты одного источника. Монотонный seq — основа дедупликации,
    поэтому счётчик должен быть один на процесс, а не заводиться на каждый вызов."""

    def __init__(self, src):
        if src not in SOURCES:
            raise ProtocolError("неизвестный src: %r (ожидалось одно из %s)" % (src, SOURCES))
        self.src = src
        self._seq = 0

    def _next(self):
        self._seq += 1
        return self._seq

    def takeoff(self, **kw):
        return make_takeoff(self.src, self._next(), **kw)

    def telemetry(self, **kw):
        return make_telemetry(self.src, self._next(), **kw)

    def station_new(self, **kw):
        return make_station_new(self.src, self._next(), **kw)

    def status_update(self, **kw):
        return make_status_update(self.src, self._next(), **kw)

    def recon_done(self, **kw):
        return make_recon_done(self.src, self._next(), **kw)


# -------------------------------------------------------------------- разбор

def _coerce(value, want, where):
    if want is float:
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ProtocolError("%s: ожидалось число, получено %r" % (where, value))
    if want is int:
        try:
            # int("3") работает, int(3.7) молча обрежет — для count это допустимо,
            # а int("3.7") упадёт, и это правильно: такого поля быть не должно.
            return int(value)
        except (TypeError, ValueError):
            raise ProtocolError("%s: ожидалось целое, получено %r" % (where, value))
    if want is str:
        if not isinstance(value, str):
            raise ProtocolError("%s: ожидалась строка, получено %r" % (where, value))
        return value
    raise ProtocolError("%s: неподдерживаемый тип поля %r" % (where, want))


def validate(msg):
    """Проверить готовый словарь-пакет и вернуть его нормализованную копию."""
    if not isinstance(msg, dict):
        raise ProtocolError("пакет не словарь: %r" % (type(msg),))

    ver = msg.get("ver")
    if ver != PROTOCOL_VERSION:
        raise ProtocolError("версия протокола %r, ожидалась %r" % (ver, PROTOCOL_VERSION))

    src = msg.get("src")
    if src not in SOURCES:
        raise ProtocolError("неизвестный src: %r" % (src,))

    msg_type = msg.get("type")
    if msg_type not in _SCHEMA:
        raise ProtocolError("неизвестный type: %r (известные: %s)" % (msg_type, ", ".join(TYPES)))

    seq = _coerce(msg.get("seq"), int, "seq")
    if seq < 0:
        raise ProtocolError("seq отрицательный: %r" % (seq,))
    ts = _coerce(msg.get("ts"), float, "ts")

    raw_data = msg.get("data")
    if not isinstance(raw_data, dict):
        raise ProtocolError("data не словарь: %r" % (type(raw_data),))

    data = {}
    for name, (want, required) in _SCHEMA[msg_type].items():
        if name not in raw_data:
            if required:
                raise ProtocolError("%s: нет обязательного поля %r" % (msg_type, name))
            continue
        data[name] = _coerce(raw_data[name], want, "%s.%s" % (msg_type, name))

    # Незнакомые поля пропускаем как есть: чужая реализация может добавить своё,
    # и это не повод отвергать пакет. Но известные поля уже приведены к типу.
    for name, value in raw_data.items():
        if name not in data:
            data[name] = value

    # Проверки, которые схема типов не ловит.
    if msg_type == "telemetry" and data["state"] not in MISSION_STATES:
        raise ProtocolError("telemetry.state=%r, ожидалось одно из %s"
                            % (data["state"], MISSION_STATES))
    if msg_type in ("station_new", "status_update") and data["status"] not in STATUSES:
        raise ProtocolError("%s.status=%r, ожидалось одно из %s"
                            % (msg_type, data["status"], STATUSES))
    if msg_type in ("station_new", "status_update") and not data["id"]:
        raise ProtocolError("%s.id пустой" % msg_type)
    if msg_type == "recon_done" and data["count"] < 0:
        raise ProtocolError("recon_done.count отрицательный: %r" % (data["count"],))

    return {"ver": PROTOCOL_VERSION, "src": src, "seq": seq, "ts": ts,
            "type": msg_type, "data": data}


def parse(raw):
    """Разобрать пакет из bytes/str/dict. Единственная точка входа для получателей."""
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ProtocolError("пакет не в UTF-8: %s" % e)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError as e:
            raise ProtocolError("невалидный JSON: %s" % e)
    return validate(raw)


def dumps(msg):
    """Пакет -> bytes для отправки. ensure_ascii=False держит русские строки читаемыми
    в логах хаба; separators убирают лишние пробелы."""
    return json.dumps(msg, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


# --------------------------------------------------------------- дедупликация

class SeqDedup:
    """Отсев повторов по (src, seq).

    Нужен и поверх HTTP: отправитель ретраит пакет при таймауте, не зная, дошёл ли
    предыдущий. Без дедупликации повтор station_new превратится в лишнюю станцию,
    а «количество обнаруженных станций соответствует реальному» — условие всех 8 баллов
    подзадачи 2.3.2.

    Память ограничена: помним последние `capacity` ключей на источник. Полёт даёт
    порядка тысяч пакетов, так что окна в 4096 хватает с запасом.
    """

    def __init__(self, capacity=4096):
        self.capacity = int(capacity)
        self._seen = {}    # src -> set ключей
        self._order = {}   # src -> список ключей в порядке появления

    def is_new(self, msg):
        """True, если пакет виден впервые (и тогда он запоминается)."""
        src = msg["src"]
        seq = msg["seq"]
        seen = self._seen.setdefault(src, set())
        if seq in seen:
            return False
        seen.add(seq)
        order = self._order.setdefault(src, [])
        order.append(seq)
        if len(order) > self.capacity:
            seen.discard(order.pop(0))
        return True

    def filter(self, msgs):
        """Оставить только новые пакеты, сохранив порядок."""
        return [m for m in msgs if self.is_new(m)]
