# -*- coding: utf-8 -*-
"""Контракт протокола. Ломать эти тесты нельзя: под этот же формат пишет трек автомобиля."""
from __future__ import annotations

import pytest

from protocol import messages as m


# --------------------------------------------------------------- round-trip

def test_round_trip_всех_типов():
    """Каждый тип переживает сборку -> bytes -> разбор без потерь."""
    factory = m.MessageFactory("drone")
    samples = [
        factory.takeoff(origin_x=1.5, origin_y=-2.0, origin_yaw=90.0),
        factory.telemetry(x=1.0, y=2.0, z=1.5, yaw=45.0, batt=7.8, state="SURVEY"),
        factory.station_new(station_id="d1", x=0.5, y=-1.25, status="dust", conf=0.91),
        factory.status_update(station_id="d1", status="ok", conf=0.77),
        factory.recon_done(count=5),
    ]
    for msg in samples:
        assert m.parse(m.dumps(msg)) == msg


def test_seq_монотонный_и_свой_на_источник():
    drone = m.MessageFactory("drone")
    car = m.MessageFactory("car")
    assert [drone.recon_done(count=1)["seq"] for _ in range(3)] == [1, 2, 3]
    # Нумерации источников независимы — иначе дедупликация склеит чужие пакеты.
    assert car.recon_done(count=1)["seq"] == 1


def test_необязательные_поля_не_подставляются():
    msg = m.make_takeoff("car", 1)
    assert "zone_w" not in msg["data"]
    assert m.make_takeoff("car", 2, zone_w=6.0, zone_h=4.0)["data"]["zone_w"] == 6.0


def test_числа_приводятся_к_типу():
    """Чужая реализация может прислать строку — это не повод отвергать пакет."""
    raw = {"ver": 1, "src": "car", "seq": "7", "ts": "123.5", "type": "recon_done",
           "data": {"count": "3"}}
    msg = m.parse(raw)
    assert msg["seq"] == 7 and msg["ts"] == 123.5 and msg["data"]["count"] == 3


def test_незнакомые_поля_сохраняются():
    """Смежный трек может добавить своё поле — пакет должен пройти целиком."""
    raw = m.make_recon_done("car", 1, count=2)
    raw["data"]["их_поле"] = "что-то"
    assert m.parse(m.dumps(raw))["data"]["их_поле"] == "что-то"


# ---------------------------------------------------------------- валидация

@pytest.mark.parametrize("broken, что_сломано", [
    ({"ver": 2}, "чужая версия протокола"),
    ({"src": "кто-то"}, "неизвестный источник"),
    ({"type": "выключи_свет"}, "неизвестный тип"),
    ({"seq": -1}, "отрицательный seq"),
    ({"data": []}, "data не словарь"),
])
def test_битая_оболочка_отвергается(broken, что_сломано):
    msg = m.make_telemetry("drone", 1, x=0, y=0, z=1, yaw=0, batt=7.5, state="IDLE")
    msg.update(broken)
    with pytest.raises(m.ProtocolError):
        m.validate(msg)


def test_нет_обязательного_поля():
    msg = m.make_station_new("drone", 1, station_id="d1", x=0, y=0, status="ok", conf=0.9)
    del msg["data"]["x"]
    with pytest.raises(m.ProtocolError):
        m.validate(msg)


def test_статус_только_из_трёх_классов():
    """Регламент 2.1 знает ровно три состояния станции — четвёртое означает
    рассинхрон меток между дроном и автомобилем, и его надо ловить сразу."""
    with pytest.raises(m.ProtocolError):
        m.make_station_new("drone", 1, station_id="d1", x=0, y=0, status="грязная", conf=0.9)


def test_состояние_миссии_проверяется():
    with pytest.raises(m.ProtocolError):
        m.make_telemetry("drone", 1, x=0, y=0, z=1, yaw=0, batt=7.5, state="ЛЕТИМ")


def test_пустой_id_станции():
    with pytest.raises(m.ProtocolError):
        m.make_station_new("drone", 1, station_id="", x=0, y=0, status="ok", conf=0.9)


def test_битый_json():
    with pytest.raises(m.ProtocolError):
        m.parse(b"{not json")


def test_неизвестный_src_у_фабрики():
    with pytest.raises(m.ProtocolError):
        m.MessageFactory("вертолёт")


# -------------------------------------------------------------- дедупликация

def test_повтор_пакета_отсеивается():
    """Отправитель ретраит пакет при таймауте, не зная, дошёл ли предыдущий.
    Без дедупликации повтор station_new даст лишнюю станцию, а совпадение их числа
    с реальным — условие всех 8 баллов подзадачи 2.3.2."""
    dedup = m.SeqDedup()
    msg = m.make_station_new("drone", 5, station_id="d1", x=0, y=0, status="ok", conf=0.9)
    assert dedup.is_new(msg)
    assert not dedup.is_new(msg)
    assert not dedup.is_new(m.parse(m.dumps(msg)))   # тот же пакет после сети


def test_одинаковый_seq_разных_источников_не_склеивается():
    dedup = m.SeqDedup()
    assert dedup.is_new(m.make_recon_done("drone", 1, count=1))
    assert dedup.is_new(m.make_recon_done("car", 1, count=1))


def test_потеря_пакета_не_ломает_последующие():
    """Пропуск в нумерации — нормальная ситуация: seq 2 потерян, 3 обязан пройти."""
    dedup = m.SeqDedup()
    got = [msg["seq"] for msg in dedup.filter([
        m.make_recon_done("drone", 1, count=1),
        m.make_recon_done("drone", 3, count=1),
        m.make_recon_done("drone", 3, count=1),
        m.make_recon_done("drone", 4, count=1),
    ])]
    assert got == [1, 3, 4]


def test_память_дедупликатора_ограничена():
    dedup = m.SeqDedup(capacity=8)
    for seq in range(1, 21):
        assert dedup.is_new(m.make_recon_done("drone", seq, count=1))
    assert len(dedup._seen["drone"]) == 8
    # Свежие пакеты по-прежнему отсеиваются как повторы...
    assert not dedup.is_new(m.make_recon_done("drone", 20, count=1))
    # ...а давно вытесненный пройдёт заново. Это осознанный размен: окно в тысячи
    # пакетов заведомо шире, чем разброс между повтором и оригиналом.
    assert dedup.is_new(m.make_recon_done("drone", 1, count=1))
