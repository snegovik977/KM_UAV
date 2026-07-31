# -*- coding: utf-8 -*-
"""Сквозная проверка транспорта: дрон -> хаб -> автомобиль и обратно.

Хаб поднимается настоящий (stdlib-вариант, на свободном порту) — mock'ать сеть здесь
нечего: проверяется именно связка HTTP-клиента с HTTP-сервером.
"""
from __future__ import annotations

import threading
import time

import pytest

import hub as hub_module
from protocol import HttpTransport, MessageFactory


TIMEOUT = 5.0     # запас: на медленной машине опрос идёт раз в 0.2 с


@pytest.fixture
def живой_хаб():
    state = hub_module.HubState(log_path=None)
    server = hub_module.make_stdlib_server(state, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = "http://127.0.0.1:%d" % server.server_address[1]
    try:
        yield url, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def ждать(предикат, timeout=TIMEOUT):
    """Опрос с таймаутом вместо sleep наугад: тест не должен зависеть от скорости машины."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = предикат()
        if result:
            return result
        time.sleep(0.02)
    return None


def test_команда_взлёта_доходит_до_дрона(живой_хаб):
    url, _ = живой_хаб
    car = HttpTransport(url, src="transmitter", poll=False).start()
    drone = HttpTransport(url, src="drone").start()
    try:
        car.send(MessageFactory("transmitter").takeoff(origin_x=1.0, origin_y=2.0,
                                                       origin_yaw=90.0, zone_w=6.0, zone_h=4.0))
        received = ждать(lambda: [m for m in drone.recv() if m["type"] == "takeoff"])
        assert received, "дрон не получил команду на взлёт"
        assert received[0]["data"]["origin_x"] == 1.0
        assert received[0]["data"]["zone_w"] == 6.0
    finally:
        car.stop()
        drone.stop()


def test_дрон_не_получает_свои_же_пакеты(живой_хаб):
    """Хаб раздаёт всем подряд; фильтрация по exclude_src не даёт дрону
    обрабатывать собственную телеметрию как входящую команду."""
    url, _ = живой_хаб
    drone = HttpTransport(url, src="drone").start()
    try:
        factory = MessageFactory("drone")
        for _ in range(3):
            drone.send(factory.telemetry(x=0, y=0, z=1, yaw=0, batt=7.9, state="SURVEY"))
        ждать(lambda: drone.stats["sent"] >= 3)
        time.sleep(0.5)
        assert drone.recv() == []
    finally:
        drone.stop()


def test_телеметрия_не_вытесняет_события(живой_хаб):
    """Ключевое свойство очереди: поток телеметрии не имеет права выдавить station_new.
    Совпадение числа станций с реальным — условие всех 8 баллов подзадачи 2.3.2."""
    url, state = живой_хаб
    drone = HttpTransport(url, src="drone", poll=False)
    factory = MessageFactory("drone")
    try:
        # Кладём всё до старта потока отправки: очередь заведомо переполнена телеметрией.
        for i in range(200):
            drone.send(factory.telemetry(x=i, y=0, z=1, yaw=0, batt=7.5, state="SURVEY"))
        for i in range(5):
            drone.send(factory.station_new(station_id="d%d" % (i + 1), x=i, y=0,
                                           status="ok", conf=0.9))
        drone.start()

        snapshot = ждать(lambda: state.snapshot() if state.snapshot()["count"] == 5 else None)
        assert snapshot, "станции потерялись в потоке телеметрии"
        # Из 200 пакетов телеметрии уйдёт только последний — остальные устарели.
        assert drone.stats["sent"] <= 10
    finally:
        drone.stop()


def test_повторная_отправка_не_плодит_станции(живой_хаб):
    """Ретрай при таймауте — норма, а вот вторая станция из того же пакета — нет."""
    url, state = живой_хаб
    drone = HttpTransport(url, src="drone", poll=False).start()
    try:
        msg = MessageFactory("drone").station_new(station_id="d1", x=1.0, y=2.0,
                                                  status="dust", conf=0.9)
        drone.send(msg)
        drone.send(msg)
        drone.send(msg)
        assert ждать(lambda: drone.stats["sent"] >= 3)
        snapshot = state.snapshot()
        assert snapshot["count"] == 1
        assert snapshot["stations"][0]["status"] == "dust"
    finally:
        drone.stop()


def test_карта_хаба_обновляется_по_статусу(живой_хаб):
    url, state = живой_хаб
    drone = HttpTransport(url, src="drone", poll=False).start()
    try:
        factory = MessageFactory("drone")
        drone.send(factory.station_new(station_id="d1", x=1.0, y=2.0, status="ok", conf=0.6))
        drone.send(factory.status_update(station_id="d1", status="dust", conf=0.94))
        drone.send(factory.recon_done(count=1))
        snapshot = ждать(lambda: state.snapshot()
                         if state.snapshot()["recon_done"] else None)
        assert snapshot["stations"][0]["status"] == "dust"
        assert snapshot["stations"][0]["x"] == 1.0      # координаты не потерялись
        assert snapshot["by_status"]["dust"] == 1
        assert snapshot["recon_done"]["count"] == 1
    finally:
        drone.stop()


def test_упавший_хаб_не_роняет_отправителя():
    """Хаб может подняться позже дрона или упасть в середине попытки.
    Полёт от этого прерываться не должен — только счётчик ошибок растёт."""
    drone = HttpTransport("http://127.0.0.1:1", src="drone", poll=False,
                          timeout=0.05, retries=0, log=lambda text: None).start()
    try:
        drone.send(MessageFactory("drone").recon_done(count=3))
        assert ждать(lambda: drone.stats["send_errors"] >= 1)
        assert drone.stats["sent"] == 0
        assert drone.health()["ok"] is False
    finally:
        drone.stop()


def test_битый_пакет_отвергается_хабом(живой_хаб):
    import urllib.request

    url, state = живой_хаб
    request = urllib.request.Request(url + "/msg", data=b'{"ver":99}',
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
    with pytest.raises(Exception):
        urllib.request.urlopen(request, timeout=2.0)
    assert state.health()["rejected"] == 1
    assert state.health()["messages"] == 0
