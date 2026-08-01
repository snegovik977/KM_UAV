# -*- coding: utf-8 -*-
"""Визуализатор и мост в текстовый формат организаторов.

Без визуализатора подзадача 2.3.2 не сдаётся вообще: 8 баллов даются за то, что он
отображает станции в реальном времени. Проверяем две вещи, которые молча ломаются:
привязку «метры -> пиксели карты» (перепутанный знак развернёт карту) и ведение
станций по id, а не по порядку прихода пакетов — на этом спотыкается сам пример
организаторов (source/2_step/MapExample.py).
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "распределительный хаб"))

import config as config_module
from protocol import MapAnchor, MessageFactory, to_legacy
from visualizer import MapView


def привязка(**правки):
    параметры = {"px_per_m": 100.0, "origin_px": (500, 400), "yaw_deg": 0.0,
                 "width": 1000, "height": 800}
    параметры.update(правки)
    return MapAnchor(**параметры)


# ------------------------------------------------------------------- привязка карты

def test_точка_взлёта_это_ноль_карты():
    assert привязка().to_px(0.0, 0.0) == (500, 400)


def test_вперёд_это_вверх_по_картинке():
    """Ось строк изображения растёт вниз, наша X — вперёд. Перепутанный знак
    развернёт всю карту станций на 180° и не упадёт с ошибкой."""
    u, v = привязка().to_px(1.0, 0.0)
    assert u == 500 and v == 300


def test_влево_это_влево_по_картинке():
    u, v = привязка().to_px(0.0, 1.0)      # Y влево
    assert u == 400 and v == 400


def test_масштаб_работает():
    assert привязка(px_per_m=50.0).to_px(2.0, 0.0) == (500, 300)


def test_поворот_карты():
    """Если ось X дрона не смотрит вверх по картинке, поворот вписывается в конфиг."""
    u, v = привязка(yaw_deg=90.0).to_px(1.0, 0.0)
    assert (u, v) == (400, 400), "поворот на 90° должен увести «вперёд» влево"


@pytest.mark.parametrize("точка", [(0.0, 0.0), (1.3, -0.7), (-2.4, 3.1)])
@pytest.mark.parametrize("поворот", [0.0, 37.0, -90.0])
def test_обратный_перевод(точка, поворот):
    """to_meters нужна, чтобы отметить точку взлёта мышью по картинке и получить
    готовые числа для конфига. Она обязана быть строго обратной to_px."""
    anchor = привязка(yaw_deg=поворот, px_per_m=200.0)
    u, v = anchor.to_px(*точка)
    обратно = anchor.to_meters(u, v)
    assert обратно[0] == pytest.approx(точка[0], abs=0.01)
    assert обратно[1] == pytest.approx(точка[1], abs=0.01)


def test_выход_за_картинку_ловится():
    anchor = привязка()
    assert anchor.inside(*anchor.to_px(0.0, 0.0))
    assert not anchor.inside(*anchor.to_px(50.0, 0.0))


def test_привязка_из_конфига():
    cfg = config_module.load()
    anchor = MapAnchor.from_config(cfg, width=1188, height=900)
    assert anchor.px_per_m > 0
    assert anchor.inside(*anchor.to_px(0.0, 0.0)), "ноль карты вне картинки"


def test_нулевой_масштаб_падает():
    with pytest.raises(ValueError):
        MapAnchor(px_per_m=0.0)


# -------------------------------------------------- мост в формат организаторов

def test_станция_переводится_в_пиксели():
    factory = MessageFactory("drone")
    пакет = factory.station_new(station_id="d1", x=1.0, y=0.0, status="ok", conf=0.9)
    отправления = to_legacy(пакет, привязка())
    assert ("/fly", "500,300") in отправления
    assert ("/eyecar", "Good") in отправления


def test_запылённая_станция_идёт_как_исправная():
    """В текстовом формате статуса «покрыта пылью» нет вообще. Станция рабочая,
    и потерять её из маршрута автомобиля хуже, чем не сообщить про пыль."""
    factory = MessageFactory("drone")
    пакет = factory.station_new(station_id="d1", x=0.0, y=0.0, status="dust", conf=0.8)
    assert ("/eyecar", "Good") in to_legacy(пакет, привязка())


def test_неисправная_станция_идёт_как_broken():
    factory = MessageFactory("drone")
    пакет = factory.station_new(station_id="d1", x=0.0, y=0.0, status="broken", conf=0.8)
    assert ("/eyecar", "Broken") in to_legacy(пакет, привязка())


def test_станция_вне_карты_не_отправляется():
    """Пример организаторов на выход за границы картинки падает с IndexError."""
    factory = MessageFactory("drone")
    пакет = factory.station_new(station_id="d1", x=100.0, y=0.0, status="ok", conf=0.5)
    assert to_legacy(пакет, привязка()) == []


def test_телеметрии_в_старом_формате_нет():
    factory = MessageFactory("drone")
    пакет = factory.telemetry(x=0.0, y=0.0, z=1.5, yaw=0.0, batt=8.0, state="SURVEY")
    assert to_legacy(пакет, привязка()) == []


# ---------------------------------------------------------------- карта в памяти

def test_станции_ведутся_по_id_а_не_по_счётчику():
    """Ровно то, чем плох пример организаторов: он красит станции счётчиком, и один
    потерянный пакет сдвигает все последующие статусы на одну станцию."""
    factory = MessageFactory("drone")
    view = MapView(привязка(), log=lambda _: None)
    view.apply(factory.station_new(station_id="d1", x=0.0, y=0.0, status="ok", conf=1.0))
    view.apply(factory.station_new(station_id="d2", x=1.0, y=0.0, status="ok", conf=1.0))
    view.apply(factory.status_update(station_id="d1", status="broken", conf=0.9))

    assert view.stations["d1"]["status"] == "broken"
    assert view.stations["d2"]["status"] == "ok"


def test_повторный_пакет_уточняет_а_не_плодит():
    """Дрон дошлёт station_new с уточнённой медианой — на карте обязана обновиться
    та же станция, а не появиться вторая."""
    factory = MessageFactory("drone")
    view = MapView(привязка(), log=lambda _: None)
    view.apply(factory.station_new(station_id="d1", x=0.0, y=0.0, status="ok", conf=1.0))
    view.apply(factory.station_new(station_id="d1", x=0.2, y=0.1, status="ok", conf=1.0))

    assert len(view.stations) == 1
    assert view.stations["d1"]["x"] == pytest.approx(0.2)


def test_статус_раньше_координат_не_теряется():
    """Пакеты могут прийти не по порядку. Терять статус нельзя — это баллы 2.3.3."""
    factory = MessageFactory("drone")
    view = MapView(привязка(), log=lambda _: None)
    view.apply(factory.status_update(station_id="d7", status="dust", conf=0.9))
    assert view.stations["d7"]["status"] == "dust"
    assert view.stations["d7"]["x"] is None


def test_след_дрона_ограничен():
    factory = MessageFactory("drone")
    view = MapView(привязка(), log=lambda _: None)
    for i in range(600):
        view.apply(factory.telemetry(x=0.01 * i, y=0.0, z=1.5, yaw=0.0,
                                     batt=8.0, state="SURVEY"))
    assert len(view.trail) <= 400


def test_сводка_считает_станции_по_статусам():
    factory = MessageFactory("drone")
    view = MapView(привязка(), log=lambda _: None)
    for имя, статус in (("d1", "ok"), ("d2", "dust"), ("d3", "ok")):
        view.apply(factory.station_new(station_id=имя, x=0.0, y=0.0,
                                       status=статус, conf=1.0))
    view.apply(factory.recon_done(count=3))

    assert view.counters() == {"ok": 2, "dust": 1}
    assert "станций: 3" in view.summary()
    assert view.recon_done == 3


def test_карта_рисуется():
    """Отрисовка не должна падать ни на пустой карте, ни на полной."""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from visualizer import нарисовать

    factory = MessageFactory("drone")
    основа = np.full((800, 1000, 3), 40, dtype=np.uint8)
    view = MapView(привязка(), log=lambda _: None)
    assert нарисовать(основа, view) is not None

    view.apply(factory.station_new(station_id="d1", x=0.5, y=0.5,
                                   status="dust", conf=0.8))
    view.apply(factory.telemetry(x=0.1, y=0.0, z=1.5, yaw=0.0, batt=8.0, state="SURVEY"))
    кадр = нарисовать(основа, view)
    assert кадр is not None and кадр.shape == основа.shape
    assert (кадр != основа).any(), "на карте ничего не нарисовалось"
