# -*- coding: utf-8 -*-
"""Облёт трассы по контуру: взлёт с площадки внутри овала, четыре угла, возврат.

Основной режим маршрута. Проверяется геометрия — что круг замкнут, что дрон
не улетает за габариты трассы и что размеры берутся из конфига, а не зашиты.
"""
from __future__ import annotations

import math

import pytest

import config as config_module
from survey import plan_from_config, plan_loop


def test_четыре_угла_и_замкнутый_круг():
    plan = plan_loop(length=3.0, width=2.0, altitude=1.5)
    # 4 угла + возврат в первый: без замыкания последний отрезок контура
    # остался бы непройденным
    assert len(plan.waypoints) == 5
    assert (plan.waypoints[0].x, plan.waypoints[0].y) == \
           (plan.waypoints[-1].x, plan.waypoints[-1].y)
    углы = {(w.x, w.y) for w in plan.waypoints}
    assert углы == {(1.5, -1.0), (1.5, 1.0), (-1.5, 1.0), (-1.5, -1.0)}


def test_контур_центрирован_на_точке_взлёта():
    """Площадка «H» внутри овала, значит контур обходит её кругом, а не уходит вбок."""
    plan = plan_loop(length=3.0, width=2.0, altitude=1.5)
    assert not plan.warnings
    x = [w.x for w in plan.waypoints]
    y = [w.y for w in plan.waypoints]
    assert min(x) == pytest.approx(-1.5) and max(x) == pytest.approx(1.5)
    assert min(y) == pytest.approx(-1.0) and max(y) == pytest.approx(1.0)


def test_смещённый_центр():
    """Если H стоит не в середине овала, сдвиг задаётся конфигом."""
    plan = plan_loop(length=3.0, width=2.0, altitude=1.5, center_x=0.5, center_y=-0.3)
    x = [w.x for w in plan.waypoints]
    assert min(x) == pytest.approx(-1.0) and max(x) == pytest.approx(2.0)
    assert min(w.y for w in plan.waypoints) == pytest.approx(-1.3)


def test_обход_против_часовой():
    """Направление обхода должно быть предсказуемым: по нему считается,
    с какой стороны дрон подходит к трассе."""
    plan = plan_loop(length=3.0, width=2.0, altitude=1.5, direction="ccw")
    площадь = _знаковая_площадь(plan.waypoints)
    assert площадь > 0, "обход пошёл по часовой вместо против"

    по_часовой = plan_loop(length=3.0, width=2.0, altitude=1.5, direction="cw")
    assert _знаковая_площадь(по_часовой.waypoints) < 0


def _знаковая_площадь(точки):
    """Формула шнурков: знак говорит о направлении обхода."""
    сумма = 0.0
    for a, b in zip(точки, точки[1:]):
        сумма += a.x * b.y - b.x * a.y
    return сумма / 2.0


def test_margin_раздвигает_и_сужает_контур():
    """Положительный margin отодвигает дрон от разметки, отрицательный заводит
    внутрь овала — подальше от сетки по краям полигона."""
    наружу = plan_loop(length=3.0, width=2.0, altitude=1.5, margin=0.2)
    внутрь = plan_loop(length=3.0, width=2.0, altitude=1.5, margin=-0.2)
    assert max(w.x for w in наружу.waypoints) == pytest.approx(1.7)
    assert max(w.x for w in внутрь.waypoints) == pytest.approx(1.3)


def test_несколько_кругов():
    один = plan_loop(length=3.0, width=2.0, altitude=1.5, laps=1)
    три = plan_loop(length=3.0, width=2.0, altitude=1.5, laps=3)
    assert len(три.waypoints) == 13          # 4 угла x 3 круга + замыкание
    assert три.length == pytest.approx(один.length * 3, rel=0.35)


def test_высота_и_курс_одинаковы():
    """Курс не меняется: камера смотрит вниз, а каждый разворот — это рывок
    ориентации, который портит локализацию."""
    plan = plan_loop(length=3.0, width=2.0, altitude=1.5)
    assert {w.z for w in plan.waypoints} == {1.5}
    assert {w.yaw for w in plan.waypoints} == {0.0}


def test_длина_и_время():
    plan = plan_loop(length=3.0, width=2.0, altitude=1.5)
    # Периметр прямоугольника 3x2
    assert plan.length == pytest.approx(10.0)
    время = plan.duration(speed=0.45)
    assert время > 10.0 / 0.45               # углы тоже стоят времени
    assert время < 120.0, plan.summary(speed=0.45)


@pytest.mark.parametrize("аргументы", [
    {"length": 0.0}, {"width": -1.0}, {"laps": 0},
    {"direction": "боком"}, {"margin": -2.0},
])
def test_негодные_параметры(аргументы):
    базовые = dict(length=3.0, width=2.0, altitude=1.5)
    базовые.update(аргументы)
    with pytest.raises(ValueError):
        plan_loop(**базовые)


def test_площадка_вне_контура_вызывает_предупреждение():
    """Если центр задан неверно, дрон полетит обходить пустое место рядом с трассой.
    Молчать об этом нельзя — это потерянная попытка."""
    plan = plan_loop(length=3.0, width=2.0, altitude=1.5, center_x=8.0)
    assert any("center_x" in w for w in plan.warnings)


# ------------------------------------------------------------------------ конфиг

def test_маршрут_из_боевого_конфига():
    cfg = config_module.load()
    assert cfg.get_path("route.mode") == "loop"
    plan = plan_from_config(cfg)
    assert not plan.warnings, plan.warnings
    assert len(plan.waypoints) == 5
    assert plan.altitude == cfg.get_path("flight.h_survey")
    assert plan.length == pytest.approx(
        2 * (cfg.get_path("loop.length") + cfg.get_path("loop.width")))


def test_пакет_не_перекрывает_габариты_трассы():
    """Трассу меряем мы рулеткой, автомобиль её размеров не знает. Случайное значение
    в чужой программе увело бы дрон обходить контур, которого нет."""
    cfg = config_module.load()
    plan = plan_from_config(cfg, zone_override={"zone_w": 6.0, "zone_h": 6.0})
    # Ожидание считается от конфига, а не от чисел в тесте: центр контура сдвинут
    # (площадка «H» стоит не в середине овала), и габариты правятся рулеткой на площадке.
    край_x = (cfg.get_path("loop.center_x") + cfg.get_path("loop.length") / 2
              + cfg.get_path("loop.margin"))
    assert max(w.x for w in plan.waypoints) == pytest.approx(край_x)


def test_габариты_из_пакета_по_явному_флагу():
    """Если договоримся, что размеры присылает автомобиль, это включается конфигом."""
    cfg = config_module.load()
    cfg["loop"]["use_packet_zone"] = True
    plan = plan_from_config(cfg, zone_override={"zone_w": 4.0, "zone_h": 2.5})
    margin = cfg.get_path("loop.margin")
    assert max(w.x for w in plan.waypoints) == pytest.approx(
        cfg.get_path("loop.center_x") + 4.0 / 2 + margin)
    assert max(w.y for w in plan.waypoints) == pytest.approx(
        cfg.get_path("loop.center_y") + 2.5 / 2 + margin)


def test_переключение_на_змейку():
    """Змейка никуда не делась: если станции окажутся разбросаны по площади,
    переключение стоит одной строки конфига."""
    cfg = config_module.load()
    cfg["route"]["mode"] = "survey"
    plan = plan_from_config(cfg)
    assert plan.label == "змейка"
    assert len(plan.waypoints) > 5


def test_неизвестный_режим_отвергается():
    cfg = config_module.load()
    cfg["route"]["mode"] = "спираль"
    with pytest.raises(ValueError):
        plan_from_config(cfg)
