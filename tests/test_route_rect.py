# -*- coding: utf-8 -*-
"""Прямоугольник от точки взлёта: вперёд, вбок, назад, обратно на площадку.

Основной режим маршрута с 01.08.2026. Отличие от контура (test_route_loop.py) —
площадка «H» не в центре фигуры, а в её углу, и последняя точка совпадает с точкой
взлёта. Ошибка знака или порядка углов не падает с исключением: дрон просто улетает
не туда, поэтому проверяются именно координаты и их порядок.
"""
from __future__ import annotations

import pytest

import config as config_module
from survey import plan_from_config, plan_rect


def test_углы_в_заданном_порядке():
    """Порядок обхода — часть задания: вперёд 4.4, влево 1.5, назад 4.4, вправо 1.5."""
    plan = plan_rect(length=4.4, width=1.5, altitude=1.5)
    assert [(w.x, w.y) for w in plan.waypoints] == [
        (0.0, 0.0),      # площадка «H»: дрон уже здесь, точка нужна для plan.length
        (4.4, 0.0),      # вперёд
        (4.4, 1.5),      # влево
        (0.0, 1.5),      # назад
        (0.0, 0.0),      # вправо, обратно на площадку
    ]


def test_маршрут_начинается_и_кончается_в_точке_взлёта():
    """Последняя точка — сама площадка: RETURN после облёта только подтверждает
    позицию, а не гонит дрон через всю площадку с почти севшей батареей."""
    plan = plan_rect(length=4.4, width=1.5, altitude=1.5)
    первая = (plan.waypoints[0].x, plan.waypoints[0].y)
    последняя = (plan.waypoints[-1].x, plan.waypoints[-1].y)
    assert первая == последняя == (0.0, 0.0)


def test_сторона_вправо_зеркалит_только_поперечную_ось():
    """side: right — самая дорогая опечатка в секции rect: облёт уходит в зеркальную
    сторону от площадки. Вперёд при этом дрон летит так же."""
    влево = plan_rect(length=4.4, width=1.5, altitude=1.5, side="left")
    вправо = plan_rect(length=4.4, width=1.5, altitude=1.5, side="right")
    assert max(w.y for w in влево.waypoints) == pytest.approx(1.5)
    assert min(w.y for w in влево.waypoints) == pytest.approx(0.0)
    assert min(w.y for w in вправо.waypoints) == pytest.approx(-1.5)
    assert max(w.y for w in вправо.waypoints) == pytest.approx(0.0)
    assert [w.x for w in влево.waypoints] == [w.x for w in вправо.waypoints]


def test_площадка_в_углу_а_не_в_центре():
    """Отличие от plan_loop: зона лежит целиком вперёд и влево от точки взлёта,
    дрон не летает назад за спину, где площадки уже нет."""
    plan = plan_rect(length=4.4, width=1.5, altitude=1.5)
    x_min, x_max, y_min, y_max = plan.zone
    assert (x_min, y_min) == (0.0, 0.0)
    assert x_max == pytest.approx(4.4)
    assert y_max == pytest.approx(1.5)
    assert all(w.x >= -1e-9 and w.y >= -1e-9 for w in plan.waypoints)


def test_старт_можно_сместить():
    """Если взлетать придётся не из угла прямоугольника, сдвиг задаётся конфигом,
    а не правкой кода."""
    plan = plan_rect(length=4.0, width=2.0, altitude=1.5, start_x=0.5, start_y=-0.3)
    assert [(w.x, w.y) for w in plan.waypoints] == [
        (0.5, -0.3), (4.5, -0.3), (4.5, 1.7), (0.5, 1.7), (0.5, -0.3),
    ]


def test_высота_и_курс_одинаковы():
    """Курс не меняется: камера смотрит вниз, а разворот — это рывок ориентации,
    который портит локализацию."""
    plan = plan_rect(length=4.4, width=1.5, altitude=1.5)
    assert {w.z for w in plan.waypoints} == {1.5}
    assert {w.yaw for w in plan.waypoints} == {0.0}


def test_длина_и_время():
    """Путь — полный периметр, включая первый проход от площадки вперёд. Занижение
    здесь означало бы заниженную оценку расхода батареи, а по ней решают, влезает ли
    попытка (регламент 1.3: полигонного времени на «не долетел» нет)."""
    plan = plan_rect(length=4.4, width=1.5, altitude=1.5)
    assert plan.length == pytest.approx(2 * (4.4 + 1.5))
    время = plan.duration(speed=0.45)
    assert время > plan.length / 0.45          # углы тоже стоят времени
    # Батарея заявлена на 11 минут; облёт обязан уложиться с большим запасом
    assert время < 120.0, plan.summary(speed=0.45)


@pytest.mark.parametrize("аргументы", [
    {"length": 0.0}, {"width": -1.0}, {"length": -4.4}, {"side": "вбок"},
])
def test_негодные_параметры(аргументы):
    базовые = dict(length=4.4, width=1.5, altitude=1.5)
    базовые.update(аргументы)
    with pytest.raises(ValueError):
        plan_rect(**базовые)


# ------------------------------------------------------- предупреждения о покрытии

def test_широкий_прямоугольник_ругается_на_дыру_посередине():
    """Длинные проходы идут по КРАЯМ прямоугольника. Если он шире полосы обзора,
    посередине остаётся незаснятый коридор — это потерянная станция и все 8 баллов."""
    plan = plan_rect(length=4.4, width=6.0, altitude=1.5, fov_deg=60.0)
    assert any("в кадр не попадёт" in w for w in plan.warnings), plan.warnings


def test_середина_только_по_краю_кадра():
    """Боевой случай 4.4 x 1.5 на 1.5 м: полоса обзора 1.73 м, середина в кадр
    попадает, но за пределами центральных 50 % (detector.central_frac). Станция
    там найдётся, координата будет грубой — молчать об этом нельзя."""
    plan = plan_rect(length=4.4, width=1.5, altitude=1.5, fov_deg=60.0)
    assert any("КРАЙ кадра" in w for w in plan.warnings), plan.warnings
    # Узкий прямоугольник укладывается в центральную половину кадра и молчит
    узкий = plan_rect(length=4.4, width=0.6, altitude=1.5, fov_deg=60.0)
    assert not узкий.warnings, узкий.warnings


def test_без_fov_геометрия_покрытия_не_проверяется():
    """fov_h_deg не измерен (НЕ ПРОВЕРЕНО в конфиге), поэтому проверка покрытия —
    необязательная надстройка, а не условие построения маршрута."""
    plan = plan_rect(length=4.4, width=6.0, altitude=1.5)
    assert not plan.warnings


# ------------------------------------------------------------------------ конфиг

def test_маршрут_из_боевого_конфига():
    """Умолчание конфига с 01.08.2026 — прямоугольник. Числа берутся из конфига,
    а не из теста: стороны меряются рулеткой перед каждой попыткой (регламент 2.5)."""
    cfg = config_module.load()
    plan = plan_from_config(cfg)
    assert plan.label == "прямоугольник"
    assert plan.altitude == cfg.get_path("flight.h_survey")
    длина = cfg.get_path("rect.length")
    ширина = cfg.get_path("rect.width")
    assert max(w.x for w in plan.waypoints) == pytest.approx(длина)
    assert max(abs(w.y) for w in plan.waypoints) == pytest.approx(ширина)


def test_пакет_не_перекрывает_стороны():
    """Прямоугольник меряем мы рулеткой, автомобиль его не знает. Случайное значение
    в чужой программе увело бы дрон облетать фигуру, которой нет."""
    cfg = config_module.load()
    plan = plan_from_config(cfg, zone_override={"zone_w": 9.0, "zone_h": 9.0})
    assert max(w.x for w in plan.waypoints) == pytest.approx(cfg.get_path("rect.length"))


def test_переключение_режима_стоит_одной_строки():
    """Все три формы достаются из одного боевого конфига одной строкой, и автомат
    миссии их не различает. Важно, что каждая реально строится."""
    cfg = config_module.load()
    assert plan_from_config(cfg).label == "прямоугольник", "умолчание конфига"

    cfg["route"]["mode"] = "survey"
    assert plan_from_config(cfg).label == "змейка"

    cfg["route"]["mode"] = "loop"
    assert plan_from_config(cfg).label == "облёт контура"
