# -*- coding: utf-8 -*-
"""Змейка облёта. Главный тест — покрытие: пропущенная полоса означает пропущенную
станцию, а совпадение их числа с реальным — условие всех 8 баллов подзадачи 2.3.2.

Глазами по списку точек ошибку знака в шаге не увидеть, поэтому зона растеризуется
сеткой и проверяется каждая ячейка.
"""
from __future__ import annotations

import math

import pytest

import config as config_module
from survey import Waypoint, plan_from_config, plan_survey, strip_width


ШАГ_СЕТКИ = 0.05      # 5 см: мельче размера станции, крупнее ошибки округления


# Кадр прямоугольный, поэтому в конце полосы камера захватывает и то, что впереди
# дрона. Вертикальный угол обзора мы узнаем только после калибровки (пункт 2 плана),
# поэтому берём консервативную оценку: матрица 4:3, вдоль полосы след короче
# поперечного в 0.75 раза. Занижение здесь безопасно — тест строже реальности.
ПРОДОЛЬНАЯ_ДОЛЯ = 0.75


def покрытие(plan, шаг=ШАГ_СЕТКИ):
    """Доля ячеек зоны, попавших хотя бы в один кадр вдоль маршрута.

    След камеры вдоль отрезка — прямоугольник: поперёк strip_width, вдоль — длина
    отрезка плюс продольный след с обоих концов. Развороты тоже снимают, но
    на них рассчитывать нельзя.
    """
    x_min, x_max, y_min, y_max = plan.zone
    half = plan.strip_width / 2.0
    along_half = half * ПРОДОЛЬНАЯ_ДОЛЯ
    отрезки = list(zip(plan.waypoints, plan.waypoints[1:]))

    всего = покрыто = 0
    y = y_min + шаг / 2.0
    while y < y_max:
        x = x_min + шаг / 2.0
        while x < x_max:
            всего += 1
            for a, b in отрезки:
                if _в_следе(x, y, a, b, half, along_half):
                    покрыто += 1
                    break
            x += шаг
        y += шаг
    return покрыто / float(всего or 1)


def _в_следе(px, py, a, b, half, along_half):
    """Точка внутри прямоугольного следа камеры вдоль отрезка a-b."""
    ax, ay, bx, by = a.x, a.y, b.x, b.y
    dx, dy = bx - ax, by - ay
    длина = math.hypot(dx, dy)
    if длина < 1e-12:
        return abs(px - ax) <= half and abs(py - ay) <= half
    ux, uy = dx / длина, dy / длина
    вдоль = (px - ax) * ux + (py - ay) * uy
    поперёк = abs(-(px - ax) * uy + (py - ay) * ux)
    return -along_half - 1e-9 <= вдоль <= длина + along_half + 1e-9 and поперёк <= half + 1e-9


# ------------------------------------------------------------------- геометрия

def test_ширина_полосы():
    # 90° обзора с 2 м дают ровно 4 м на земле: tan(45°) = 1
    assert strip_width(2.0, 90.0) == pytest.approx(4.0)
    assert strip_width(1.0, 60.0) == pytest.approx(2 * math.tan(math.radians(30)))
    # Выше — шире: иначе перепутан знак или радианы с градусами
    assert strip_width(3.0, 60.0) > strip_width(2.0, 60.0)


@pytest.mark.parametrize("высота, fov", [(0.0, 60.0), (-1.0, 60.0), (2.0, 0.0), (2.0, 180.0)])
def test_негодная_геометрия_отвергается(высота, fov):
    with pytest.raises(ValueError):
        strip_width(высота, fov)


@pytest.mark.parametrize("аргументы", [
    {"width": 0.0}, {"height": -1.0}, {"overlap": 1.0}, {"overlap": -0.1},
    {"margin": -0.5}, {"strip_axis": "диагональ"},
])
def test_негодные_параметры_змейки(аргументы):
    базовые = dict(width=6.0, height=6.0, altitude=2.0, fov_deg=60.0)
    базовые.update(аргументы)
    with pytest.raises(ValueError):
        plan_survey(**базовые)


# ------------------------------------------------------------------- покрытие

@pytest.mark.parametrize("ширина, длина, высота, fov", [
    (6.0, 6.0, 2.0, 60.0),
    (6.0, 4.0, 2.0, 60.0),
    (4.0, 8.0, 2.0, 70.0),
    (10.0, 3.0, 2.5, 55.0),
    (3.0, 3.0, 1.2, 60.0),      # низко и узко: много тесных полос
    (7.0, 5.5, 2.0, 62.0),      # некратные размеры: полосы не лягут ровно
])
def test_зона_покрыта_полностью(ширина, длина, высота, fov):
    plan = plan_survey(width=ширина, height=длина, altitude=высота, fov_deg=fov,
                       overlap=0.3, margin=0.4)
    assert not plan.warnings, plan.warnings
    assert покрытие(plan) == pytest.approx(1.0), plan.summary()


def test_перекрытие_соблюдается():
    """Шаг между полосами не должен превышать (1 - overlap) от ширины обзора,
    иначе перекрытия 30 % на деле нет и края полос локализуются плохо."""
    plan = plan_survey(width=6.0, height=6.0, altitude=2.0, fov_deg=60.0, overlap=0.3)
    расстояния = [b - a for a, b in zip(plan.lines, plan.lines[1:])]
    assert расстояния, "змейка из одной полосы — нечего проверять"
    assert max(расстояния) <= plan.step + 1e-9
    # Полосы равномерны: неравномерный шаг означает ошибку в раскладке
    assert max(расстояния) - min(расстояния) < 1e-9


def test_маленькая_зона_вырождается_в_один_проход():
    plan = plan_survey(width=1.0, height=1.0, altitude=2.0, fov_deg=90.0)
    assert len(plan.lines) == 1
    assert any("один проход" in w for w in plan.warnings)
    assert покрытие(plan) == pytest.approx(1.0)


def test_большой_отступ_честно_ругается():
    """margin больше половины полосы обзора оставляет край зоны неснятым.
    Молчать об этом нельзя: это потерянная станция."""
    plan = plan_survey(width=8.0, height=8.0, altitude=1.0, fov_deg=40.0, margin=1.5)
    assert any("край зоны" in w for w in plan.warnings)
    assert покрытие(plan) < 1.0


# -------------------------------------------------------------------- маршрут

def test_дрон_не_выходит_за_зону():
    """Вылет за границу зоны — это и риск (сетка, стены), и лишний путь."""
    plan = plan_survey(width=6.0, height=4.0, altitude=2.0, fov_deg=60.0, margin=0.4)
    x_min, x_max, y_min, y_max = plan.zone
    for точка in plan:
        assert x_min - 1e-9 <= точка.x <= x_max + 1e-9
        assert y_min - 1e-9 <= точка.y <= y_max + 1e-9


def test_проходы_чередуют_направление():
    """Змейка, а не гребёнка: иначе дрон возвращается в начало каждой полосы вхолостую
    и тратит вдвое больше батареи."""
    plan = plan_survey(width=6.0, height=6.0, altitude=2.0, fov_deg=60.0)
    точки = plan.waypoints
    направления = []
    for i in range(0, len(точки) - 1, 2):
        направления.append(math.copysign(1.0, точки[i + 1].x - точки[i].x))
    assert all(a * b < 0 for a, b in zip(направления, направления[1:])), направления


def test_ось_проходов_вдоль_длинной_стороны():
    """auto обязан выбрать длинную сторону: меньше разворотов — меньше времени."""
    широкая = plan_survey(width=10.0, height=3.0, altitude=2.0, fov_deg=60.0)
    узкая = plan_survey(width=3.0, height=10.0, altitude=2.0, fov_deg=60.0)
    # У широкой зоны проходы идут вдоль X: у концов полосы разный X, одинаковый Y
    assert широкая.waypoints[0].y == pytest.approx(широкая.waypoints[1].y)
    assert узкая.waypoints[0].x == pytest.approx(узкая.waypoints[1].x)
    # Принудительный выбор оси переопределяет auto
    принудительно = plan_survey(width=10.0, height=3.0, altitude=2.0, fov_deg=60.0,
                                strip_axis="y")
    assert принудительно.waypoints[0].x == pytest.approx(принудительно.waypoints[1].x)


def test_высота_и_курс_одинаковы_на_всём_маршруте():
    """Курс не меняется: камера смотрит вниз, разворачиваться незачем, а каждый
    разворот — это время и рывок ориентации, который портит локализацию."""
    plan = plan_survey(width=6.0, height=6.0, altitude=2.0, fov_deg=60.0)
    assert {точка.z for точка in plan} == {2.0}
    assert {точка.yaw for точка in plan} == {0.0}


def test_разворотов_на_один_меньше_числа_полос():
    plan = plan_survey(width=6.0, height=6.0, altitude=2.0, fov_deg=60.0)
    развороты = sum(1 for точка in plan if точка.kind == "turn")
    assert развороты == len(plan.lines) - 1


def test_зона_начинается_у_точки_взлёта():
    """По умолчанию дрон стартует у ближнего края зоны, а не в её центре:
    он взлетает с площадки, а зона лежит перед ним."""
    plan = plan_survey(width=6.0, height=4.0, altitude=2.0, fov_deg=60.0)
    x_min, x_max, y_min, y_max = plan.zone
    assert x_min == pytest.approx(0.0)
    assert x_max == pytest.approx(6.0)
    assert (y_min + y_max) / 2 == pytest.approx(0.0)
    # Центр можно сместить явно
    по_центру = plan_survey(width=6.0, height=4.0, altitude=2.0, fov_deg=60.0, center_x=0.0)
    assert по_центру.zone[0] == pytest.approx(-3.0)


# --------------------------------------------------------------------- бюджет

def test_оценка_пути_и_времени():
    plan = plan_survey(width=6.0, height=6.0, altitude=2.0, fov_deg=60.0)
    # Зона 6x6 при полосе ~2.3 м -> около 4 проходов по 5+ м
    assert 20.0 < plan.length < 45.0
    время = plan.duration(speed=0.45)
    assert время > plan.length / 0.45          # развороты тоже стоят времени
    # Батарея заявлена на 11 минут; облёт обязан уложиться с запасом
    assert время < 300.0, plan.summary(speed=0.45)


def test_пустой_путь_не_ломает_длину():
    plan_ = plan_survey(width=1.0, height=1.0, altitude=2.0, fov_deg=90.0)
    assert plan_.length >= 0.0
    assert isinstance(plan_.waypoints[0], Waypoint)


# --------------------------------------------------------------------- конфиг

def змеиный_конфиг():
    """Боевой конфиг как есть: с 01.08.2026 змейка и есть умолчание — станции
    разбросаны по площади, и контур трассы их не накроет. Режим всё равно задаётся
    явно: тест не должен молча начать проверять контур, если умолчание снова сменят."""
    cfg = config_module.load()
    cfg["route"]["mode"] = "survey"
    return cfg


def test_план_из_боевого_конфига():
    cfg = змеиный_конфиг()
    plan = plan_from_config(cfg)
    assert not plan.warnings, plan.warnings
    assert покрытие(plan) == pytest.approx(1.0)
    assert plan.altitude == cfg.get_path("flight.h_survey")


def test_размеры_из_пакета_takeoff_важнее_конфига():
    """Автомобиль может прислать фактические размеры зоны — они свежее вчерашнего
    конфига. Захардкоженная под вчерашнюю расстановку зона — риск по регламенту 2.5."""
    plan = plan_from_config(змеиный_конфиг(), zone_override={"zone_w": 4.0, "zone_h": 3.0})
    x_min, x_max, y_min, y_max = plan.zone
    assert x_max - x_min == pytest.approx(4.0)
    assert y_max - y_min == pytest.approx(3.0)
