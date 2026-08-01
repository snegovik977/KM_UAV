# -*- coding: utf-8 -*-
"""Генератор маршрута облёта. Три формы, все — чистая геометрия.

    loop    — облёт трассы по контуру: дрон стартует с площадки внутри овала,
              обходит его по четырём угловым точкам и возвращается.
    survey  — змейка, покрывающая прямоугольную зону построчно. Нужна, если станции
              окажутся разбросаны по площади, а не вдоль трассы.
    rect    — прямоугольник ОТ ТОЧКИ ВЗЛЁТА: вперёд, вбок, назад, обратно вбок.
              Отличие от loop: там площадка в центре контура, здесь — в его углу,
              и маршрут заканчивается ровно там, где начался. Основной режим.

Форма выбирается в config.yaml -> route.mode, обе возвращают один и тот же объект
Route, поэтому автомат миссии их не различает.

Модуль не знает ни про SDK, ни про камеру. Поэтому он полностью проверяется тестами
на любой машине, без дрона и без полигона (регламент 1.3: лимит времени утекает,
даже когда полигон простаивает).

Ни одно число здесь не зашито: размеры приходят из config.yaml, а размеры зоны могут
прийти прямо в пакете takeoff от автомобиля. Это требование универсальности
(регламент 2.5) — маршрут обязан строиться под фактическую расстановку.
"""
from __future__ import annotations

import math
from collections import namedtuple

# kind: "line" — рабочий проход, "turn" — разворот между полосами змейки,
#       "corner" — угол контура при облёте трассы
Waypoint = namedtuple("Waypoint", "x y z yaw kind")

STRIP_AXES = ("auto", "x", "y")
DIRECTIONS = ("ccw", "cw")
SIDES = ("left", "right")
MODES = ("loop", "survey", "rect")

# Точки, на которых дрон гасит скорость и разворачивается, стоят дополнительного
# времени сверх «длина пути / скорость».
_ПОВОРОТНЫЕ = ("turn", "corner")


class Route(object):
    """Маршрут плюс всё, чем его можно защитить на разборе решения."""

    def __init__(self, waypoints, label, altitude, zone, warnings,
                 strip_width=None, step=None, lines=None):
        self.waypoints = waypoints
        self.label = label
        self.altitude = altitude
        self.zone = zone                    # (x_min, x_max, y_min, y_max)
        self.warnings = warnings
        self.strip_width = strip_width      # только для змейки
        self.step = step
        self.lines = lines or []

    @property
    def length(self):
        """Длина пути по земле, м."""
        total = 0.0
        for a, b in zip(self.waypoints, self.waypoints[1:]):
            total += math.hypot(b.x - a.x, b.y - a.y)
        return total

    def duration(self, speed, turn_time=2.0):
        """Оценка времени облёта, с. Нужна, чтобы понять, влезает ли попытка
        в батарею (11 минут заявленных, реально меньше), ещё до вылета."""
        повороты = sum(1 for w in self.waypoints if w.kind in _ПОВОРОТНЫЕ)
        return self.length / max(speed, 0.05) + повороты * turn_time

    def summary(self, speed=None):
        text = "%s: %d точек, путь %.1f м" % (self.label, len(self.waypoints), self.length)
        if self.lines:
            text += ", %d полос, шаг %.2f м, полоса %.2f м" % (
                len(self.lines), self.step, self.strip_width)
        if speed:
            text += ", ~%.0f с" % self.duration(speed)
        return text

    def __len__(self):
        return len(self.waypoints)

    def __iter__(self):
        return iter(self.waypoints)


# ------------------------------------------------------- облёт трассы по контуру

def plan_loop(length, width, altitude, center_x=0.0, center_y=0.0, yaw=0.0,
              laps=1, direction="ccw", margin=0.0):
    """Обход трассы по четырём угловым точкам.

    length  — размер контура вдоль X (вперёд), м
    width   — размер контура вдоль Y (влево), м
    center_x, center_y — центр контура относительно точки взлёта. Ноль означает,
              что дрон стартует из середины трассы (посадочная площадка внутри овала).
    margin  — на сколько раздвинуть контур наружу. Положительный отодвигает дрон
              от разметки, отрицательный — заводит внутрь овала.
    laps    — сколько кругов подряд.

    Возвращает Route. Последняя точка совпадает с первой: круг замкнут, и дальше
    автомат миссии сам уводит дрон на посадочную площадку.
    """
    if length <= 0 or width <= 0:
        raise ValueError("размеры контура должны быть положительными: %r x %r"
                         % (length, width))
    if laps < 1:
        raise ValueError("кругов должно быть не меньше одного, получено %r" % (laps,))
    if direction not in DIRECTIONS:
        raise ValueError("direction должно быть одним из %s, получено %r"
                         % (DIRECTIONS, direction))

    warnings = []
    полудлина = length / 2.0 + margin
    полуширина = width / 2.0 + margin
    if полудлина <= 0 or полуширина <= 0:
        raise ValueError("margin=%r съел весь контур" % (margin,))

    # Наша СК: X вперёд, Y влево. Обход против часовой стрелки, если смотреть сверху:
    # вперёд-вправо -> вперёд-влево -> назад-влево -> назад-вправо.
    углы = [
        (center_x + полудлина, center_y - полуширина),
        (center_x + полудлина, center_y + полуширина),
        (center_x - полудлина, center_y + полуширина),
        (center_x - полудлина, center_y - полуширина),
    ]
    if direction == "cw":
        углы = [углы[0]] + list(reversed(углы[1:]))

    расстояние_до_старта = math.hypot(углы[0][0], углы[0][1])
    if расстояние_до_старта > max(length, width):
        warnings.append(
            "первый угол в %.1f м от точки взлёта — проверьте center_x/center_y: "
            "площадка должна быть внутри контура" % расстояние_до_старта)

    waypoints = []
    for _ in range(laps):
        for x, y in углы:
            waypoints.append(Waypoint(round(x, 3), round(y, 3), altitude, yaw, "corner"))
    # Замыкаем круг: без этого последний отрезок контура остаётся непройденным.
    x, y = углы[0]
    waypoints.append(Waypoint(round(x, 3), round(y, 3), altitude, yaw, "corner"))

    zone = (center_x - полудлина, center_x + полудлина,
            center_y - полуширина, center_y + полуширина)
    подпись = "облёт контура" + (" x%d" % laps if laps > 1 else "")
    return Route(waypoints, подпись, altitude, zone, warnings)


# ------------------------------------------------- прямоугольник от точки взлёта

def plan_rect(length, width, altitude, side="left", yaw=0.0,
              start_x=0.0, start_y=0.0, fov_deg=None):
    """Прямоугольник, у которого один угол — точка взлёта.

    length  — сколько пройти ВПЕРЁД (вдоль нашего X), м
    width   — сколько пройти ВБОК, м
    side    — в какую сторону уходит короткая сторона: left (наш +Y) или right (−Y)
    start_x, start_y — где стоит дрон относительно точки фиксации нуля. Обычно 0, 0:
              ноль СК фиксируется в arm(), то есть ровно на площадке «H».

    Порядок обхода: вперёд на length, вбок на width, назад на length, вбок обратно.
    Последняя точка — сама точка взлёта, поэтому RETURN после облёта не гонит дрон
    через всю площадку, а подтверждает уже занятую позицию.

    Отличие от plan_loop: там контур центрирован на площадке (дрон стартует ВНУТРИ
    овала) и круг замыкается лишним пятым перелётом в первый угол. Здесь площадка —
    это угол, и замыкать нечего: четвёртая точка и есть начало.

    fov_deg — необязательный угол обзора. Если задан, проверяется, что короткая
    сторона не шире полосы обзора: иначе середина прямоугольника не попадает в кадр
    ни на одном из двух длинных проходов, и станция в ней теряется вместе с 8 баллами.
    """
    if length <= 0 or width <= 0:
        raise ValueError("стороны прямоугольника должны быть положительными: %r x %r"
                         % (length, width))
    if side not in SIDES:
        raise ValueError("side должно быть одним из %s, получено %r" % (SIDES, side))

    warnings = []
    знак = 1.0 if side == "left" else -1.0
    поперёк = start_y + знак * width

    if fov_deg is not None:
        полоса = strip_width(altitude, float(fov_deg))
        if width > полоса + 1e-9:
            # Длинные стороны идут по краям прямоугольника, между ними расстояние width.
            # Камера накрывает полосу шириной `полоса` вокруг каждой — если width больше,
            # посередине остаётся незаснятый коридор.
            warnings.append(
                "короткая сторона %.2f м шире полосы обзора %.2f м: середина "
                "прямоугольника в кадр не попадёт. Поднять h_survey или сузить rect.width"
                % (width, полоса))
        elif width > полоса / 2.0 + 1e-9:
            # В оценку координат идут только центральные 50 % кадра
            # (detector.central_frac): станция посередине видна, но по краю кадра,
            # и её координата держится на худших наблюдениях — это 4 балла за «<30 см».
            warnings.append(
                "середина прямоугольника попадает только на КРАЙ кадра "
                "(сторона %.2f м против половины полосы %.2f м): станция там найдётся, "
                "но координата будет грубой" % (width, полоса / 2.0))

    # Первая точка — сама площадка, хотя дрон уже стоит на ней после взлёта. Она здесь
    # не ради перелёта (он мгновенный), а ради честного plan.length: без неё первый
    # проход вперёд не попадал бы в оценку пути, и duration() занижала бы расход
    # батареи на треть периметра — а по этой оценке решают, влезает ли попытка.
    углы = [
        (start_x, start_y),                 # площадка «H», она же последний угол
        (start_x + length, start_y),        # вперёд
        (start_x + length, поперёк),        # вбок
        (start_x, поперёк),                 # назад
        (start_x, start_y),                 # вбок обратно, в точку взлёта
    ]
    waypoints = [Waypoint(round(x, 3), round(y, 3), altitude, yaw, "corner")
                 for x, y in углы]

    zone = (min(start_x, start_x + length), max(start_x, start_x + length),
            min(start_y, поперёк), max(start_y, поперёк))
    return Route(waypoints, "прямоугольник", altitude, zone, warnings)


# ------------------------------------------------------------------------ змейка

def strip_width(altitude, fov_deg):
    """Ширина полосы обзора на земле при съёмке вниз."""
    if altitude <= 0:
        raise ValueError("высота должна быть положительной, получено %r" % (altitude,))
    if not 0 < fov_deg < 180:
        raise ValueError("угол обзора вне (0, 180), получено %r" % (fov_deg,))
    return 2.0 * altitude * math.tan(math.radians(fov_deg) / 2.0)


def _line_positions(span_min, span_max, width, step, margin, warnings):
    """Где провести полосы вдоль поперечной оси, чтобы покрыть [span_min, span_max].

    Крайние полосы отстоят от границы зоны на половину ширины обзора: тогда камера
    захватывает край, и дрону не нужно вылетать за пределы зоны.
    """
    span = span_max - span_min
    half = width / 2.0
    inset = min(half, span / 2.0)

    if margin > inset + 1e-9:
        # Отступ безопасности не даёт подойти к краю достаточно близко: полоса вдоль
        # границы останется незаснятой. Молчать об этом нельзя — это прямая потеря
        # станции, а с ней и всех 8 баллов подзадачи 2.3.2.
        warnings.append(
            "margin=%.2f м больше половины полосы обзора (%.2f м): край зоны шириной "
            "до %.2f м в кадр не попадёт. Поднять высоту или уменьшить margin."
            % (margin, half, margin - inset))
        inset = margin

    lo = span_min + inset
    hi = span_max - inset
    if hi - lo < 1e-6:
        return [(span_min + span_max) / 2.0]

    count = max(2, int(math.ceil((hi - lo) / step - 1e-9)) + 1)
    spacing = (hi - lo) / (count - 1)
    return [lo + spacing * i for i in range(count)]


def plan_survey(width, height, altitude, fov_deg, overlap=0.3, margin=0.4,
                center_x=None, center_y=0.0, strip_axis="auto", yaw=0.0):
    """Змейка, покрывающая прямоугольную зону.

    Ширина полосы на земле W = 2*h*tan(FOV/2), шаг между проходами W*(1-overlap).
    Перекрытие 30 % закрывает дрейф позиции и то, что объект у края кадра
    локализуется хуже (в оценку координат берутся только центральные 50 % кадра).

    center_x по умолчанию смещает зону вперёд на полширины: дрон стартует у ближнего
    края зоны, а не в её центре.
    """
    if width <= 0 or height <= 0:
        raise ValueError("размеры зоны должны быть положительными: %r x %r" % (width, height))
    if not 0.0 <= overlap < 1.0:
        raise ValueError("перекрытие вне [0, 1): %r" % (overlap,))
    if strip_axis not in STRIP_AXES:
        raise ValueError("strip_axis должно быть одним из %s, получено %r"
                         % (STRIP_AXES, strip_axis))
    if margin < 0:
        raise ValueError("margin отрицательный: %r" % (margin,))

    center_x = width / 2.0 if center_x is None else center_x
    x_min, x_max = center_x - width / 2.0, center_x + width / 2.0
    y_min, y_max = center_y - height / 2.0, center_y + height / 2.0

    warnings = []
    band = strip_width(altitude, fov_deg)
    step = band * (1.0 - overlap)

    if strip_axis == "auto":
        strip_axis = "x" if width >= height else "y"

    # Проходы идут вдоль along-оси, сдвигаются по cross-оси.
    if strip_axis == "x":
        along_min, along_max = x_min, x_max
        cross_min, cross_max = y_min, y_max
    else:
        along_min, along_max = y_min, y_max
        cross_min, cross_max = x_min, x_max

    # Вдоль полосы дрон тоже не подходит к границе вплотную: край добирает камера.
    along_inset = min(margin, (along_max - along_min) / 2.0)
    along_lo = along_min + along_inset
    along_hi = along_max - along_inset

    lines = _line_positions(cross_min, cross_max, band, step, margin, warnings)
    if len(lines) == 1:
        warnings.append("зона уже одной полосы обзора — облёт вырождается в один проход")

    waypoints = []
    for index, cross in enumerate(lines):
        # Змейка: чётные полосы вперёд, нечётные назад. Так дрон не возвращается
        # вхолостую в начало каждой полосы.
        ends = (along_lo, along_hi) if index % 2 == 0 else (along_hi, along_lo)
        for position, along in enumerate(ends):
            if strip_axis == "x":
                x, y = along, cross
            else:
                x, y = cross, along
            # Первая точка новой полосы — разворот, вторая — конец рабочего прохода.
            kind = "turn" if (position == 0 and index > 0) else "line"
            waypoints.append(Waypoint(round(x, 3), round(y, 3), altitude, yaw, kind))

    return Route(waypoints, "змейка", altitude, (x_min, x_max, y_min, y_max), warnings,
                 strip_width=band, step=step, lines=lines)


# ------------------------------------------------------------------------ конфиг

def plan_from_config(cfg, zone_override=None):
    """Маршрут по config.yaml.

    zone_override — размеры из пакета takeoff, если автомобиль их прислал: параметры
    попытки важнее вчерашних значений в конфиге.
    """
    mode = str(cfg.get("route", {}).get("mode", "loop"))
    if mode not in MODES:
        raise ValueError("route.mode должно быть одним из %s, получено %r" % (MODES, mode))
    altitude = float(cfg["flight"]["h_survey"])

    if mode == "loop":
        loop = cfg["loop"]
        length = float(loop["length"])
        width = float(loop["width"])
        # Габариты трассы меряем мы рулеткой, автомобиль их не знает. Поэтому пакет
        # takeoff по умолчанию их НЕ перекрывает: случайное значение в чужой программе
        # увело бы дрон обходить контур, которого нет. Включается явно, если
        # договоримся, что размеры приходят от автомобиля.
        if zone_override and loop.get("use_packet_zone", False):
            length = float(zone_override.get("zone_w") or length)
            width = float(zone_override.get("zone_h") or width)
        return plan_loop(
            length=length,
            width=width,
            altitude=altitude,
            center_x=float(loop.get("center_x", 0.0)),
            center_y=float(loop.get("center_y", 0.0)),
            laps=int(loop.get("laps", 1)),
            direction=str(loop.get("direction", "ccw")),
            margin=float(loop.get("margin", 0.0)),
        )

    if mode == "rect":
        rect = cfg["rect"]
        # Стороны прямоугольника меряет рулеткой оператор, автомобиль их не знает —
        # тот же довод, по которому пакет takeoff не перекрывает габариты контура.
        return plan_rect(
            length=float(rect["length"]),
            width=float(rect["width"]),
            altitude=altitude,
            side=str(rect.get("side", "left")),
            start_x=float(rect.get("start_x", 0.0)),
            start_y=float(rect.get("start_y", 0.0)),
            fov_deg=float(cfg["zone"]["fov_h_deg"]),
        )

    zone = cfg["zone"]
    width = float(zone["width"])
    height = float(zone["height"])
    if zone_override:
        width = float(zone_override.get("zone_w") or width)
        height = float(zone_override.get("zone_h") or height)
    return plan_survey(
        width=width,
        height=height,
        altitude=altitude,
        fov_deg=float(zone["fov_h_deg"]),
        overlap=float(zone["overlap"]),
        margin=float(zone["margin"]),
        center_x=zone.get("center_x"),
        center_y=float(zone.get("center_y", 0.0)),
        strip_axis=str(zone.get("strip_axis", "auto")),
    )
