#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Проверка привязки карты: правильно ли померены map.px_per_m, origin_px_* и yaw_deg.

Зачем отдельный инструмент. Привязка — единственное место, где ошибка НЕ ВИДНА
в логе: координаты станций в метрах будут верные, дрон слетает правильно, а кружки
на карте лягут не туда. Условие 8 баллов подзадачи 2.3.2 — что визуализатор
показывает станции на карте, поэтому ошибка здесь стоит столько же, сколько
пропущенная станция. Проверяется глазами, за десять секунд и без дрона.

    python tools/check_map.py                      # нарисовать зону и оси на карте
    python tools/check_map.py --out проверка.png   # и сохранить картинку
    python tools/check_map.py --yaw-to 640,300     # какой yaw_deg даёт этот ориентир

Что смотреть на картинке:
  жёлтая точка   где стоит дрон в момент arm (origin_px_x, origin_px_y);
  зелёная стрелка ось X дрона — ВПЕРЁД по курсу взлёта, длиной 1 метр;
  синяя стрелка  ось Y — ВЛЕВО, тоже метр. По длине стрелок проверяется px_per_m:
                 метр на картинке обязан выглядеть как метр на площадке;
  красная рамка  зона облёта из секции zone. Обязана лечь на ту часть площадки,
                 которую дрон реально будет облетать.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import imgio

_ЗДЕСЬ = os.path.dirname(os.path.abspath(__file__))
_КОРЕНЬ = os.path.dirname(_ЗДЕСЬ)
for _путь in (os.path.join(_КОРЕНЬ, "квадрокоптер"),
              os.path.join(_КОРЕНЬ, "распределительный хаб"), _КОРЕНЬ):
    if os.path.isdir(_путь) and _путь not in sys.path:
        sys.path.insert(0, _путь)

try:
    import cv2
except ImportError:
    cv2 = None


def yaw_по_ориентиру(anchor, u, v):
    """Какой yaw_deg вписать, если нос дрона смотрит на пиксель (u, v).

    yaw_deg — поворот оси X дрона относительно направления «вверх по картинке»,
    против часовой стрелки. Вывод прямо из MapAnchor.to_px: единичный вектор
    «вперёд» уходит в (origin_u - sin(yaw)*s, origin_v - cos(yaw)*s).
    """
    return math.degrees(math.atan2(anchor.origin_px[0] - u, anchor.origin_px[1] - v))


def main():
    parser = argparse.ArgumentParser(description="Проверка привязки карты")
    parser.add_argument("--config", default=None, help="путь к config.yaml")
    parser.add_argument("--map", default=None, help="картинка карты полигона")
    parser.add_argument("--out", default=None, help="куда сохранить проверочную картинку")
    parser.add_argument("--yaw-to", default=None, metavar="U,V",
                        help="пиксель, на который смотрит НОС дрона: "
                             "печатает готовый yaw_deg и выходит")
    args = parser.parse_args()

    import config as config_module
    from protocol import MapAnchor

    cfg = config_module.load(args.config or os.path.join(
        _КОРЕНЬ, "квадрокоптер", "config.yaml"))

    путь = args.map or str(cfg.get("map", {}).get("image", ""))
    if путь and not os.path.isabs(путь):
        путь = os.path.join(_КОРЕНЬ, "распределительный хаб", путь)
    карта = imgio.imread(путь) if путь else None
    if карта is None:
        print("карта %s не читается — без неё проверять нечего" % путь)
        return 1
    высота, ширина = карта.shape[:2]
    anchor = MapAnchor.from_config(cfg, width=ширина, height=высота)

    if args.yaw_to:
        u, v = (float(ч) for ч in args.yaw_to.replace(" ", "").split(","))
        print("map.yaw_deg: %.1f" % yaw_по_ориентиру(anchor, u, v))
        return 0

    print("картинка: %d x %d пикселей" % (ширина, высота))
    print("масштаб:  %.1f пикс/м  ->  вся картинка %.2f x %.2f м"
          % (anchor.px_per_m, ширина / anchor.px_per_m, высота / anchor.px_per_m))
    print("точка взлёта: пиксель (%d, %d)"
          % (anchor.origin_px[0], anchor.origin_px[1]))
    print("поворот карты: %.1f град (0 = нос дрона смотрит ВВЕРХ по картинке)"
          % anchor.yaw_deg)

    z = cfg["zone"]
    w, h = float(z["width"]), float(z["height"])
    cx, cy = float(z.get("center_x", w / 2.0)), float(z.get("center_y", 0.0))
    углы = [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2),
            (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)]

    print("\nзона облёта %.2f x %.2f м, центр (%+.2f, %+.2f):" % (w, h, cx, cy))
    за_краем = 0
    пиксели = []
    for x, y in углы:
        u, v = anchor.to_px(x, y)
        пиксели.append((u, v))
        внутри = anchor.inside(u, v)
        за_краем += 0 if внутри else 1
        print("  угол x=%+.2f y=%+.2f -> пиксель (%d, %d)%s"
              % (x, y, u, v, "" if внутри else "   ЗА КРАЕМ КАРТИНКИ"))
    if за_краем:
        print("\n⚠️ %d угла зоны вне картинки. Либо зона больше площадки, либо "
              "неверны px_per_m / origin_px_*. Станции в этих местах визуализатор "
              "не покажет вообще (MapAnchor.inside отсеет пакет)." % за_краем)
    else:
        print("\nвсе углы зоны внутри картинки")

    if cv2 is None:
        print("нет OpenCV — картинку не рисую")
        return 0

    for i in range(4):
        cv2.line(карта, пиксели[i], пиксели[(i + 1) % 4], (0, 0, 255), 3)
    u0, v0 = anchor.to_px(0.0, 0.0)
    cv2.arrowedLine(карта, (u0, v0), anchor.to_px(1.0, 0.0), (0, 255, 0), 4,
                    tipLength=0.25)
    cv2.arrowedLine(карта, (u0, v0), anchor.to_px(0.0, 1.0), (255, 128, 0), 4,
                    tipLength=0.25)
    cv2.circle(карта, (u0, v0), 10, (0, 255, 255), -1)
    # Подписи латиницей: cv2.putText кириллицу не умеет и рисует «???».
    cv2.putText(карта, "X vpered 1 m", anchor.to_px(1.05, 0.0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(карта, "Y vlevo 1 m", anchor.to_px(0.0, 1.7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)

    if args.out:
        if imgio.imwrite(args.out, карта):
            print("сохранено: %s" % args.out)
        else:
            print("не удалось сохранить %s" % args.out)
        return 0

    cv2.imshow("Proverka privyazki karty", карта)
    print("\nокно открыто, любая клавиша — выход")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
