#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Прогон перцепции по записи полёта: что дрон отправил бы, не вылетая.

Прямое следствие механики лимита времени (регламент 1.3): время утекает, даже когда
полигон простаивает. Значит подбор порогов, проверка калибровки и разбор «почему
станций оказалось шесть вместо четырёх» обязаны делаться по записи, а полигон
тратиться только на попытки.

На вход — пара файлов от tools/record.py:

    flight_20260731_142530.mp4      кадры, как их видела перцепция
    flight_20260731_142530.jsonl    снимки состояния миссии

Совмещаются они по полю frame в jsonl: там записан номер кадра на момент снимка.

    python3 tools/replay.py flight_20260731_142530.jsonl
    python3 tools/replay.py flight_*.jsonl --set detector.min_area_m2=0.05
    python3 tools/replay.py flight_*.jsonl --out разбор/     # кадры с рамками
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

_ЗДЕСЬ = os.path.dirname(os.path.abspath(__file__))
_КОРЕНЬ = os.path.dirname(_ЗДЕСЬ)
for _путь in (_ЗДЕСЬ, _КОРЕНЬ, os.path.join(_КОРЕНЬ, "квадрокоптер"),
              os.path.join(_КОРЕНЬ, "распределительный хаб")):
    if os.path.isdir(_путь) and _путь not in sys.path:
        sys.path.insert(0, _путь)


def прочитать_телеметрию(path):
    """jsonl -> список записей. Кривые строки пропускаем: обрыв записи при жёсткой
    посадке — штатная ситуация, и терять из-за неё весь разбор незачем."""
    записи = []
    with io.open(path, "r", encoding="utf-8") as f:
        for строка in f:
            строка = строка.strip()
            if not строка:
                continue
            try:
                записи.append(json.loads(строка))
            except ValueError:
                continue
    return записи


def поза_для_кадра(записи, номер):
    """Ближайшая по номеру кадра запись телеметрии.

    Телеметрия пишется реже кадров (5 Гц против 15), поэтому берётся последняя
    запись, сделанная не позже этого кадра.
    """
    подходящая = None
    for запись in записи:
        if запись.get("frame", 0) <= номер:
            подходящая = запись
        else:
            break
    return подходящая


def применить_правки(cfg, правки):
    """--set detector.min_area_m2=0.05 -> cfg["detector"]["min_area_m2"] = 0.05.

    Подбор порогов идёт ключами, а не правкой config.yaml: иначе рано или поздно
    подобранное значение уедет на борт вместе с отладочным мусором.
    """
    for правка in правки or []:
        путь, _, значение = правка.partition("=")
        части = путь.strip().split(".")
        узел = cfg
        for часть in части[:-1]:
            узел = узел.setdefault(часть, {})
        try:
            узел[части[-1]] = float(значение)
        except ValueError:
            узел[части[-1]] = значение.strip()
        print("[разбор] %s = %r" % (путь.strip(), узел[части[-1]]))


class СобирающийТранспорт(object):
    """Вместо хаба — список: разбор не должен требовать сети."""

    def __init__(self):
        self.отправленные = []

    def send(self, msg):
        self.отправленные.append(msg)


def main():
    parser = argparse.ArgumentParser(description="Прогон перцепции по записи полёта")
    parser.add_argument("log", help="файл flight_*.jsonl (видео ищется рядом)")
    parser.add_argument("--video", default=None, help="видео, если имя не совпадает")
    parser.add_argument("--config", default=None)
    parser.add_argument("--task", default="2.3.2")
    parser.add_argument("--set", action="append", dest="правки", metavar="ПУТЬ=ЗНАЧЕНИЕ",
                        help="перекрыть параметр конфига, можно несколько раз")
    parser.add_argument("--out", default=None,
                        help="куда класть кадры с рамками (по одному на детекцию)")
    parser.add_argument("--stride", type=int, default=1, help="брать каждый N-й кадр")
    args = parser.parse_args()

    import cv2

    import config as config_module
    import tasks as tasks_module
    from fusion import StationRegistry
    from mission import MissionState
    from perception.calib import from_config as калибровка_из_конфига
    from perception.detector import ClassicDetector, create_detector
    from perception.pipeline import Perception
    from protocol import MessageFactory

    видео = args.video or (os.path.splitext(args.log)[0] + ".mp4")
    if not os.path.exists(видео):
        print("[разбор] нет видео %s — по одной телеметрии перцепцию не прогнать" % видео)
        return 1

    cfg = config_module.load(args.config)
    применить_правки(cfg, args.правки)

    записи = прочитать_телеметрию(args.log)
    if not записи:
        print("[разбор] телеметрия пуста: %s" % args.log)
        return 1
    print("[разбор] %s: %d записей, %.1f с" % (args.log, len(записи),
                                               записи[-1].get("t", 0.0)))

    state = MissionState()
    transport = СобирающийТранспорт()
    task = tasks_module.TaskProfile(args.task)
    калибровка = калибровка_из_конфига(cfg, log=print)
    детектор = create_detector(cfg, log=print)
    реестр = StationRegistry(cfg, transport=transport, factory=MessageFactory("drone"),
                             state=state, task=task, log=print)
    перцепция = Perception(cfg, state, реестр, калибровка, детектор, log=print,
                           # В записи есть кадры со всех этапов полёта, а не только
                           # с облёта; ограничение по состоянию оставляем как в полёте.
                           only_in_survey=True)
    перцепция.every_n = 1        # по записи считаем каждый кадр: спешить некуда

    if args.out and not os.path.isdir(args.out):
        os.makedirs(args.out)

    поток = cv2.VideoCapture(видео)
    номер = 0
    сохранено = 0
    try:
        while True:
            есть, кадр = поток.read()
            if not есть:
                break
            номер += 1
            if args.stride > 1 and номер % args.stride:
                continue

            запись = поза_для_кадра(записи, номер)
            if запись is None:
                continue
            state.update(state=запись.get("state", "SURVEY"),
                         position=tuple(запись.get("position", (0.0, 0.0, 0.0))),
                         roll=запись.get("roll", 0.0), pitch=запись.get("pitch", 0.0),
                         yaw=запись.get("yaw", 0.0),
                         origin=tuple(запись.get("origin", (0.0, 0.0, 0.0))))

            детекции = перцепция.process(кадр)
            if детекции and args.out:
                перцепция.draw(кадр, детекции)
                cv2.imwrite(os.path.join(args.out, "кадр_%05d.jpg" % номер), кадр)
                сохранено += 1
    finally:
        поток.release()

    # То же, что миссия делает в конце облёта: дослать уточнённые координаты
    # перед recon_done. Без этого разбор показывал бы худшие координаты, чем
    # реально ушли бы на карту.
    реестр.flush()

    print()
    print(перцепция.summary())
    print(реестр.summary())
    if isinstance(детектор, ClassicDetector):
        print("[разбор] порог детектора адаптивный: %.1f разброса фона, но не меньше "
              "%.0f%% диапазона яркости" % (детектор.contrast_sigma,
                                            детектор.contrast_min * 100))

    print("\nЧто ушло бы в хаб:")
    for msg in transport.отправленные:
        data = msg["data"]
        if msg["type"] == "station_new":
            print("  station_new %s: x=%+.2f y=%+.2f %s (conf %.2f)"
                  % (data["id"], data["x"], data["y"], data["status"], data["conf"]))
        elif msg["type"] == "status_update":
            print("  status_update %s -> %s" % (data["id"], data["status"]))
    if not transport.отправленные:
        print("  (ничего)")

    print("\nИтоговый реестр:")
    for станция in реестр.snapshot():
        print("  %s: x=%+.2f y=%+.2f %s, наблюдений %d (точных %d)%s"
              % (станция["id"], станция["x"], станция["y"], станция["status"],
                 станция["n_obs"], станция["n_precise"],
                 "" if станция["precise"] else "  ТОЛЬКО ПО КРАЮ КАДРА"))
    if args.out:
        print("\n[разбор] кадров с детекциями сохранено: %d -> %s" % (сохранено, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
