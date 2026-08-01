#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Кадры для разметки: видео попыток и снимки -> training/images_to_label/<клип>/.

Первый шаг конвейера обучения (docs/YOLO_TRAINING.md §2). Берёт записи полётов
(`records/flight_*.mp4`), видео с `tools/handheld.py --record` и любые снимки,
раскладывает их по КЛИПАМ — одна папка на одну съёмку.

Клип, а не сваленная куча кадров, нужен ради сплита train/val: соседние кадры одного
пролёта почти идентичны, и при случайном разбиении они протекают в val, после чего
mAP показывает 0.95 на модели, которая на площадке не видит ничего
(docs/lessons_from_archipelago.md §4). Разбиение по клипам делает build_dataset.py.

Два фильтра, без которых из 600 кадров пролёта выйдет 600 почти одинаковых картинок:

    --stride  брать каждый N-й кадр (грубо, по времени);
    --diff    и выбрасывать те, что слишком похожи на предыдущий СОХРАНЁННЫЙ кадр
              (тонко, по картинке: дрон висел на месте — кадры не копятся).

⚠️ ЗАПИСИ ПОЛЁТА СОДЕРЖАТ HUD И РАМКИ ДЕТЕКТОРА. `квадрокоптер/main.py` до правки
от 2026-08-01 писал в видео уже аннотированный кадр: чёрная полоса с телеметрией
сверху и зелёные рамки вокруг найденного. Обучать на таком нельзя — сеть выучит
рамку, которую сама же и нарисовала. Полоса срезается ключом `--crop-top`, а рамки
не срезаются ничем, поэтому старые записи годятся только как фон/негативы. Новые
записи пишутся сырыми (`mission.record_raw: true`) — им ничего срезать не нужно.

Примеры:

    # все записи попыток, каждый 5-й кадр, без почти одинаковых
    python training/extract_frames.py records/*.mp4

    # старая запись с HUD: срезать верхнюю полосу
    python training/extract_frames.py records/flight_20260801_083847.mp4 --crop-top 24

    # снимки организаторов одной папкой (клип назовётся organizers)
    python training/extract_frames.py "фото станции" --clip organizers

    # ограничить выход, чтобы разметка была подъёмной за вечер
    python training/extract_frames.py records/*.mp4 --max 120
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common                                                      # noqa: E402

try:
    import cv2
except ImportError:                       # pragma: no cover
    cv2 = None

try:
    import numpy as np
except ImportError:                       # pragma: no cover
    np = None


def отпечаток(кадр):
    """Маленькая серая копия кадра — по ней считается «похоже на предыдущий»."""
    серое = cv2.cvtColor(кадр, cv2.COLOR_BGR2GRAY)
    return cv2.resize(серое, (64, 64)).astype("float32")


def различие(a, b):
    """Средний модуль разницы двух отпечатков, в уровнях яркости 0..255."""
    return float(np.abs(a - b).mean())


class Сборщик(object):
    """Складывает кадры одного клипа, отбрасывая похожие на уже сохранённые."""

    def __init__(self, папка, клип, порог_различия, crop_top, лимит, начальный_номер=0):
        self.папка = папка
        self.клип = клип
        self.порог = float(порог_различия)
        self.crop_top = int(crop_top)
        self.лимит = лимит
        self.номер = начальный_номер
        self.прошлый = None
        self.сохранено = 0
        self.похожих = 0

    def добавить(self, кадр):
        if кадр is None:
            return False
        if self.лимит is not None and self.сохранено >= self.лимит:
            return False
        if self.crop_top:
            кадр = кадр[self.crop_top:, :]
        отп = отпечаток(кадр)
        if self.прошлый is not None and различие(отп, self.прошлый) < self.порог:
            self.похожих += 1
            return False
        имя = os.path.join(self.папка, "%s_%05d.jpg" % (self.клип, self.номер))
        if not common.imwrite(имя, кадр):
            print("  не записался кадр %s" % имя)
            return False
        self.прошлый = отп
        self.номер += 1
        self.сохранено += 1
        return True


def имя_клипа(источник, задано):
    if задано:
        return задано
    основа = os.path.basename(os.path.normpath(источник))
    if os.path.isfile(источник):
        основа = os.path.splitext(основа)[0]
    # flight_20260801_083847 -> 20260801_083847: префикс одинаков у всех записей
    # и в имени клипа только мешает читать статистику.
    if основа.startswith("flight_"):
        основа = основа[len("flight_"):]
    return основа.replace(" ", "_")


def из_видео(путь, сборщик, stride):
    поток = cv2.VideoCapture(путь)
    if not поток.isOpened():
        print("  видео не открылось: %s" % путь)
        return
    всего = 0
    while True:
        есть, кадр = поток.read()
        if not есть:
            break
        if всего % stride == 0:
            сборщик.добавить(кадр)
        всего += 1
    поток.release()
    print("  прочитано кадров: %d" % всего)


def из_картинок(пути, сборщик):
    for путь in пути:
        кадр = common.imread(путь)
        if кадр is None:
            print("  не прочиталась: %s" % путь)
            continue
        сборщик.добавить(кадр)


def развернуть(шаблоны):
    """Маски и папки -> список источников. Папка со снимками = один источник."""
    источники = []
    for шаблон in шаблоны:
        совпадения = sorted(glob.glob(шаблон)) or ([шаблон] if os.path.exists(шаблон) else [])
        if not совпадения:
            print("ничего не нашлось по %r" % шаблон)
        источники.extend(совпадения)
    return источники


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("источники", nargs="+",
                        help="видео, папки со снимками или маски (records/*.mp4)")
    parser.add_argument("--out", default=common.ПАПКА_РАЗМЕТКИ,
                        help="куда складывать (по умолчанию training/images_to_label)")
    parser.add_argument("--clip", default=None,
                        help="имя клипа; по умолчанию — имя файла или папки")
    parser.add_argument("--stride", type=int, default=5,
                        help="брать каждый N-й кадр видео (по умолчанию 5)")
    parser.add_argument("--diff", type=float, default=4.0,
                        help="порог непохожести на предыдущий сохранённый кадр, "
                             "уровни яркости 0..255 (по умолчанию 4.0; 0 — не фильтровать)")
    parser.add_argument("--crop-top", type=int, default=0,
                        help="срезать сверху N пикселей: полоса HUD в старых записях")
    parser.add_argument("--max", type=int, default=None,
                        help="не больше N кадров с одного источника")
    аргументы = parser.parse_args()

    if cv2 is None or np is None:
        print("нужны opencv-python и numpy: pip install -r training/requirements.txt")
        return 1

    источники = развернуть(аргументы.источники)
    if not источники:
        return 1

    # Папки с картинками группируются в один клип, видео — каждое в свой.
    всего = 0
    for источник in источники:
        клип = имя_клипа(источник, аргументы.clip)
        папка = os.path.join(аргументы.out, клип)
        if not os.path.isdir(папка):
            os.makedirs(папка)
        # Продолжаем нумерацию, если в папку уже клали кадры: повторный запуск
        # с новым видео не должен затирать разложенное и размеченное.
        уже = len(common.файлы(папка, common.РАСШИРЕНИЯ_КАРТИНОК))
        сборщик = Сборщик(папка, клип, аргументы.diff, аргументы.crop_top,
                          аргументы.max, начальный_номер=уже)
        print("%s -> %s/ (уже было %d)" % (источник, папка, уже))

        расширение = os.path.splitext(источник)[1].lower()
        if os.path.isdir(источник):
            картинки = common.файлы(источник, common.РАСШИРЕНИЯ_КАРТИНОК)
            # Снимки часто лежат подпапками (`фото станции/чистые`, `грязные`).
            for под in sorted(os.listdir(источник)):
                подпуть = os.path.join(источник, под)
                if os.path.isdir(подпуть):
                    картинки.extend(common.файлы(подпуть, common.РАСШИРЕНИЯ_КАРТИНОК))
            из_картинок(картинки, сборщик)
        elif расширение in common.РАСШИРЕНИЯ_ВИДЕО:
            из_видео(источник, сборщик, max(1, аргументы.stride))
        elif расширение in common.РАСШИРЕНИЯ_КАРТИНОК:
            из_картинок([источник], сборщик)
        else:
            print("  не знаю, что это за файл, пропускаю")
            continue

        print("  сохранено %d, отброшено похожих %d"
              % (сборщик.сохранено, сборщик.похожих))
        всего += сборщик.сохранено

    print("\nВсего кадров к разметке: %d" % всего)
    print("Дальше: python training/label.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
