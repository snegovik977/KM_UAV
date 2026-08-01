#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Набор кадров для калибровки INT8: dataset/images/train -> rknn/calib + dataset.txt.

INT8-квантование подбирает масштабы по РЕАЛЬНЫМ кадрам, и брать их надо из нашей
съёмки, а не из чужого набора: диапазоны яркости тёмного покрытия площадки не похожи
ни на COCO, ни на что-либо ещё. Кадров нужно 100-200, больше не улучшает, но
заметно замедляет конвертацию.

Кадры берутся РАВНОМЕРНО по всему train-набору, а не первые сто подряд: сто кадров
одного пролёта дадут масштабы под одно освещение.

    python training/rknn/make_calib.py
    python training/rknn/make_calib.py --count 300
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common                                                      # noqa: E402

ЗДЕСЬ = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", default=os.path.join(common.ПАПКА_ДАТАСЕТА,
                                                         "images", "train"))
    parser.add_argument("--out", default=os.path.join(ЗДЕСЬ, "calib"))
    parser.add_argument("--count", type=int, default=200)
    аргументы = parser.parse_args()

    источник = common.файлы(аргументы.images, common.РАСШИРЕНИЯ_КАРТИНОК)
    if not источник:
        print("нет кадров в %s — сначала: python training/build_dataset.py"
              % аргументы.images)
        return 1

    if os.path.isdir(аргументы.out):
        shutil.rmtree(аргументы.out)
    os.makedirs(аргументы.out)

    шаг = max(1, len(источник) // аргументы.count)
    отобранные = источник[::шаг][:аргументы.count]
    пути = []
    for i, путь in enumerate(отобранные):
        # Имя латиницей и без пробелов: dataset.txt читает rknn-toolkit2 внутри
        # контейнера, и кириллица в путях ему только вредит.
        имя = "calib_%04d.jpg" % i
        shutil.copy2(путь, os.path.join(аргументы.out, имя))
        пути.append("./calib/%s" % имя)

    список = os.path.join(ЗДЕСЬ, "dataset.txt")
    with open(список, "wb") as f:
        f.write(("\n".join(пути) + "\n").encode("utf-8"))

    print("кадров для калибровки: %d (из %d доступных)" % (len(пути), len(источник)))
    print("список: %s" % список)
    print("Дальше: training/rknn/convert.sh <путь к best.onnx>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
