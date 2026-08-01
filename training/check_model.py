#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Проверка обученной модели глазами и по метрикам — до конвертации в .rknn.

Пятый шаг конвейера (docs/YOLO_TRAINING.md §5.3). Смысл: mAP сам по себе ничего не
говорит о том, что будет на площадке. Смотреть надо на две вещи, и обе видны только
глазами на картинках:

  1. что модель считает станцией ЛИШНЕГО — белый логотип на покрытии, серые комки
     мусора, посадочный знак «H». Это подтверждённые ложные срабатывания
     классического детектора (CLAUDE.md), и сеть наследует их, если в датасете
     не было негативов;
  2. путает ли она ok и dust. Класс кодируется цветными телами на панели, различие
     тонкое, и первым же ломается при INT8-квантовании — поэтому эталон снимаем
     здесь, ДО конвертации, чтобы потом было с чем сравнивать.

    python training/check_model.py                          # val-набор датасета
    python training/check_model.py --images "фото станции/*/*.jpg"
    python training/check_model.py --conf 0.15 --out разбор_модели/
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common                                                      # noqa: E402

ВЕСА = os.path.join(common.КОРЕНЬ_ОБУЧЕНИЯ, "runs", "stations", "weights", "best.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=ВЕСА)
    parser.add_argument("--images", default=None,
                        help="маска картинок; по умолчанию val-набор датасета")
    parser.add_argument("--out", default=os.path.join(common.КОРЕНЬ_ОБУЧЕНИЯ, "предсказания"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--no-metrics", action="store_true",
                        help="не считать mAP (нужен собранный датасет)")
    аргументы = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("нужен ultralytics: см. docs/YOLO_TRAINING.md §1")
        return 1
    if not os.path.exists(аргументы.model):
        print("нет весов %s — сначала обучение: python training/train.py" % аргументы.model)
        return 1

    имена = common.классы()
    модель = YOLO(аргументы.model)

    data = os.path.join(common.ПАПКА_ДАТАСЕТА, "data.yaml")
    if not аргументы.no_metrics and os.path.exists(data):
        метрики = модель.val(data=data, split="val", verbose=False)
        print("mAP50 = %.3f | mAP50-95 = %.3f"
              % (метрики.box.map50, метрики.box.map))
        for i, имя in enumerate(имена):
            try:
                print("  %-6s AP50 = %.3f" % (имя, метрики.box.ap50[i]))
            except (IndexError, TypeError):
                print("  %-6s в val не встретился — метрики по нему нет" % имя)
        print("⚠ mAP считается на val-клипах. Если сплит был не по клипам, число врёт")

    картинки = (sorted(glob.glob(аргументы.images)) if аргументы.images
                else common.файлы(os.path.join(common.ПАПКА_ДАТАСЕТА, "images", "val"),
                                  common.РАСШИРЕНИЯ_КАРТИНОК))
    if not картинки:
        print("нечего смотреть: ни --images, ни val-набора")
        return 1
    if not os.path.isdir(аргументы.out):
        os.makedirs(аргументы.out)

    счёт = [0] * len(имена)
    for путь in картинки:
        кадр = common.imread(путь)
        if кадр is None:
            continue
        итог = модель.predict(кадр, conf=аргументы.conf, verbose=False)[0]
        for бокс in итог.boxes:
            класс = int(бокс.cls.item())
            if класс < len(счёт):
                счёт[класс] += 1
        common.imwrite(os.path.join(аргументы.out, os.path.basename(путь)), итог.plot())

    print("")
    print("картинок: %d, найдено: %s"
          % (len(картинки), ", ".join("%s=%d" % (и, с) for и, с in zip(имена, счёт))))
    print("смотреть: %s" % аргументы.out)
    print("Дальше: python training/export_onnx.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
