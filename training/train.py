#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Обучение YOLO11n на станциях (два класса: ok, dust).

Четвёртый шаг конвейера (docs/YOLO_TRAINING.md §5).

    python training/train.py                      # 100 эпох, 640, автоустройство
    python training/train.py --epochs 200 --batch 8
    python training/train.py --resume             # продолжить прерванное обучение

Почему YOLO11n: `pioneer_rknn` умеет yolov8 и yolov11, а «n» — самая мелкая, и это
не экономия, а требование. Инференс идёт на борту в общем цикле с полётом, детекция
крутится раз в `detector.every_n` кадров (config.yaml), и модель побольше просто
не успеет.

Аугментации подобраны под НАШУ задачу, а не по умолчанию, и два отличия принципиальны:

  hsv_h = 0.0   Оттенок НЕ трогаем. «Покрыта пылью» на площадке изображается
                цветными многогранниками, положенными на панель (разбор фотографий
                организаторов, CLAUDE.md), то есть класс кодируется ЦВЕТОМ. Jitter
                по оттенку травит ровно тот признак, по которому различаются классы.
  hsv_v = 0.4   Яркость крутим наоборот смело: освещение на площадке меняется от
                попытки к попытке, а класс от него не зависит.

  degrees=180, flipud=0.5   Съёмка сверху: станция может лежать под любым углом,
                и «вверх ногами» для неё — такой же валидный кадр.
  scale=0.5     Разные высоты полёта (1.2-2.5 м) меняют размер станции в кадре вдвое.

imgsz=640 обязан совпадать со входом модели на борту (detector.img_size в config.yaml)
и со входом .rknn при конвертации. Меняете здесь — меняйте в обоих местах.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common                                                      # noqa: E402

ВЕСА_ПО_УМОЛЧАНИЮ = os.path.join(common.КОРЕНЬ_ОБУЧЕНИЯ, "weights", "yolo11n.pt")


def устройство():
    """cuda -> mps -> cpu. Отдельной функцией: на площадке машина чужая, и знать,
    на чём считалось, нужно из лога."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=os.path.join(common.ПАПКА_ДАТАСЕТА, "data.yaml"))
    parser.add_argument("--model", default=ВЕСА_ПО_УМОЛЧАНИЮ,
                        help="стартовые веса (по умолчанию training/weights/yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--name", default="stations")
    parser.add_argument("--device", default=None, help="cpu | mps | 0 (по умолчанию авто)")
    parser.add_argument("--resume", action="store_true")
    аргументы = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("нужен ultralytics: см. docs/YOLO_TRAINING.md §1 (окружение training/.venv)")
        return 1

    if not os.path.exists(аргументы.data):
        print("нет %s — сначала: python training/build_dataset.py" % аргументы.data)
        return 1

    имена = common.классы()
    dev = аргументы.device or устройство()
    print("устройство: %s | классы: %s" % (dev, ", ".join(имена)))
    if dev == "cpu":
        print("⚠ считается на CPU: 100 эпох на 300 кадрах — это часы, а не минуты. "
              "Если есть машина с видеокартой, учить лучше на ней")

    модель = YOLO(аргументы.model)
    модель.train(
        data=аргументы.data,
        epochs=аргументы.epochs,
        imgsz=аргументы.imgsz,           # = detector.img_size на борту и входу .rknn
        batch=аргументы.batch,
        device=dev,
        patience=аргументы.patience,
        project=os.path.join(common.КОРЕНЬ_ОБУЧЕНИЯ, "runs"),
        name=аргументы.name,
        resume=аргументы.resume,
        # --- аугментации, обоснование — в шапке файла ---
        hsv_h=0.0,                       # НЕ крутить оттенок: цвет = класс
        hsv_s=0.4,
        hsv_v=0.4,                       # освещение площадки гуляет, класс от него нет
        degrees=180.0,                   # вид сверху: любая ориентация станции валидна
        flipud=0.5,
        fliplr=0.5,
        translate=0.1,
        scale=0.5,                       # высоты полёта 1.2-2.5 м
        mosaic=1.0,
        close_mosaic=10,                 # последние эпохи без мозаики: она искажает
                                         # масштаб, а у нас порог площади в метрах
    )

    итог = os.path.join(common.КОРЕНЬ_ОБУЧЕНИЯ, "runs", аргументы.name, "weights", "best.pt")
    print("")
    print("лучшие веса: %s" % итог)
    print("Дальше: python training/check_model.py  (посмотреть глазами), затем")
    print("        python training/export_onnx.py  (raw-branch ONNX для .rknn)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
