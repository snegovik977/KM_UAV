#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Диагностика модели НА БОРТУ: отвечает на «детекций ноль, а почему?».

Самый дорогой способ потерять время на площадке — это ноль детекций БЕЗ ЕДИНОЙ ОШИБКИ.
Он бывает от трёх разных причин, и вслепую они неразличимы
(docs/lessons_from_archipelago.md §2):

  1. МОДЕЛЬ. Стандартный ONNX с плоской головой (1x6x8400, decode внутри графа) после
     INT8 отдаёт нули: DFL-голова не квантуется. Признак — форма выхода плоская,
     а не девять сырых веток.
  2. АРХИТЕКТУРА В РЕЕСТРЕ. Модель зарегистрирована как yolov8/yolov11, и ModelContainer
     постобработал выход сам. Признак — выходов не девять, а один-два и уже разобранных.
  3. ПРЕПРОЦЕССИНГ. Камера отдаёт BGR, модель обучена на RGB. Признак — максимальный
     классовый скор при RGB заметно выше, чем при BGR.

Скрипт печатает форму каждого сырого выхода и сравнивает скоры при подаче BGR и RGB.
Переносить обученную модель ради этого не нужно: всё, что он делает, — прогоняет
несколько кадров с камеры.

Запуск на борту (камеру навести на разложенные станции):
    cd ~/workspace
    python3 tools/diag_model.py --model stations > diag.txt 2>&1
"""
from __future__ import annotations

import argparse
import os
import sys

_ЗДЕСЬ = os.path.dirname(os.path.abspath(__file__))
for _путь in (_ЗДЕСЬ, os.path.dirname(_ЗДЕСЬ),
              os.path.join(os.path.dirname(_ЗДЕСЬ), "квадрокоптер")):
    if os.path.isdir(_путь) and _путь not in sys.path:
        sys.path.insert(0, _путь)

_REG = 16          # DFL-бины: box-ветка это 4 стороны * 16 бинов = 64 канала


def опознать_ветку(форма, nc):
    """Что это за выход, по числу каналов. Именно так его опознаёт наш декодер:
    RKNN выходы переставляет, и полагаться на порядок нельзя."""
    каналы = [d for d in форма if d not in (1, 0)]
    for канал in форма:
        if канал == 4 * _REG:
            return "box-DFL (64 канала)"
        if канал == nc:
            return "классы (%d канала)" % nc
    if len(форма) >= 3 and форма[-1] in (4 + nc, 6, 84):
        return "ПЛОСКАЯ ГОЛОВА — decode внутри графа, после INT8 будет ноль детекций"
    if len(форма) >= 2 and форма[1] in (4 + nc, 6, 84):
        return "ПЛОСКАЯ ГОЛОВА — decode внутри графа, после INT8 будет ноль детекций"
    return "не опознана (каналы %s)" % (каналы,)


def максимальный_скор(выходы, nc):
    """Наибольший классовый скор среди всех классовых веток. Ветки после sigmoid,
    поэтому число сразу читается как уверенность."""
    лучшее = None
    for выход in выходы:
        массив = _как_массив(выход)
        if массив is None:
            continue
        if nc in массив.shape:
            значение = float(массив.max())
            лучшее = значение if лучшее is None else max(лучшее, значение)
    return лучшее


def _как_массив(тензор):
    import numpy as np

    try:
        return np.array(тензор)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Диагностика .rknn на борту")
    parser.add_argument("--model", default="stations", help="имя модели в реестре борта")
    parser.add_argument("--classes", type=int, default=3,
                        help="сколько классов у модели (у нас 3: ok, dust, broken)")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=640, dest="img_size")
    parser.add_argument("--servo", type=int, default=-80,
                        help="угол камеры; тот же, что в config.yaml")
    args = parser.parse_args()

    import cv2
    import numpy as np

    print("=== ДИАГНОСТИКА МОДЕЛИ %r ===" % args.model)

    try:
        from pioneer_rknn.base import ModelContainer
    except ImportError as e:
        print("pioneer_rknn недоступен (%s). Скрипт имеет смысл только НА БОРТУ." % e)
        return 1

    class Сырая(ModelContainer):
        """Без пост-процесса: отдаём выходы как есть.

        Если здесь придёт что-то уже разобранное, значит модель зарегистрирована
        не как custom, и ModelContainer постобработал её сам.
        """

    модель = Сырая(model_name=args.model)

    from pioneer_sdk2 import Camera, CameraType
    try:
        from pioneer_sdk2 import ServoCamera
        ServoCamera().set_angle(args.servo)
        print("сервопривод: %d" % args.servo)
    except Exception as e:
        print("сервопривод не отработал: %s: %s" % (type(e).__name__, e))

    камера = Camera(camera_type=CameraType.MAIN)
    снято = 0
    try:
        while снято < args.frames:
            кадр = камера.get_cv_frame(timeout=5.0)
            if кадр is None:
                continue
            снято += 1

            уменьшенный = cv2.resize(кадр, (args.img_size, args.img_size))
            как_bgr = np.expand_dims(уменьшенный, 0)
            как_rgb = np.expand_dims(cv2.cvtColor(уменьшенный, cv2.COLOR_BGR2RGB), 0)

            выходы_bgr = модель.run([как_bgr])
            выходы_rgb = модель.run([как_rgb])

            if снято == 1:
                выходы = list(выходы_bgr) if выходы_bgr is not None else []
                print("кадр с камеры: %s, %s" % (кадр.shape, кадр.dtype))
                print("выходов модели: %d (ждём 9 сырых веток)" % len(выходы))
                for номер, выход in enumerate(выходы):
                    массив = _как_массив(выход)
                    форма = tuple(массив.shape) if массив is not None else "?"
                    print("  [%d] %s — %s"
                          % (номер, форма,
                             опознать_ветку(форма, args.classes)
                             if массив is not None else "не массив"))
                if len(выходы) < 9:
                    print("  ВНИМАНИЕ: веток меньше девяти. Либо модель экспортирована "
                          "с плоской головой, либо зарегистрирована не как custom — "
                          "см. docs/lessons_from_archipelago.md §2")

            скор_bgr = максимальный_скор(выходы_bgr or [], args.classes)
            скор_rgb = максимальный_скор(выходы_rgb or [], args.classes)
            print("[%d] макс. скор:  BGR=%s  RGB=%s"
                  % (снято,
                     "?" if скор_bgr is None else "%.3f" % скор_bgr,
                     "?" if скор_rgb is None else "%.3f" % скор_rgb))
    finally:
        # Камера отдаётся через shared memory: не остановить — следующий запуск
        # получит TimeoutError.
        камера.stop()

    print("=== КОНЕЦ ===")
    print("Как читать: если RGB заметно выше BGR — в детекторе забыт cvtColor. "
          "Если веток не девять — виновата модель или её архитектура в реестре. "
          "Если скоры около нуля при обоих вариантах — модель недоучена или "
          "сломана квантованием.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
