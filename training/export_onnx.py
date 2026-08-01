#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""best.pt -> raw-branch ONNX для RKNN, с проверкой, что голова действительно сырая.

Шестой шаг конвейера (docs/YOLO_TRAINING.md §6) и место, где чаще всего теряют день.

⚠️ ОБЫЧНЫЙ ЭКСПОРТ ONNX НЕ ГОДИТСЯ. Стандартная голова YOLO отдаёт один плоский
тензор (1 x 4+nc x 8400) с decode внутри графа. После INT8-квантования такая модель
на борту даёт РОВНО НОЛЬ ДЕТЕКЦИЙ и НИ ОДНОЙ ОШИБКИ — DFL-часть не квантуется.
Проверено другой командой на этом же железе (docs/lessons_from_archipelago.md §2).

Рабочий путь — форк airockchip/ultralytics_yolo11: у него `format='rknn'` вырезает
decode из графа, и модель отдаёт девять сырых веток, по три на масштаб:

    box-DFL   [1, 64, H, W]     4 стороны x 16 бинов
    классы    [1, nc, H, W]     уже после sigmoid
    score-sum [1,  1, H, W]     служебная, для быстрой отбраковки

Разбирает их наш декодер на CPU борта — `perception/detector.py -> decode_rawbranch`.
Модель и декодер — ПАРА: raw-branch модель с плоским декодером даст мусор, плоская
модель с нашим декодером — пустоту, и оба случая проходят без ошибок.

    python training/export_onnx.py                        # runs/stations/weights/best.pt
    python training/export_onnx.py --model путь/к/best.pt
    python training/export_onnx.py --check best.onnx      # только проверить готовый файл
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common                                                      # noqa: E402

ФОРК = os.path.join(common.КОРЕНЬ_ОБУЧЕНИЯ, "vendor", "ultralytics_yolo11")
ВЕСА = os.path.join(common.КОРЕНЬ_ОБУЧЕНИЯ, "runs", "stations", "weights", "best.pt")
_REG = 16                                  # DFL-бины, как в decode_rawbranch


def проверить_onnx(путь, число_классов, log=print):
    """Тот ли это экспорт. Возвращает True, если голова сырая и каналы сходятся.

    Проверка стоит секунду, а цена ошибки — попытка на полигоне, в которой дрон
    красиво летает и не находит ничего.
    """
    try:
        import onnx
    except ImportError:
        log("⚠ нет пакета onnx — проверить структуру нечем, ставьте по "
            "docs/YOLO_TRAINING.md §1")
        return False
    модель = onnx.load(путь)
    выходы = []
    for выход in модель.graph.output:
        форма = [(d.dim_value or -1) for d in выход.type.tensor_type.shape.dim]
        выходы.append((выход.name, форма))

    log("выходов: %d" % len(выходы))
    for имя, форма in выходы:
        log("   %-24s %s" % (имя, форма))

    if len(выходы) == 1:
        форма = выходы[0][1]
        if len(форма) == 3 and (4 + число_классов) in форма:
            log("")
            log("🔴 ЭТО ПЛОСКАЯ ГОЛОВА (decode внутри графа). После INT8 на борту будет")
            log("   ноль детекций без единой ошибки. Экспортировать надо из форка")
            log("   airockchip с format='rknn' — см. шапку этого файла")
            return False

    # Ветки опознаём по числу каналов — ровно как декодер на борту.
    боксовых = классовых = служебных = 0
    for _, форма in выходы:
        каналы = форма[1] if len(форма) >= 2 else None
        if каналы == 4 * _REG:
            боксовых += 1
        elif каналы == число_классов:
            классовых += 1
        elif каналы == 1:
            служебных += 1

    log("")
    log("box-DFL веток: %d, классовых: %d, служебных (score-sum): %d"
        % (боксовых, классовых, служебных))
    if боксовых == 3 and классовых == 3:
        log("✅ raw-branch экспорт, %d класса — то, что ждёт decode_rawbranch на борту"
            % число_классов)
        return True
    if классовых == 0:
        log("🔴 нет ни одной ветки с %d каналами. Либо модель обучена на другое число"
            % число_классов)
        log("   классов, либо classes.txt разошёлся с обучением")
    else:
        log("🔴 ожидалось по три ветки каждого вида (три масштаба). Экспорт не тот")
    return False


def экспортировать(веса, форк, imgsz, log=print):
    """Запускает exporter.py форка. Правка cfg временная и откатывается всегда.

    Форк читает путь к модели ТОЛЬКО из своего ultralytics/cfg/default.yaml
    (его exporter.py при запуске как скрипт не разбирает argv), поэтому файл
    правится на время запуска и возвращается как был — иначе следующий экспорт
    молча возьмёт прошлую модель.
    """
    cfg = os.path.join(форк, "ultralytics", "cfg", "default.yaml")
    if not os.path.exists(cfg):
        log("нет форка в %s. Клонировать:" % форк)
        log("  git clone --depth 1 https://github.com/airockchip/ultralytics_yolo11.git %s"
            % форк)
        return None
    копия = cfg + ".исходный"
    shutil.copy2(cfg, копия)
    try:
        with open(cfg, "rb") as f:
            строки = f.read().decode("utf-8").splitlines()
        новые = []
        for строка in строки:
            if строка.startswith("model:"):
                новые.append("model: %s" % os.path.abspath(веса))
            elif строка.startswith("format:"):
                новые.append("format: rknn  # сырые ветки, decode на CPU борта")
            elif строка.startswith("imgsz:"):
                новые.append("imgsz: %d" % imgsz)
            else:
                новые.append(строка)
        with open(cfg, "wb") as f:
            f.write(("\n".join(новые) + "\n").encode("utf-8"))

        окружение = dict(os.environ)
        # PYTHONPATH на форк: в venv стоит ОБЫЧНЫЙ ultralytics, и без этого
        # экспорт пойдёт им — то есть даст ту самую плоскую голову.
        окружение["PYTHONPATH"] = форк
        код = subprocess.call([sys.executable, os.path.join("ultralytics", "engine",
                                                            "exporter.py")],
                              cwd=форк, env=окружение)
        if код != 0:
            log("exporter.py вернул %d" % код)
            return None
    finally:
        shutil.move(копия, cfg)

    onnx_путь = os.path.splitext(веса)[0] + ".onnx"
    return onnx_путь if os.path.exists(onnx_путь) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=ВЕСА)
    parser.add_argument("--fork", default=ФОРК)
    parser.add_argument("--imgsz", type=int, default=640,
                        help="должен совпадать с detector.img_size на борту")
    parser.add_argument("--check", default=None,
                        help="только проверить готовый .onnx, ничего не экспортируя")
    аргументы = parser.parse_args()

    число_классов = len(common.классы())

    if аргументы.check:
        return 0 if проверить_onnx(аргументы.check, число_классов) else 1

    if not os.path.exists(аргументы.model):
        print("нет весов %s — сначала: python training/train.py" % аргументы.model)
        return 1

    путь = экспортировать(аргументы.model, аргументы.fork, аргументы.imgsz)
    if путь is None:
        return 1
    print("")
    print("ONNX: %s" % путь)
    print("")
    if not проверить_onnx(путь, число_классов):
        return 1
    print("")
    print("Дальше: python training/rknn/make_calib.py && training/rknn/convert.sh %s" % путь)
    return 0


if __name__ == "__main__":
    sys.exit(main())
