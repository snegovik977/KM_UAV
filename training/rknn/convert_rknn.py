# -*- coding: utf-8 -*-
"""ONNX -> RKNN для борта (SoC rk3576, подтверждён на нашем дроне 31.07.2026).

⚠️ Запускается ВНУТРИ контейнера (см. Dockerfile рядом): rknn-toolkit2 существует
только под x86-64 Linux. Снаружи звать training/rknn/convert.sh.

Вход  — raw-branch ONNX из training/export_onnx.py (девять сырых веток).
Выход — stations_rk3576_int8raw.rknn.

Препроцессинг задаётся здесь и обязан совпадать с тем, что делает борт:
ultralytics обучает на RGB и пикселях 0..1, отсюда mean=0, std=255. Камера борта
отдаёт BGR, поэтому в perception/detector.py стоит cvtColor — без него сеть
работает, но заметно хуже, и это легко списать на «недоучена».

Переменные окружения (их выставляет convert.sh):
    RKNN_ONNX     входной файл (по умолчанию best.onnx)
    RKNN_TARGET   чип (rk3576)
    RKNN_QUANT    1 — INT8 (по умолчанию), 0 — FP16 без квантования
    RKNN_OUT      имя результата
"""
from __future__ import annotations

import os
import sys

from rknn.api import RKNN

ONNX = os.environ.get("RKNN_ONNX", "best.onnx")
# rk3576 — не догадка: `check_sdk.py` прочитал на нашем борту
# `geoscan,pioneermini2 rockchip,rk3576` (CLAUDE.md). Чужая инструкция под rk3566
# относится к другому дрону, и модель под неверный чип не загрузится.
TARGET = os.environ.get("RKNN_TARGET", "rk3576")
QUANT = os.environ.get("RKNN_QUANT", "1") == "1"
DATASET = os.environ.get("RKNN_DATASET", "dataset.txt")
OUT = os.environ.get("RKNN_OUT",
                     "stations_%s_%s.rknn" % (TARGET, "int8raw" if QUANT else "fp16raw"))

if not os.path.exists(ONNX):
    print("нет входного файла %s" % ONNX)
    sys.exit(1)
if QUANT and not os.path.exists(DATASET):
    print("нет %s — сначала: python training/rknn/make_calib.py" % DATASET)
    sys.exit(1)

rknn = RKNN(verbose=True)
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform=TARGET,
    quantized_dtype="asymmetric_quantized-8",
    optimization_level=3,
)

print("== load_onnx: %s ==" % ONNX)
if rknn.load_onnx(model=ONNX) != 0:
    print("ОШИБКА load_onnx")
    sys.exit(1)

print("== build (квантование INT8: %s) ==" % QUANT)
if rknn.build(do_quantization=QUANT, dataset=DATASET if QUANT else None) != 0:
    print("ОШИБКА build")
    sys.exit(1)

print("== export_rknn: %s ==" % OUT)
if rknn.export_rknn(OUT) != 0:
    print("ОШИБКА export_rknn")
    sys.exit(1)

rknn.release()
print("")
print("✅ готово: %s" % OUT)
print("Залить на борт: python training/upload_model.py %s" % OUT)
print("Архитектуру при регистрации указывать custom — иначе ModelContainer")
print("постобработает выход сам, и наш декодер сырых веток не увидит.")
