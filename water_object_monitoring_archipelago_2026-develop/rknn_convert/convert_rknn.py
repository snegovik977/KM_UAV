"""
Конвертация ONNX -> RKNN (INT8) для БВС «Пионер Мини 2» (SoC rk3566).

⚠️ rknn-toolkit2 работает ТОЛЬКО на x86-64 Linux (не macOS, не ARM).
Запуск — на Linux-машине или в x86-Docker (см. README.md рядом).

    python convert_rknn.py

Вход:  best.onnx  (экспорт YOLOv11n, 640x640, вход RGB 0..1)
Выход: ships_v2_rk3566_int8.rknn
Калибровка INT8: dataset.txt (пути к ~200 кадрам в ./calib/)
"""
import sys
from rknn.api import RKNN

import os
# ONNX-вход. Для INT8 берём RAW-BRANCH экспорт (форк airockchip): decode-голова
# вынесена из графа -> INT8 квантуется чисто (плоский best.onnx давал нули).
# Переопределить: RKNN_ONNX=best_rawbranch.onnx python convert_rknn.py
ONNX = os.environ.get("RKNN_ONNX", "best.onnx")
# Целевой чип: борт, судя по рабочей модели best-rk3576, на RK3576.
# Можно переопределить: RKNN_TARGET=rk3566 python convert_rknn.py
TARGET = os.environ.get("RKNN_TARGET", "rk3576")
# INT8 (по умолч.) или FP16 без квантования (RKNN_QUANT=0) — FP16 точнее, если INT8 ломает decode-голову.
QUANT = os.environ.get("RKNN_QUANT", "1") == "1"
# tag в имени: rawbranch-onnx -> "int8raw" (штатный decode на борту), плоский -> "int8".
_raw = "raw" if "raw" in os.path.basename(ONNX).lower() else ""
OUT = os.environ.get("RKNN_OUT",
                     f"ships_v2_{TARGET}_{'int8' if QUANT else 'fp16'}{_raw}.rknn")
DATASET = "dataset.txt"

rknn = RKNN(verbose=True)

# Препроцессинг = как в ultralytics: RGB, пиксели /255 (mean=0, std=255).
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform=TARGET,
    quantized_dtype="asymmetric_quantized-8",
    optimization_level=3,
)

print("== load_onnx ==")
if rknn.load_onnx(model=ONNX) != 0:
    print("ОШИБКА load_onnx"); sys.exit(1)

print(f"== build (quant={QUANT}) ==")
if rknn.build(do_quantization=QUANT, dataset=DATASET) != 0:
    print("ОШИБКА build"); sys.exit(1)

print("== export_rknn ==")
if rknn.export_rknn(OUT) != 0:
    print("ОШИБКА export_rknn"); sys.exit(1)

rknn.release()
print(f"\n✅ Готово: {OUT}")
print("Загрузить на дрон: веб-UI «Модели ИИ → Загрузить модель» (arch=yolov11),")
print('или ModelRegistry().upload_model(name, version, filepath, arch="yolov11").')
