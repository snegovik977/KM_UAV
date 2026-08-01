#!/bin/sh
# Конвертация ONNX -> RKNN в контейнере: снаружи rknn-toolkit2 не работает
# (только x86-64 Linux). Обёртка над docker, чтобы не помнить ключи.
#
#   training/rknn/convert.sh training/runs/stations/weights/best.onnx
#   RKNN_QUANT=0 training/rknn/convert.sh best.onnx     # FP16, если INT8 просадил классы
#
# Результат кладётся рядом со скриптом: stations_rk3576_int8raw.rknn
set -e

ЗДЕСЬ=$(cd "$(dirname "$0")" && pwd)
ВХОД=${1:-}
if [ -z "$ВХОД" ]; then
    echo "укажите .onnx: training/rknn/convert.sh путь/к/best.onnx"
    exit 1
fi
if [ ! -f "$ВХОД" ]; then
    echo "нет файла $ВХОД"
    exit 1
fi

# ONNX кладём в рабочую папку контейнера под фиксированным именем: в контейнер
# монтируется только training/rknn.
cp "$ВХОД" "$ЗДЕСЬ/best.onnx"

if [ ! -f "$ЗДЕСЬ/dataset.txt" ] && [ "${RKNN_QUANT:-1}" = "1" ]; then
    echo "нет dataset.txt — сначала: python training/rknn/make_calib.py"
    exit 1
fi

if ! docker image inspect rknn-conv >/dev/null 2>&1; then
    echo "== собираю образ rknn-conv (первый раз это долго: эмуляция x86) =="
    docker build --platform linux/amd64 -t rknn-conv "$ЗДЕСЬ"
fi

echo "== конвертация =="
docker run --rm --platform linux/amd64 \
    -e RKNN_ONNX=best.onnx \
    -e RKNN_TARGET="${RKNN_TARGET:-rk3576}" \
    -e RKNN_QUANT="${RKNN_QUANT:-1}" \
    -v "$ЗДЕСЬ":/work rknn-conv

ls -lh "$ЗДЕСЬ"/*.rknn
