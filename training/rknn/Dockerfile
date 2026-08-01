# x86-64 Linux окружение для rknn-toolkit2: на нашем Mac (arm64) идёт через эмуляцию.
#
# ⚠️ rknn-toolkit2 существует ТОЛЬКО под x86-64 Linux. Ни macOS, ни ARM, ни Windows —
# поэтому конвертация .onnx -> .rknn делается в контейнере, а не «просто pip install».
#
# Сборка и запуск — training/rknn/convert.sh либо руками:
#   docker build --platform linux/amd64 -t rknn-conv training/rknn
#   docker run --rm --platform linux/amd64 -v "$PWD/training/rknn":/work rknn-conv
FROM --platform=linux/amd64 python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# --timeout/--retries не украшение: rknn-toolkit2 весит сотни мегабайт, и на слабой
# сети (а на площадке она такая) обычный pip обрывается на середине по Read timeout.
RUN pip install --no-cache-dir --timeout 180 --retries 10 \
        rknn-toolkit2 opencv-python-headless

# onnx откатывается на 1.14.1 ОТДЕЛЬНЫМ слоём, и это не суеверие: rknn-toolkit2 тянет
# onnx>=1.16, где убрали onnx.mapping, а load_onnx зовёт его при перестройке графа —
# ровно на raw-branch модели, которая нам и нужна. Проверено другой командой на этом
# же железе (docs/lessons_from_archipelago.md §2). --force-reinstall --no-deps
# перебивает подтянутую версию, не трогая остальное.
RUN pip install --no-cache-dir --timeout 180 --retries 10 \
        --force-reinstall --no-deps "onnx==1.14.1"

WORKDIR /work
CMD ["python", "convert_rknn.py"]
