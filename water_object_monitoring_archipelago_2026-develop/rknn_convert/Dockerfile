# x86-64 Linux окружение для rknn-toolkit2 (на Mac/ARM запускается через эмуляцию).
FROM --platform=linux/amd64 python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# rknn-toolkit2 + зависимости.
RUN pip install --no-cache-dir rknn-toolkit2 opencv-python-headless
# onnx ЖЁСТКО откатываем на 1.14.1 ОТДЕЛЬНЫМ слоём: rknn-toolkit2 тянет onnx>=1.16,
# где убрали onnx.mapping -> load_onnx падает на моделях, требующих rebuild графа
# (raw-branch экспорт). --force-reinstall --no-deps перебивает подтянутую версию.
RUN pip install --no-cache-dir --force-reinstall --no-deps "onnx==1.14.1"

WORKDIR /work
CMD ["python", "convert_rknn.py"]
