# Конвертация модели в RKNN (для дрона, SoC rk3566)

Переводит обученную YOLOv11n (`best.onnx`) в `.rknn` с INT8-квантованием для NPU борта.

## ⚠️ Платформа
`rknn-toolkit2` работает **только на x86-64 Linux**. Варианты запуска:
- **A. Docker** (в т.ч. на этом Mac через эмуляцию amd64) — см. ниже.
- **B. Родной x86-64 Linux** (Ubuntu 20.04/22.04): `pip install rknn-toolkit2 opencv-python-headless`, затем `python convert_rknn.py`.

## Содержимое папки
- `best.onnx` — экспорт модели (640×640, вход RGB 0..1, выход 1×6×8400).
- `calib/` + `dataset.txt` — ~200 кадров для калибровки INT8.
- `convert_rknn.py` — сам скрипт конвертации (target `rk3566`).
- `Dockerfile` — x86-окружение с rknn-toolkit2.

## Запуск через Docker
```bash
cd rknn_convert
docker build --platform linux/amd64 -t rknn-conv .
docker run --rm --platform linux/amd64 -v "$PWD":/work rknn-conv
```
Результат: `ships_v2_rk3566_int8.rknn` в этой папке.
> На Apple Silicon сборка/запуск идут через эмуляцию — медленно, но для одной модели ок.

## Загрузка на дрон
- Веб-UI Pioneer Code → «Модели ИИ → Загрузить модель» (архитектура `yolov11`).
- Или `ModelRegistry().upload_model(name, version, "ships_v2_rk3566_int8.rknn", arch="yolov11")`.

## После конвертации — ОБЯЗАТЕЛЬНО проверить
INT8-квантование может просадить точность (особенно слабый зелёный класс).
Сравнить детекции `.rknn` vs `.pt` на нескольких кадрах; если зелёный просел —
поднять порог-компенсацию/`imgsz` или пробовать `hybrid`-квантование.

## Примечание про точность (если зелёный сильно просядет)
Стандартный ONNX включает decode-голову (DFL+sigmoid), которую INT8 квантует неохотно.
Для лучшей точности использовать RKNN-оптимизированный экспорт: форк
`airockchip/ultralytics_yolo11` (голова отдаёт сырые тензоры до decode) — тогда
`convert_rknn.py` тот же, но `best.onnx` из форка.
