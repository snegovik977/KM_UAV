"""
Дообучение YOLOv11n на корабликах (2 класса: registered / unregistered).

Модель: YOLOv11n (рекомендация ML-инженера команды; pioneer_rknn поддерживает
yolov8 и yolov11). Формат разметки YOLO — общий для v8/v11, менять не нужно.
При экспорте в .rknn arch = "yolov11".

Запуск (из папки training/, окружение .venv активировано):
    cd training
    python train.py

Результат: runs/ships/weights/best.pt — его потом конвертируем в .rknn для дрона.
"""
from ultralytics import YOLO
import torch

# На Apple Silicon используем GPU (MPS), иначе CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Устройство обучения: {device}")

# Предобученная на COCO YOLOv11n — скачается автоматически при первом запуске
model = YOLO("yolo11n.pt")

model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,          # ДОЛЖЕН совпадать с входом .rknn на борту (pioneer_rknn.Yolo).
                        # Если борт умеет больше (960/1280) — поднять И тут, И при экспорте:
                        # это самый сильный рычаг для мелких/зелёных лодок. Уточнить у ML.
    batch=16,           # если MPS ругнётся на память — снизить до 8/4
    device=device,
    patience=30,        # больше данных (клип 2) -> даём дообучиться подольше
    project="runs",
    name="ships_v2",    # новый прогон (старый ships не затрётся)

    # --- Аугментации ---
    hsv_h=0.0,          # ⚠️ КРИТИЧНО: НЕ крутить оттенок — цвет = класс (оранж/зелёный)!
    hsv_s=0.4,          # насыщенность/яркость можно
    hsv_v=0.4,
    fliplr=0.5,         # горизонтальное отражение
    flipud=0.5,         # вид сверху -> вертикальное отражение тоже валидно
    degrees=180.0,      # лодка может быть в любой ориентации (надир)
    translate=0.1,
    scale=0.5,          # разные высоты полёта
)

print("Обучение завершено. Лучшая модель: runs/**/ships_v2/weights/best.pt")
