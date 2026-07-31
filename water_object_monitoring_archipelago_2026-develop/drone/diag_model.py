"""
Диагностика модели на борту: печатает форму сырого выхода .rknn и статистику скоров.
Помогает понять, почему детекций 0 (форма выхода / препроцессинг / качество).

Запуск на дроне (камеру направить на лодки в бассейне):
    cd ~/workspace
    python3 diag_model.py > diag.txt 2>&1
Потом прислать diag.txt.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))

import cv2
import numpy as np
from pioneer_rknn.base import ModelContainer
from pioneer_sdk2 import Camera, CameraType
try:
    from pioneer_sdk2 import ServoCamera
except Exception:
    ServoCamera = None

MODEL_NAME = "yolo11nnew"


class Raw(ModelContainer):
    """Без пост-процесса — отдаём сырые выходы как есть."""
    pass


def main():
    print("=== ДИАГНОСТИКА МОДЕЛИ ===")
    m = Raw(model_name=MODEL_NAME)
    cam = Camera(camera_type=CameraType.MAIN)
    if ServoCamera is not None:
        try:
            ServoCamera().set_angle(-80)
        except Exception as e:
            print("servo:", e)

    def class_max(o):
        """макс классовый скор для формата (1, 4+nc, 8400)."""
        if o.ndim == 3 and o.shape[1] in (5, 6, 84):
            return float(o[0, 4:, :].max())
        if o.ndim == 3 and o.shape[2] in (5, 6, 84):   # вдруг (1, 8400, 4+nc)
            return float(o[0, :, 4:].max())
        return float("nan")

    n = 0
    while n < 12:
        f = cam.get_cv_frame(timeout=5.0)
        if f is None:
            continue
        n += 1
        r640 = cv2.resize(f, (640, 640))
        inp_bgr = np.expand_dims(r640, 0)                                  # как сейчас в коде
        inp_rgb = np.expand_dims(cv2.cvtColor(r640, cv2.COLOR_BGR2RGB), 0)  # если модель ждёт RGB

        o_b = np.array(m.run([inp_bgr])[0])
        o_r = np.array(m.run([inp_rgb])[0])
        if n == 1:
            print(f"raw out shape = {o_b.shape}  (ждём (1,6,8400))")
        print(f"[{n}] frame={f.shape} | class-max: BGR={class_max(o_b):.3f}  RGB={class_max(o_r):.3f}")
    cam.stop()
    print("=== КОНЕЦ ===")


if __name__ == "__main__":
    main()
