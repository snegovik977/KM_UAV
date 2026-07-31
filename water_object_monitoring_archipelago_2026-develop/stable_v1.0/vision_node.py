"""
vision_node.py — единая точка входа для нейросети и всего vision-функционала.

ДВА РЕЖИМА:

1) Standalone (проверка без взлёта):
       python vision_node.py
   Берёт кадры с главной камеры, детектит суда, рисует цветные рамки и выводит
   в трансляцию + терминал. Полёта нет — удобно проверять код на земле.

2) Из кода автономного полёта:
       from vision_node import VisionNode
       node = VisionNode()                 # до взлёта
       ...
       annotated = node.process(frame)     # в каждом кадре полётного цикла
       viewer.imshow("main_camera", annotated)
       ...
       node.finish()                       # после посадки

Внутри использует нашу vision-библиотеку (code/vision) и обвязку VisionRuntime.
Размер кадра определяется автоматически по первому кадру.

РАЗВЁРТЫВАНИЕ: рядом с этим файлом положить папки vision/ и onboard/ (из code/).
"""
import os
import sys
import time

# пути к vision/onboard (репо: ../code ; на дроне: рядом со скриптом)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_NAME = "yolo11nnew"         # ТОЧНОЕ имя из реестра («Модели ИИ» -> «Название»), БЕЗ «v»
OUT_DIR = "/home/pioneermini/workspace"


class VisionNode:
    """
    Обёртка над нейросетью и всей vision-логикой. Ленивая инициализация: реальный
    VisionRuntime создаётся на первом кадре (когда известен размер).

    detector=None -> боевой YoloRknnDetector (.rknn на NPU). Для offline-тестов
    можно передать ColorBlobDetector (заглушка на цвете).
    """

    def __init__(self, detector=None, model_name=MODEL_NAME, out_dir=OUT_DIR,
                 require_aruco=True, conf_registered=0.25, conf_unregistered=0.40,
                 use_roi=True):
        self._detector = detector
        self._model_name = model_name
        self._out_dir = out_dir
        self._require_aruco = require_aruco
        self._conf_reg = conf_registered
        self._conf_unreg = conf_unregistered
        self._use_roi = use_roi               # False -> детект на КАЖДОМ кадре (наземный тест)
        self._rt = None                       # VisionRuntime, создаётся лениво
        self._failed = False                  # если init не удался — не долбить каждый кадр

    def _ensure(self, frame):
        if self._rt is not None or self._failed:
            return
        try:
            from onboard.runtime import VisionRuntime
            detector = self._detector
            if detector is None:
                from vision.detector import YoloRknnDetector
                detector = YoloRknnDetector(model_name=self._model_name,
                                            object_thresh=min(self._conf_reg, self._conf_unreg),
                                            nms_thresh=0.45)
            size = (frame.shape[1], frame.shape[0])
            self._rt = VisionRuntime(detector, out_dir=self._out_dir, frame_size=size,
                                     conf_registered=self._conf_reg,
                                     conf_unregistered=self._conf_unreg,
                                     require_aruco=self._require_aruco,
                                     use_roi=self._use_roi)
            print(f"[vision_node] инициализирована, размер кадра {size}")
        except Exception as e:
            # один раз сообщаем и больше не пытаемся (иначе SDK спамит 404 каждый кадр)
            self._failed = True
            print(f"[vision_node] НЕ удалось поднять детектор (модель '{self._model_name}'?): {e}")
            print("[vision_node] проверь имя модели в «Модели ИИ» и что она загружена. "
                  "Кадры пойдут в трансляцию БЕЗ детекции.")

    def process(self, frame, t: float = 0.0):
        """Обработать кадр -> аннотированный кадр (для viewer.imshow). None -> None."""
        if frame is None:
            return None
        self._ensure(frame)
        if self._rt is None:                  # детектор не поднялся -> отдаём кадр как есть
            return frame
        return self._rt.process(frame, t=t)

    def finish(self):
        """Финальный вывод (итоги), сохранение видео и лога."""
        if self._rt is not None:
            return self._rt.finish()
        return None


def run_standalone(duration=None, require_aruco=True, tilt_camera=True, use_roi=True):
    """Проверка без взлёта: камера -> детект -> трансляция квадратов. Ctrl+C для выхода.

    use_roi=False — детект на КАЖДОМ кадре без фильтра «есть ли бассейн». Нужно для
    наземного теста (лодка на полу/ковре), где воды в кадре нет.
    """
    from pioneer_sdk2 import Camera, ImageViewer, CameraType
    try:
        from pioneer_sdk2 import ServoCamera
    except Exception:
        ServoCamera = None

    node = VisionNode(require_aruco=require_aruco, use_roi=use_roi)
    cam = Camera(camera_type=CameraType.MAIN)
    viewer = ImageViewer()

    if tilt_camera and ServoCamera is not None:
        try:
            ServoCamera().set_angle(-80)      # камера в надир
        except Exception as e:
            print(f"[servo] не удалось наклонить камеру: {e}")

    print("[vision_node] standalone-режим (без полёта). Ctrl+C для остановки.")
    t0 = time.time()
    try:
        while duration is None or (time.time() - t0) < duration:
            frame = cam.get_cv_frame(timeout=5.0)
            if frame is None:
                continue
            annotated = node.process(frame, t=time.time() - t0)
            viewer.imshow("main_camera", annotated, fps=30)
    except KeyboardInterrupt:
        print("\n[vision_node] остановлено пользователем")
    finally:
        node.finish()
        try:
            cam.stop()
        except Exception:
            pass
        try:
            viewer.close()
        except Exception:
            pass


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Standalone-запуск нейросети (без полёта)")
    ap.add_argument("--no-roi", action="store_true",
                    help="детект на каждом кадре без фильтра бассейна (наземный тест)")
    ap.add_argument("--no-tilt", action="store_true", help="не наклонять камеру")
    ap.add_argument("--seconds", type=float, default=None, help="автостоп через N секунд")
    a = ap.parse_args()
    run_standalone(duration=a.seconds, tilt_camera=not a.no_tilt, use_roi=not a.no_roi)
