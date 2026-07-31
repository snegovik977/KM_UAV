"""
Автономная миссия + нейросеть (интеграция vision-обвязки в код полёта коллег).

Базируется на drone/autonomous_mission. Добавлено (наши изменения):
  - VisionRuntime: детекция (YOLO .rknn) + классификация + ArUco-id + подсчёт
    уникальных + цветные рамки + вывод в терминал (форматы оргов) + запись видео +
    фотофиксация + лог. Работает В ПОТОКЕ ВИДЕО (обрабатывает кадр перед показом).
  - Наклон камеры в надир (set_angle(-80)) — нужен для съёмки бассейна сверху.
  - Высота полёта приведена к регламенту 26.07.2026: FLIGHT_Z = 1.5 м (макс. 2 м!).

РАЗВЁРТЫВАНИЕ НА ДРОНЕ: положить рядом папки `vision/` и `onboard/` (из code/),
либо оставить репозиторий и путь ниже подхватит их автоматически.

⚠️ Проверить на дроне: сигнатуру Yolo.run (см. vision/detector.py), реальный размер
кадра, что модель загружена в «Модели ИИ» (arch=yolov11).
"""
import sys
import os
import time
import threading
import cv2
import numpy as np

# --- путь к нашим vision/onboard пакетам (репо: ../code ; на дроне: рядом со скриптом) ---
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Импорт SDK
try:
    from pioneer_sdk2 import Pioneer, Camera, ImageViewer, CameraType, ServoCamera
except ImportError:
    print("Ошибка: Не удалось импортировать pioneer_sdk2. Убедитесь, что библиотека установлена.")
    sys.exit(1)

# Наша vision-нода (та же, что запускается standalone). Не критична для полёта:
# при сбое импорта миссия продолжится без нейросети.
try:
    from vision_node import VisionNode
    VISION_AVAILABLE = True
except Exception as e:
    print(f"[vision] предупреждение: не удалось импортировать vision_node: {e}")
    VISION_AVAILABLE = False

# ------- параметры -------
wait_time = 30
# Высота полёта: регламент 26.07.2026 — реком. 1.5 м, макс. 2 м, мин. 0.5 м.
# (У коллег было 2.55 м — это ВЫШЕ нового потолка 2 м, приведено к 1.5 м.)
FLIGHT_Z = 1.5

# Глобальные флаги для управления потоками
stop_video_thread = False


def video_worker(cam_main, cam_opt, viewer, vr):
    """Поток видео: обработка главного кадра нейросетью -> показ аннотированного."""
    global stop_video_thread
    t0 = time.time()
    while not stop_video_thread:
        try:
            if cam_main and viewer:
                frame_main = cam_main.get_cv_frame()
                if frame_main is not None:
                    if vr is not None:
                        # детекция+классификация+ArUco+оверлей+запись+лог
                        frame_main = vr.process(frame_main, t=time.time() - t0)
                    viewer.imshow(name='main_camera', frame=frame_main)

            if cam_opt and viewer:
                frame_opt = cam_opt.get_cv_frame()
                if frame_opt is not None:
                    viewer.imshow(name='additional_camera', frame=frame_opt)

            time.sleep(0.03)
        except Exception:
            pass


def hover_with_countdown(pioneer, total_wait_time, label=""):
    """Зависание над точкой с обратным отсчётом в терминале."""
    print(f"{label}Ожидание {total_wait_time} секунд...")
    start_time = time.time()
    end_time = start_time + total_wait_time
    while time.time() < end_time:
        remaining = max(0, int(end_time - time.time()))
        sys.stdout.write(f"\r{label}Осталось времени до конца sleep: {remaining} сек   ")
        sys.stdout.flush()
        time.sleep(1)
    print(f"\n{label}Время ожидания истекло.")


def main():
    global stop_video_thread

    pioneer = None
    iv = None
    cam_main_obj = None
    cam_opt_obj = None
    video_thread = None
    vr = None

    try:
        # Инициализация дрона и камер
        pioneer = Pioneer()
        cam_main_obj = Camera(camera_type=CameraType.MAIN)
        cam_opt_obj = Camera(camera_type=CameraType.OPT)
        iv = ImageViewer()

        # Камера в надир (для съёмки бассейна сверху) — нужно для vision
        try:
            ServoCamera().set_angle(-80)
        except Exception as e:
            print(f"[servo] не удалось наклонить камеру: {e}")

        print("Начало программы")

        # Нейросеть — та же VisionNode, что и в standalone-режиме (ленивая init на
        # первом кадре в video_worker). При сбое миссия продолжится без неё.
        vr = VisionNode() if VISION_AVAILABLE else None

        # Запуск потока видео (с обработкой нейросетью)
        stop_video_thread = False
        video_thread = threading.Thread(
            target=video_worker,
            args=(cam_main_obj, cam_opt_obj, iv, vr),
            daemon=True
        )
        video_thread.start()
        print("Поток вывода видео запущен")

        time.sleep(0.5)

        # Взлет
        pioneer.arm()
        print("Дрон заармлен")
        pioneer.takeoff()
        print("Дрон взлетел")

        # ======================== ТОЧКА 1 ========================
        print("Дрон двигается к точке 1")
        pioneer.go_to_local_point(-1.8, 1.9, FLIGHT_Z, 0, 5)
        while not pioneer.point_reached():
            time.sleep(0.1)
        print("Дрон висит над точкой 1")
        hover_with_countdown(pioneer, wait_time, label="[Точка 1] ")

        # ======================== ТОЧКА 2 ========================
        print("Дрон двигается к точке 2")
        pioneer.go_to_local_point(-1.8, 0.9, FLIGHT_Z, 0, 5)
        while not pioneer.point_reached():
            time.sleep(0.1)
        print("Дрон висит над точкой 2")
        hover_with_countdown(pioneer, wait_time, label="[Точка 2] ")

        # ======================== ПОСАДКА ========================
        print("Дрон направляется на посадку")
        pioneer.go_to_local_point(0, 0, 0.3, 0, 5)
        while not pioneer.point_reached():
            time.sleep(0.1)
        print("Дрон садится")
        pioneer.land()
        time.sleep(2)
        pioneer.disarm()
        print("Дрон дизармлен")

    except KeyboardInterrupt:
        print("\nПрервано пользователем (Ctrl+C)")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
    finally:
        print("Завершение работы и очистка ресурсов...")

        # Останавливаем поток видео
        stop_video_thread = True
        if video_thread is not None:
            video_thread.join(timeout=2.0)

        # Финальный вывод нейросети (итоги, лог, видео)
        if vr is not None:
            try:
                vr.finish()
            except Exception as e:
                print(f"[vision] ошибка finish: {e}")

        # Безопасная остановка дрона
        if pioneer is not None:
            try:
                pioneer.wait_callback = False
                try:
                    pioneer.land()
                    time.sleep(1)
                except Exception:
                    pass
                pioneer.disarm()
                pioneer.close_connection()
            except Exception as e:
                print(f"Ошибка при закрытии соединения с дроном: {e}")

        # Остановка камер
        for cam_name, cam_obj in [("cam_main", cam_main_obj), ("cam_opt", cam_opt_obj)]:
            if cam_obj is not None:
                try:
                    cam_obj.stop()
                except Exception:
                    pass

        # Закрытие viewer
        if iv is not None:
            try:
                iv.close()
            except Exception:
                pass

        print("Конец программы")


if __name__ == "__main__":
    main()
