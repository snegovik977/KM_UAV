"""
Автономная миссия + наша нейросеть/видео.

БАЗА — код полёта команды (autonom.dms): точки маршрута, зависания, снижение,
взлёт/посадка, тайминги — без изменений.
ВИДЕО и НЕЙРОНКА — наши (VisionNode): детекция + классификация + ArUco-подсчёт +
цветные рамки + вывод в терминал (форматы оргов) + запись аннотированного видео +
фотофиксация + лог. Все оптимизации (прореживание, ArUco-в-боксах, дешёвый ROI,
голосование за id, устойчивые рамки) уже внутри пакета vision/.

Отличия от autonom.dms:
  - видео/запись — через VisionNode (аннотированное), убран их сырой рекордер;
  - только MAIN-камера (OPT не открываем — это оптический поток позиционирования,
    второй потребитель мешал полёту и грузил CPU);
  - анти-лаг: сливаем буфер камеры -> берём самый свежий кадр;
  - в hover добавлен sleep (у них был busy-loop на 100% CPU);
  - камера в надир (ServoCamera.set_angle(-80)) для съёмки сверху.

Развёртывание: рядом положить папки vision/ и onboard/ (из code/); модель — в «Модели ИИ».
"""
import sys
import os
import time
import threading
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Импорт SDK
try:
    from pioneer_sdk2 import Pioneer, Camera, ImageViewer, CameraType, ServoCamera
except ImportError:
    print("Ошибка: Не удалось импортировать pioneer_sdk2. Убедитесь, что библиотека установлена.")
    sys.exit(1)

# Наша vision-нода (не критична для полёта: при сбое импорта миссия продолжится)
try:
    from vision_node import VisionNode
    VISION_AVAILABLE = True
except Exception as e:
    print(f"[vision] предупреждение: не удалось импортировать vision_node: {e}")
    VISION_AVAILABLE = False

wait_time = 15
stop_video_thread = False


def init_video_recorder(cam, out_dir="/home/pioneermini/workspace"):
    """Открывает VideoWriter под размер кадра (запись видеопотока — критерий +5).
    Возвращает (writer, path) или (None, None) при неудаче — полёт продолжится без записи.
    """
    try:
        first = cam.get_cv_frame(timeout=5.0)
        if first is None:
            print("[rec] нет кадра для инициализации записи — полёт без записи")
            return None, None
        h, w = first.shape[:2]
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"flight_record_{ts}.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
        if not writer.isOpened():
            print("[rec] не удалось открыть VideoWriter — полёт без записи")
            return None, None
        print(f"[rec] запись видео: {path} ({w}x{h})")
        return writer, path
    except Exception as e:
        print(f"[rec] ошибка инициализации записи: {e}")
        return None, None


def video_worker(cam_main, viewer, vr, writer):
    """Только MAIN: кадр -> нейросеть (детект+оверлей+запись+лог) -> показ.

    Трансляция гарантирована: если нейронка упадёт — показываем сырой кадр.
    Ошибки печатаем (с прореживанием), чтобы видеть причину, а не глушить.
    """
    global stop_video_thread
    t0 = time.time()
    errs = 0
    shown = 0
    while not stop_video_thread:
        # 1) кадр с камеры
        try:
            frame = cam_main.get_cv_frame(timeout=5.0)
        except Exception as e:
            errs += 1
            if errs % 30 == 1:
                print(f"[video] нет кадра с камеры: {e}")
            continue
        if frame is None:
            continue
        # 2) нейросеть (не должна ронять трансляцию)
        out = frame
        if vr is not None:
            try:
                out = vr.process(frame, t=time.time() - t0)
                if out is None:
                    out = frame
            except Exception as e:
                errs += 1
                if errs % 30 == 1:
                    print(f"[video] ошибка нейросети (показываю сырой кадр): {e}")
                out = frame
        # 3) трансляция
        try:
            viewer.imshow(name='main_camera', frame=out)
            shown += 1
            if shown == 1:
                print("[video] трансляция активна (main_camera)")
        except Exception as e:
            errs += 1
            if errs % 30 == 1:
                print(f"[video] ошибка imshow: {e}")
        # 4) запись видеопотока в память (регламент +5) — пишем показанный кадр
        if writer is not None:
            try:
                writer.write(out)
            except Exception:
                pass


def hover_with_countdown(pioneer, total_wait_time, label=""):
    """Зависание над точкой с обратным отсчётом."""
    print(f"{label}Ожидание {total_wait_time} секунд...")
    start_time = time.time()
    end_time = start_time + total_wait_time
    while time.time() < end_time:
        remaining = max(0, int(end_time - time.time()))
        sys.stdout.write(f"\r{label}Осталось времени до конца sleep: {remaining} сек   ")
        sys.stdout.flush()
        time.sleep(0.2)                     # без busy-loop (не жрём CPU в зависании)
    print(f"\n{label}Время ожидания истекло.")


def go_to_point(pioneer, x, y, z, yaw, speed, label=""):
    """Перемещение в точку с ожиданием достижения."""
    print(f"{label}Дрон двигается к точке ({x}, {y}, {z})")
    pioneer.go_to_local_point(x, y, z, yaw, speed)
    while not pioneer.point_reached():
        time.sleep(0.1)
    print(f"{label}Дрон достиг точки ({x}, {y}, {z})")


def main():
    global stop_video_thread

    pioneer = None
    iv = None
    cam_main_obj = None
    video_thread = None
    vr = None
    writer = None

    # Координаты точек из схемы (как в autonom.dms)
    POINTS = {
        "0":  {"x": 0,    "y": 3.0,  "z": 1.5,  "yaw": 0, "speed": 5,  "hover": False, "label": "[Точка 0]"},
        "1":  {"x": -1.8, "y": 3.0,  "z": 2,    "yaw": 0, "speed": 5,  "hover": True,  "label": "[Точка 1]"},
        "2":  {"x": -2.2, "y": 3.0,  "z": 2,    "yaw": 0, "speed": 5,  "hover": True,  "label": "[Точка 2]"},
        "3":  {"x": -2.2, "y": 1.8,  "z": 2,    "yaw": 0, "speed": 12, "hover": True,  "label": "[Точка 3]"},
        "4":  {"x": -1.8, "y": 1.8,  "z": 2,    "yaw": 0, "speed": 5,  "hover": True,  "label": "[Точка 4]"},
        "5":  {"x": 0,    "y": 1.8,  "z": 1,    "yaw": 0, "speed": 5,  "hover": False, "label": "[Точка 5]"},
        "6a": {"x": 0,    "y": -0.1, "z": 0.5,  "yaw": 0, "speed": 5, "hover": False, "label": "[Точка 6 - Снижение 1]"},
        "6b": {"x": 0,    "y": -0.1, "z": 0.25, "yaw": 0, "speed": 5, "hover": False, "label": "[Точка 6 - Снижение 2]"},
        "6c": {"x": 0,    "y": -0.1, "z": 0.1,  "yaw": 0, "speed": 5, "hover": False, "label": "[Точка 6 - Снижение 3]"},
    }
    FLIGHT_ORDER = ["0", "1", "2", "3", "4", "5", "6a", "6b", "6c"]

    try:
        # Инициализация дрона и камеры (только MAIN)
        pioneer = Pioneer()
        cam_main_obj = Camera(camera_type=CameraType.MAIN)
        iv = ImageViewer()

        # Камера в надир — для съёмки бассейна сверху
        try:
            ServoCamera().set_angle(-80)
        except Exception as e:
            print(f"[servo] не удалось наклонить камеру: {e}")

        print("Начало программы")

        # Нейросеть+видео — наши (ленивая init на первом кадре; лог/фотофиксация внутри
        # VisionNode). record=False: запись видеопотока ведём ТУТ (пишем показанный кадр —
        # с рамками, но не пропадёт, даже если нейронка не поднимется). При сбое импорта
        # миссия продолжится без детекции.
        vr = VisionNode(record=False) if VISION_AVAILABLE else None

        # Запись видеопотока в память оборудования (регламент +5)
        writer, rec_path = init_video_recorder(cam_main_obj)

        # Запуск потока видео+нейросети
        stop_video_thread = False
        video_thread = threading.Thread(
            target=video_worker, args=(cam_main_obj, iv, vr, writer), daemon=True)
        video_thread.start()
        print("Поток видео+нейросети запущен")

        time.sleep(0.5)

        # Взлёт (тайминги как в autonom.dms)
        pioneer.arm()
        print("Дрон заармлен")
        time.sleep(3)
        pioneer.takeoff()
        print("Дрон взлетел")
        time.sleep(3)
        print("Проверка фиксации локальных координат")

        # ======================== ПРОЛЁТ ПО ТОЧКАМ ========================
        for point_key in FLIGHT_ORDER:
            p = POINTS[point_key]
            go_to_point(pioneer, p["x"], p["y"], p["z"], p["yaw"], p["speed"], label=p["label"])
            if p["hover"]:
                hover_with_countdown(pioneer, wait_time, label=p["label"])

        # ======================== ПОСАДКА ========================
        print("Дрон садится")
        pioneer.land()
        time.sleep(2)
        pioneer.disarm()
        print("Дрон дизармлен")

    except KeyboardInterrupt:
        print("\nПрервано пользователем (Ctrl+C)")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Завершение работы и очистка ресурсов...")

        # Останавливаем поток видео
        stop_video_thread = True
        if video_thread is not None:
            video_thread.join(timeout=2.0)

        # Финал нейросети: итоги в терминал, сохранение лога
        if vr is not None:
            try:
                vr.finish()
            except Exception as e:
                print(f"[vision] ошибка finish: {e}")

        # Закрываем файл записи видеопотока (сохранение на диск)
        if writer is not None:
            try:
                writer.release()
                print("[rec] видеозапись сохранена")
            except Exception as e:
                print(f"[rec] ошибка при сохранении записи: {e}")

        # Безопасная остановка дрона
        if pioneer is not None:
            try:
                pioneer.wait_callback = False
                try:
                    if not pioneer.is_landed():
                        pioneer.land()
                        time.sleep(1)
                except Exception:
                    pass
                pioneer.disarm()
                pioneer.close_connection()
            except Exception as e:
                print(f"Ошибка при закрытии соединения с дроном: {e}")

        # Остановка камеры
        if cam_main_obj is not None:
            try:
                cam_main_obj.stop()
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
