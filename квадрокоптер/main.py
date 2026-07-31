#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Точка входа бортового ПО квадрокоптера. Подзадача 2.3.1 регламента.

Два потока, и это принципиально, а не для красоты.

  ГЛАВНЫЙ ПОТОК — перцепция: кадр -> HUD -> отладочная трансляция -> запись.
  Сюда же в пункте 2 плана встанет инференс. Он обязан крутиться непрерывно.

  ОТДЕЛЬНЫЙ ПОТОК — полётная логика (mission.Mission). При wait_callback=True
  полётные команды блокирующие: гони миссию из главного потока — и цикл перцепции
  встанет на время каждого перелёта между точками, а станции, пролетевшие под дроном,
  просто не попадут в кадр.

Обмен между потоками — только через MissionState под замком. К SDK обращается один
поток (полётный) плюс камера в главном; параллельных вызовов автопилота нет.

Запуск:
    python3 main.py                       # ждём команду takeoff от хаба
    python3 main.py --manual              # взлёт сразу, без хаба (отладка)
    PIONEER_MOCK=1 python3 main.py --manual    # вообще без дрона

На борту работает только один процесс с камерой: она отдаётся через shared memory,
и второй потребитель роняет её в TimeoutError.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

# Раскладка на борту и в репозитории разная: push_to_drone.sh кладёт всё плоско
# (protocol/ и tools/ рядом с main.py), а в репозитории они лежат в директориях
# с именами из регламента 2.9. Добавляем оба варианта — существующие.
_ЗДЕСЬ = os.path.dirname(os.path.abspath(__file__))
_КОРЕНЬ = os.path.dirname(_ЗДЕСЬ)
for _путь in (_ЗДЕСЬ,
              os.path.join(_ЗДЕСЬ, "tools"),
              os.path.join(_КОРЕНЬ, "распределительный хаб"),
              os.path.join(_КОРЕНЬ, "tools")):
    if os.path.isdir(_путь) and _путь not in sys.path:
        sys.path.insert(0, _путь)

import config as config_module            # noqa: E402
from flight import Flight, create_pioneer  # noqa: E402
from mission import Mission, MissionState  # noqa: E402
from protocol import HttpTransport, MessageFactory  # noqa: E402
from record import RecordWriter           # noqa: E402

try:
    import cv2
except ImportError:
    cv2 = None


def открыть_sdk(log):
    """Камера, трансляция и сервопривод. Настоящие или из заглушки — по PIONEER_MOCK."""
    if os.environ.get("PIONEER_MOCK"):
        from mock_pioneer import Camera, CameraType, ImageViewer, ServoCamera
    else:
        from pioneer_sdk2 import Camera, CameraType, ImageViewer, ServoCamera
    return Camera, CameraType, ImageViewer, ServoCamera


def наложить_hud(frame, текст):
    """Строка состояния поверх кадра. Дёшево и делается каждый кадр, в отличие
    от инференса: трансляция должна оставаться плавной."""
    if frame is None or cv2 is None:
        return frame
    try:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(frame, текст, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    except Exception:
        pass
    return frame


def main():
    parser = argparse.ArgumentParser(description="Бортовое ПО квадрокоптера (2.3.1)")
    parser.add_argument("--config", default=None, help="путь к config.yaml")
    parser.add_argument("--hub", default=None, help="адрес хаба (перекрывает конфиг)")
    parser.add_argument("--manual", action="store_true",
                        help="взлетать не дожидаясь пакета takeoff")
    parser.add_argument("--no-hub", action="store_true",
                        help="вообще не поднимать транспорт (оффлайн-прогон, только с --manual)")
    parser.add_argument("--no-record", action="store_true", help="не писать попытку")
    parser.add_argument("--no-camera", action="store_true", help="без камеры и трансляции")
    args = parser.parse_args()

    log = print
    cfg = config_module.load(args.config)
    if args.hub:
        cfg["hub"]["url"] = args.hub
    if args.manual:
        cfg["mission"]["manual_start"] = True
    if args.no_hub and not cfg["mission"]["manual_start"]:
        # Без транспорта команда takeoff физически не может прийти — дрон простоял бы
        # в IDLE до отмены. Лучше сказать это сразу, чем на полигоне.
        print("[main] --no-hub без --manual: команду на взлёт неоткуда получить. "
              "Добавьте --manual или уберите --no-hub.")
        return 1

    pioneer = camera = viewer = None
    transport = recorder = None
    mission = flight = None
    поток_миссии = None
    state = MissionState()

    try:
        # -------------------------------------------------------------- транспорт
        if args.no_hub:
            log("[main] --no-hub: транспорт не поднимается, телеметрия никуда не уйдёт")
        else:
            transport = HttpTransport(
                cfg["hub"]["url"], src=cfg["mission"]["src"],
                poll_interval=float(cfg["hub"]["poll_interval"]),
                timeout=float(cfg["hub"]["timeout"]),
                retries=int(cfg["hub"]["retries"]), log=log).start()
            здоровье = transport.health()
            if здоровье.get("ok"):
                log("[main] хаб %s: жив" % cfg["hub"]["url"])
            else:
                # Не авария: хаб может подняться позже дрона. Но на зачётной попытке
                # молчащий хаб означает, что команда на взлёт не придёт.
                log("[main] хаб %s НЕ ОТВЕЧАЕТ (%s)"
                    % (cfg["hub"]["url"], здоровье.get("error")))
                log("[main] полёт продолжится, но пакеты уйдут в пустоту. "
                    "Проверьте hub.url в config.yaml или ключ --hub")

        # ------------------------------------------------------------------ дрон
        pioneer = create_pioneer(log=log)
        flight = Flight(pioneer, cfg, log=log)   # нужен и в finally — см. аварийную посадку

        # ---------------------------------------------------------------- камера
        if not args.no_camera:
            Camera, CameraType, ImageViewer, ServoCamera = открыть_sdk(log)
            try:
                # Камеру вниз. Реальный минимум может быть -80, а не -90 — фактический
                # угол лежит в конфиге рядом с матрицей R_mount, измеренной при нём.
                ServoCamera().set_angle(int(cfg["camera"]["servo_angle"]))
                log("[main] сервопривод камеры: %s" % cfg["camera"]["servo_angle"])
            except Exception as e:
                log("[main] сервопривод не отработал: %s: %s" % (type(e).__name__, e))
            camera = Camera(camera_type=CameraType.MAIN)
            viewer = ImageViewer()

        # ---------------------------------------------------------------- запись
        if cfg["mission"]["record"] and not args.no_record:
            recorder = RecordWriter(out_dir=str(cfg["mission"]["record_dir"]),
                                    fps=float(cfg["mission"]["record_fps"]), log=log)

        # --------------------------------------------------------------- миссия
        mission = Mission(flight, transport, cfg, state=state, log=log,
                          factory=MessageFactory(cfg["mission"]["src"]))
        поток_миссии = threading.Thread(target=mission.run, name="mission", daemon=True)
        поток_миссии.start()
        log("[main] поток миссии запущен, главный поток — перцепция")

        # ------------------------------------------------- главный цикл перцепции
        имя_потока = str(cfg["camera"]["viewer_name"])
        слить = int(cfg["camera"].get("drain_frames", 0))
        таймаут_кадра = float(cfg["camera"]["frame_timeout"])
        ошибки = 0
        последняя_телеметрия = 0.0

        while поток_миссии.is_alive():
            snapshot = state.snapshot()

            frame = None
            if camera is not None:
                try:
                    # Анти-лаг: буфер сливается, работаем по самому свежему кадру.
                    # Иначе перцепция видит прошлое, а дрон уже улетел.
                    for _ in range(слить):
                        camera.get_cv_frame(timeout=таймаут_кадра)
                    frame = camera.get_cv_frame(timeout=таймаут_кадра)
                except Exception as e:
                    ошибки += 1
                    if ошибки % 30 == 1:
                        log("[перцепция] нет кадра (%d-я ошибка): %s: %s"
                            % (ошибки, type(e).__name__, e))

            # ЗДЕСЬ в пункте 2 плана появится инференс. Он обязан быть в try/except:
            # перцепция не имеет права ронять полёт.

            if frame is not None:
                наложить_hud(frame, state.hud())
                if viewer is not None:
                    try:
                        viewer.imshow(name=имя_потока, frame=frame)
                    except Exception as e:
                        ошибки += 1
                        if ошибки % 30 == 1:
                            log("[перцепция] трансляция: %s: %s" % (type(e).__name__, e))
                if recorder is not None:
                    recorder.frame(frame)

            if recorder is not None and time.time() - последняя_телеметрия > 0.2:
                recorder.telemetry(snapshot)
                последняя_телеметрия = time.time()

            if camera is None:
                time.sleep(0.05)      # без камеры цикл нечем притормозить

        поток_миссии.join(timeout=5.0)
        log("[main] миссия завершена: %s" % state.hud())

    except KeyboardInterrupt:
        log("\n[main] прервано оператором (Ctrl+C)")
        if mission is not None:
            mission.stop("Ctrl+C")
            if поток_миссии is not None:
                поток_миссии.join(timeout=15.0)
    except Exception as e:
        log("[main] ОШИБКА: %s: %s" % (type(e).__name__, e))
        import traceback
        traceback.print_exc()
        if mission is not None:
            mission.stop("ошибка главного потока")
            if поток_миссии is not None:
                поток_миссии.join(timeout=15.0)
    finally:
        # finally делает всё: любой из этих объектов, оставшись открытым, портит
        # следующий запуск — а на площадке следующий запуск это следующая попытка.
        log("[main] завершение, освобождаю ресурсы")

        if pioneer is not None:
            try:
                # wait_callback=False, чтобы аварийная посадка не заблокировалась
                # ожиданием события, которого уже не будет.
                pioneer.wait_callback = False

                # ⚠️ Проверка «на земле ли» идёт через Flight, а не через pioneer:
                # метода is_landed() на нашем борту НЕТ, и прямой вызов бросал бы
                # AttributeError. Раньше он глушился, и дрон получал disarm в воздухе —
                # то есть падал. Flight.is_landed() умеет обходиться дальномером.
                на_земле = flight.is_landed() if flight is not None else False
                if not на_земле:
                    log("[main] дрон не на земле — аварийная посадка")
                    try:
                        pioneer.land()
                        time.sleep(3.0)
                    except Exception as e:
                        log("[main] land() не отработал: %s: %s" % (type(e).__name__, e))

                pioneer.disarm()
                pioneer.close_connection()
            except Exception as e:
                log("[main] закрытие автопилота: %s: %s" % (type(e).__name__, e))

        # Камера отдаётся через shared memory: не остановить — следующий запуск
        # получит TimeoutError («shm_open failed»).
        for объект, метод, имя in ((camera, "stop", "камера"),
                                   (viewer, "close", "трансляция"),
                                   (recorder, "close", "запись")):
            if объект is not None:
                try:
                    getattr(объект, метод)()
                except Exception as e:
                    log("[main] %s: %s: %s" % (имя, type(e).__name__, e))

        if transport is not None:
            log("[main] %s" % transport.summary())
            transport.stop()

        log("[main] конец")
    return 0


if __name__ == "__main__":
    sys.exit(main())
