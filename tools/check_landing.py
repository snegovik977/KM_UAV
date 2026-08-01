#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Проверка центрирования по посадочному знаку «H» — БЕЗ ДРОНА И ПОЛИГОНА.

Отдельная программа под ветку центрирования (perception/landing.py + Mission._center_over_pad):
её задача — на одном экране показать, что цепочка «кадр -> детектор знака -> центр ->
точка на земле -> сведение дрона» работает и с какой точностью. То же, что делают
tests/test_landing.py, но наглядно и с настройкой параметров из командной строки —
чтобы подбирать пороги landing.* по config.yaml, не запуская pytest.

Две проверки, обе на заглушке (регламент 1.3: время утекает и на простое полигона):

  СТАТИКА   один кадр: знак рисуется honest-моделью в точке pad, детектор его находит,
            центр проецируется ОБРАТНО на землю. Печатается ошибка локализации в см —
            это и есть «на сколько дрон промахнётся мимо знака». Кадр с найденным
            знаком можно сохранить (--save) и посмотреть глазами.
  СВЕДЕНИЕ  полная миссия на заглушке: знак смещён от нуля (имитирует увод локальной СК
            за облёт), поток перцепции кормит LandingGuide, полётный автомат сводит
            дрон над знаком. Печатается итоговая точка посадки и промах.

⚠️ Чего проверка НЕ ловит (как и весь мок): заглушка рисует знак теми же интринсиками
и R_mount, которыми потом считает, поэтому ЛЮБАЯ калибровка выглядит идеальной. Здесь
проверяется логика и геометрия, а не точность оптики — её даёт только рулетка под
дроном (см. RUN.md и CLAUDE.md, раздел про заглушку камеры).

Запуск (заглушка):
    python3 tools/check_landing.py                       # сводка по нескольким смещениям
    python3 tools/check_landing.py --pad 0.3,-0.25       # одно смещение
    python3 tools/check_landing.py --height 1.2 --save кадр.png   # сохранить кадр
    python3 tools/check_landing.py --no-flight           # только статика, без миссии
    python3 tools/check_landing.py --config путь/config.yaml

Код возврата 0, если все промахи уложились в допуск landing.tol (+запас), иначе 1 —
чтобы проверку можно было гонять как регрессионную.

--------------------------------------------------------------------------------
ЖИВАЯ ПРОВЕРКА НА БОРТУ БЕЗ ЗАПУСКА МОТОРОВ (--live)
--------------------------------------------------------------------------------
Отдельно проверяет ту же ветку на НАСТОЯЩЕЙ камере и настоящем знаке, но БЕЗ ПОЛЁТА:
дрон стоит (или его держат в руке) камерой вниз над площадкой, а программа в реальном
времени показывает, находит ли детектор жёлтый знак и куда и на сколько дрон стал бы
доводиться. Ни arm(), ни takeoff(), ни land() — моторы НЕ включаются: создание объекта
Pioneer винты не крутит, крутит их только arm(), а его здесь нет.

Это и есть способ закрыть главное «НЕ ПРОВЕРЕНО» ветки — пороги цвета landing.* под
реальную краску знака (заглушка рисует и считает одними числами, реальный цвет не
проверяет). Наведи камеру на знак, подвигай дрон рукой и смотри, держится ли захват.

    python3 tools/check_landing.py --live                # камера вниз, поток на :8889
    python3 tools/check_landing.py --live --height 1.2   # если дальномер молчит — задать высоту руками
    python3 tools/check_landing.py --live --seconds 120  # ограничить время (0 = до Ctrl+C)

Высота берётся с нижнего дальномера (get_dist_sensor_data), без него — из --height.
Смотреть в браузере: http://<IP дрона>:8889/<viewer_name>/ — на нашем борту это
http://172.17.49.2:8889/main_camera/ (имя потока = camera.viewer_name в config.yaml,
не «/video»: SDK ImageViewer.imshow(name=...) отдаёт RTSP под этим именем, mediamtx
раздаёт его по :8889/<имя>/). Точный адрес программа печатает при старте --live.
На кадре знак обведён, выведен вектор доводки «вперёд/влево, см».
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time

# Консоль Windows по умолчанию не в UTF-8 и роняет вывод кириллицы. На борту и на
# платформе заочного этапа это Linux/UTF-8, там переключение — no-op. Без эмодзи в
# печати обходимся намеренно: не всякий терминал их кодирует.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_ЗДЕСЬ = os.path.dirname(os.path.abspath(__file__))
_КОРЕНЬ = os.path.dirname(_ЗДЕСЬ)
for _путь in (_ЗДЕСЬ, _КОРЕНЬ, os.path.join(_КОРЕНЬ, "квадрокоптер"),
              os.path.join(_КОРЕНЬ, "распределительный хаб")):
    if os.path.isdir(_путь) and _путь not in sys.path:
        sys.path.insert(0, _путь)

# Смещения знака от точки взлёта по умолчанию, метры (наша СК: X вперёд, Y влево).
# Ноль — знак ровно под точкой возврата (центрировать нечего, но детектор обязан
# найти); остальные — правдоподобный увод локальной СК за облёт в разные стороны.
СМЕЩЕНИЯ_ПО_УМОЛЧАНИЮ = ((0.0, 0.0), (0.30, -0.25), (-0.35, 0.30), (0.15, 0.40))


def разобрать_смещение(текст):
    части = [ч.strip() for ч in текст.split(",")]
    if len(части) != 2:
        raise argparse.ArgumentTypeError("смещение задаётся как X,Y, например 0.3,-0.25")
    try:
        return (float(части[0]), float(части[1]))
    except ValueError:
        raise argparse.ArgumentTypeError("смещение %r — не пара чисел" % текст)


def детектор_из(cfg, log):
    from perception.landing import LandingPadDetector

    раздел = cfg["landing"]
    return LandingPadDetector(
        hue_lo=раздел["hue_lo"], hue_hi=раздел["hue_hi"], sat_min=раздел["sat_min"],
        val_min=раздел["val_min"], close_px=раздел["close_px"],
        min_area_px=раздел["min_area_px"], min_fill=раздел["min_fill"], log=log)


# --------------------------------------------------------------------- статика

# Доля кадра у края, в которой знак считается «частично за кадром»: minEnclosingCircle
# по обрезанному кольцу смещает центр, и одиночный снимок такому знаку верить не должен —
# как раз поэтому центрирование итеративное, а локализация берёт центральные 50 % кадра.
_ПОЛЕ_КРАЯ = 0.04


def статическая_проверка(cfg, калибровка, детектор, pad, height, save=None):
    """Один кадр: найти знак и вернуть ошибку локализации его центра, метры.

    Возвращает (ошибка_м | None, у_края, строка). None — знак не найден; у_края True —
    знак прижат к границе кадра и обрезан, ошибке одиночного снимка верить нельзя.
    """
    from localization import Pose, pixel_to_ground
    from mock_pioneer import MockCamera

    поза = Pose(0.0, 0.0, float(height), 0.0, 0.0, 0.0)
    камера = MockCamera(fps=1e6, stations=[], cfg=cfg, pad=pad)
    камера.pose = lambda: поза
    кадр = камера._синтетика()
    высота_кадра, ширина_кадра = кадр.shape[:2]

    детекция = детектор.detect(кадр)
    if детекция is None:
        return None, False, ("pad=(%+.2f,%+.2f) h=%.1f  ЗНАК НЕ НАЙДЕН"
                             % (pad[0], pad[1], height))

    точка = pixel_to_ground(детекция.cx, детекция.cy, поза,
                            калибровка.intrinsics, калибровка.r_mount)
    if точка is None:
        return None, False, ("pad=(%+.2f,%+.2f) h=%.1f  центр не спроецировался на землю"
                             % (pad[0], pad[1], height))

    у_края = (детекция.cx < _ПОЛЕ_КРАЯ * ширина_кадра
              or детекция.cx > (1 - _ПОЛЕ_КРАЯ) * ширина_кадра
              or детекция.cy < _ПОЛЕ_КРАЯ * высота_кадра
              or детекция.cy > (1 - _ПОЛЕ_КРАЯ) * высота_кадра)
    ошибка = math.hypot(точка[0] - pad[0], точка[1] - pad[1])
    строка = ("pad=(%+.2f,%+.2f) h=%.1f  центр в пикселях (%.0f,%.0f) заполнение %.0f%%  "
              "-> земля (%+.3f,%+.3f)  ошибка %.1f см"
              % (pad[0], pad[1], height, детекция.cx, детекция.cy, 100.0 * детекция.score,
                 точка[0], точка[1], 100.0 * ошибка))

    if save:
        _сохранить_кадр(кадр, детекция, save)
        строка += "  [кадр -> %s]" % save
    return ошибка, у_края, строка


def _сохранить_кадр(кадр, детекция, path):
    try:
        import cv2

        центр = (int(детекция.cx), int(детекция.cy))
        cv2.circle(кадр, центр, int(детекция.radius), (0, 255, 255), 2)
        cv2.drawMarker(кадр, центр, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.imwrite(path, кадр)
    except Exception as e:
        print("не удалось сохранить кадр: %s: %s" % (type(e).__name__, e))


# -------------------------------------------------------------------- сведение

def _ускорить(cfg):
    """Урезать тайминги: это проверка, а не полёт. Пороги детектора не трогаем."""
    cfg["flight"]["arm_settle"] = 0.05
    cfg["flight"]["takeoff_settle"] = 0.05
    cfg["flight"]["settle"] = 0.02
    cfg["flight"]["point_timeout"] = 3.0
    cfg["mission"]["manual_start"] = True
    cfg["mission"]["telemetry_hz"] = 20.0
    cfg["loop"]["length"] = 1.0
    cfg["loop"]["width"] = 1.0
    return cfg


def проверка_сведения(cfg, калибровка, детектор, pad):
    """Полная миссия на заглушке со знаком в точке pad. Возвращает (промах_м, строка)."""
    from flight import Flight
    from mission import LAND, RETURN, Mission, MissionState
    from mock_pioneer import MockCamera, MockPioneer
    from perception.landing import LandingGuide
    from tasks import TaskProfile

    pioneer = MockPioneer(speed=6.0)          # станет _последним_дроном для камеры
    state = MissionState()
    flight = Flight(pioneer, cfg, log=lambda _: None)
    mission = Mission(flight, None, cfg, state=state, log=lambda _: None,
                      task=TaskProfile("2.3.1"))

    камера = MockCamera(fps=60.0, stations=[], cfg=cfg, pad=pad)
    гид = LandingGuide(детектор, калибровка, state, phases=(RETURN, LAND),
                       log=lambda _: None)

    стоп = threading.Event()

    def перцепция():
        while not стоп.is_set():
            гид.update(камера._синтетика())
            time.sleep(0.01)

    поток = threading.Thread(target=перцепция, daemon=True)
    поток.start()
    try:
        mission.run()
    finally:
        стоп.set()
        поток.join(timeout=2.0)

    x, y, _ = flight.position()
    промах = math.hypot(x - pad[0], y - pad[1])
    авария = state.snapshot()["abort_reason"]
    строка = ("pad=(%+.2f,%+.2f)  сел в (%+.3f,%+.3f)  промах %.1f см  захватов знака %d"
              % (pad[0], pad[1], x, y, 100.0 * промах, гид.hits))
    if авария:
        строка += "  АВАРИЯ: %s" % авария
    return промах, строка


# ------------------------------------------------- живая проверка без моторов

def _ip_борта():
    """IP, по которому крутится видеопоток. На заглушке — localhost; на борту адрес
    даёт сам SDK-транспорт (шлюз Wi-Fi дрона), иначе берём известный адрес нашего борта.
    Только для печати ссылки — на логику не влияет."""
    if os.environ.get("PIONEER_MOCK"):
        return "127.0.0.1"
    for пер in ("PIONEER_IP", "DRONE_IP"):
        зн = os.environ.get(пер)
        if зн:
            return зн.strip()
    return "172.17.49.2"                        # наш борт, проверено 31.07.2026 (CLAUDE.md)


def _открыть_sdk():
    """Камера, тип, трансляция и сервопривод — настоящие или из заглушки (PIONEER_MOCK).
    Ровно как в квадрокоптер/main.py: на борту и на ноутбуке один и тот же код."""
    if os.environ.get("PIONEER_MOCK"):
        from mock_pioneer import Camera, CameraType, ImageViewer, ServoCamera
    else:
        from pioneer_sdk2 import Camera, CameraType, ImageViewer, ServoCamera
    return Camera, CameraType, ImageViewer, ServoCamera


def проверка_вживую(cfg, калибровка, детектор, seconds, height_override, log):
    """Настоящая камера + знак, но БЕЗ ПОЛЁТА. Показывает захват знака и вектор доводки.

    Моторы не включаются: тут нет ни arm(), ни takeoff(), ни land() — только чтение
    камеры, дальномера и ориентации. Создание Pioneer винты не крутит.
    """
    from flight import Flight, create_pioneer
    from localization import Pose, pixel_to_ground

    try:
        import cv2
    except ImportError:
        cv2 = None

    Camera, CameraType, ImageViewer, ServoCamera = _открыть_sdk()
    имя_потока = str(cfg["camera"]["viewer_name"])
    таймаут_кадра = float(cfg["camera"]["frame_timeout"])
    h_min = float(cfg["flight"]["h_min"])
    допуск = float(cfg["landing"]["tol"])

    pioneer = create_pioneer(log=log)
    flight = Flight(pioneer, cfg, log=log)     # НЕ вооружаем — только сенсоры
    камера = viewer = None
    захватов = кадров = 0
    подряд_без_кадра = 0                        # счётчик таймаутов подряд — для подсказки
    транслируется = False                      # первый успешный imshow -> подтверждаем поток
    log("[live] БЕЗ МОТОРОВ: arm/takeoff/land не вызываются. Держи дрон камерой вниз "
        "над знаком; Ctrl+C — выход.")
    # Куда смотреть: имя потока (camera.viewer_name) отдаётся SDK по RTSP и раздаётся
    # mediamtx на :8889/<имя>/. НЕ «/video» — этой ошибкой поток и «не показывался».
    log("[live] видеопоток: http://%s:8889/%s/  (открой в браузере, пока идёт --live)"
        % (_ip_борта(), имя_потока))
    try:
        # Сервопривод — не двигатель: он опускает камеру к земле, без него знак не в кадре.
        try:
            угол = int(cfg["camera"]["servo_angle"])
            ServoCamera().set_angle(угол)
            log("[live] камера опущена на %d°" % угол)
        except Exception as e:
            log("[live] сервопривод не отработал: %s: %s" % (type(e).__name__, e))
        камера = Camera(camera_type=CameraType.MAIN)
        viewer = ImageViewer()

        дедлайн = (time.time() + seconds) if seconds > 0 else None
        печать = 0.0
        while дедлайн is None or time.time() < дедлайн:
            try:
                frame = камера.get_cv_frame(timeout=таймаут_кадра)
            except Exception as e:
                подряд_без_кадра += 1
                log("[live] нет кадра: %s: %s" % (type(e).__name__, e))
                # Кадры не идут не из-за нашего кода: обычно завис продюсер камеры на
                # борту (shared memory /tmp/maincamera пуст, shm_open failed). Тогда
                # даже штатный camera_trans.py молчит. Подсказываем, как поднять, а не
                # молотим таймаутами вслепую.
                if подряд_без_кадра == 3:
                    log("[live] камера молчит подряд. Скорее всего завис продюсер потока "
                        "на борту (не наш код). На борту: pgrep -af maincamera; убить "
                        "зависший gst/shm-процесс — mediamtx поднимет новый. Проверить "
                        "поток можно штатным camera_trans.py из референс-репозитория.")
                time.sleep(0.2)
                continue
            if frame is None:
                time.sleep(0.05)
                continue
            подряд_без_кадра = 0
            кадров += 1

            # Высота: дальномер, иначе заданная руками. Без высоты луч на землю не
            # спроецировать — покажем только пиксель знака.
            высота = height_override if height_override else flight.range_down()
            ориентация = flight.orientation() or (0.0, 0.0, 0.0)
            roll, pitch, _ = калибровка.fix_attitude(*ориентация)

            детекция = детектор.detect(frame)
            строка = None
            if детекция is None:
                строка = "знак не виден"
            elif высота is None or высота <= 0.05:
                строка = ("знак в пикселе (%.0f,%.0f), но высота неизвестна — задай "
                          "--height" % (детекция.cx, детекция.cy))
            else:
                поза = Pose(0.0, 0.0, float(высота), roll, pitch, 0.0)
                точка = pixel_to_ground(детекция.cx, детекция.cy, поза,
                                        калибровка.intrinsics, калибровка.r_mount)
                if точка is None:
                    строка = "знак виден, но луч не встретил землю (наклон/высота)"
                else:
                    захватов += 1
                    dist = math.hypot(точка[0], точка[1])
                    в_допуске = "В ДОПУСКЕ" if dist <= допуск else "доводка"
                    строка = ("знак: вперёд %+.2f м, влево %+.2f м  (%.0f см, %s)  h=%.2f м"
                              % (точка[0], точка[1], 100.0 * dist, в_допуске, высота))
                    _наложить_вектор(frame, детекция, точка, dist <= допуск, cv2)

            if детекция is not None:
                _обвести_знак(frame, детекция, cv2)
            _подпись(frame, "LIVE (без моторов)  " + (строка or ""), cv2)
            if viewer is not None:
                try:
                    viewer.imshow(name=имя_потока, frame=frame)
                    if not транслируется:      # первый показанный кадр — поток поднялся
                        транслируется = True
                        log("[live] трансляция активна: http://%s:8889/%s/"
                            % (_ip_борта(), имя_потока))
                except Exception as e:
                    log("[live] imshow не отработал: %s: %s" % (type(e).__name__, e))

            # Печать не чаще 2 Гц, чтобы не залить консоль; поток остаётся плавным.
            if time.time() - печать > 0.5:
                log("[live] " + (строка or ""))
                печать = time.time()
    except KeyboardInterrupt:
        log("\n[live] остановлено оператором")
    finally:
        # Никакого disarm/land: моторы не запускались. Только освобождаем камеру —
        # иначе следующий запуск не откроет shared memory.
        for объект, метод in ((камера, "stop"), (viewer, "close")):
            if объект is not None:
                try:
                    getattr(объект, метод)()
                except Exception:
                    pass
        try:
            pioneer.close_connection()
        except Exception:
            pass

    log("\nкадров %d, знак захвачен на %d из них" % (кадров, захватов))
    if кадров and захватов == 0:
        log("знак ни разу не локализован. Если он в кадре — подвинуть пороги landing.* "
            "(цвет/площадь) в config.yaml и/или задать --height. См. RUN.md")
        return 1
    return 0


def _обвести_знак(frame, детекция, cv2):
    if frame is None or cv2 is None:
        return
    try:
        центр = (int(детекция.cx), int(детекция.cy))
        cv2.circle(frame, центр, int(детекция.radius), (0, 255, 255), 2)
        cv2.drawMarker(frame, центр, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
    except Exception:
        pass


def _наложить_вектор(frame, детекция, точка, в_допуске, cv2):
    """Стрелка от центра кадра к знаку: наглядно, куда пошёл бы дрон."""
    if frame is None or cv2 is None:
        return
    try:
        высота_кадра, ширина_кадра = frame.shape[:2]
        начало = (ширина_кадра // 2, высота_кадра // 2)
        конец = (int(детекция.cx), int(детекция.cy))
        цвет = (80, 220, 80) if в_допуске else (0, 165, 255)
        cv2.arrowedLine(frame, начало, конец, цвет, 2, tipLength=0.2)
    except Exception:
        pass


def _подпись(frame, текст, cv2):
    if frame is None or cv2 is None:
        return
    try:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(frame, текст, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    except Exception:
        pass


# ------------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(
        description="Проверка центрирования по посадочному знаку без дрона")
    parser.add_argument("--config", default=None, help="путь к config.yaml")
    parser.add_argument("--pad", type=разобрать_смещение, default=None,
                        help="одно смещение знака X,Y (иначе прогон по набору)")
    parser.add_argument("--height", type=float, default=None,
                        help="высота для статической проверки, м (по умолчанию landing.center_height)")
    parser.add_argument("--save", default=None,
                        help="сохранить кадр статической проверки с найденным знаком в файл")
    parser.add_argument("--no-flight", action="store_true",
                        help="только статика (детектор+локализация), без прогона миссии")
    parser.add_argument("--live", action="store_true",
                        help="живая проверка на настоящей камере БЕЗ ЗАПУСКА МОТОРОВ: "
                             "детекция знака и вектор доводки в реальном времени")
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="ограничить --live по времени, с (0 = до Ctrl+C)")
    args = parser.parse_args()

    try:
        import numpy  # noqa: F401
        import cv2     # noqa: F401
    except ImportError as e:
        print("Нужны numpy и cv2 (%s). Проверка работает только там, где они есть." % e)
        return 2

    import config as config_module
    from perception.calib import from_config as калибровка_из_конфига

    cfg = config_module.load(args.config)
    if not cfg.get("landing", {}).get("enabled", False):
        print("ВНИМАНИЕ: landing.enabled=false в конфиге — в бою центрирование ВЫКЛЮЧЕНО. "
              "Проверяю логику всё равно.")
    калибровка = калибровка_из_конфига(cfg, log=lambda _: None)
    детектор = детектор_из(cfg, log=lambda _: None)

    if args.live:
        # Живая проверка на настоящей камере, без моторов. Свой поток исполнения:
        # ни статики, ни миссии, ни кода возврата по допуску.
        return проверка_вживую(cfg, калибровка, детектор, args.seconds, args.height,
                               log=print)

    допуск = float(cfg["landing"]["tol"])
    height = args.height if args.height is not None else float(cfg["landing"]["center_height"])
    смещения = [args.pad] if args.pad is not None else list(СМЕЩЕНИЯ_ПО_УМОЛЧАНИЮ)

    print("=" * 78)
    print("ПРОВЕРКА ЦЕНТРИРОВАНИЯ ПО ПОСАДОЧНОМУ ЗНАКУ (заглушка, без дрона)")
    print("калибровка: %s" % калибровка)
    if not калибровка.measured:
        print("  (!) интринсики НЕ ИЗМЕРЕНЫ — заглушка рисует и считает одними числами, "
              "точность здесь всегда идеальна. Реальную проверяет только рулетка.")
    print("допуск landing.tol = %.0f см, высота статики = %.1f м" % (100.0 * допуск, height))
    print("=" * 78)

    # Статика — диагностика точности локализации, не приговор: знак у края кадра
    # одиночным снимком локализуется хуже (кольцо обрезано), и именно поэтому
    # центрирование итеративное. ИТОГ считаем по сведению — это и есть проверяемая ветка.
    ошибки_в_центре = []
    не_найдено = 0

    print("\n[1] СТАТИКА: кадр -> детектор -> центр -> точка на земле (диагностика)")
    for i, pad in enumerate(смещения):
        сохранить = args.save if (args.save and (args.pad is not None or i == 0)) else None
        ошибка, у_края, строка = статическая_проверка(cfg, калибровка, детектор, pad,
                                                      height, save=сохранить)
        if ошибка is None:
            метка = "  ЗНАК НЕ НАЙДЕН"
            не_найдено += 1
        elif у_края:
            метка = "  (знак у края кадра — одиночному снимку не верим, сводит полёт)"
        elif ошибка <= допуск:
            метка = "  ok"
            ошибки_в_центре.append(ошибка)
        else:
            метка = "  ВЕЛИКА для знака в центре кадра"
            ошибки_в_центре.append(ошибка)
        print("  " + строка + метка)

    провалов = 0
    if not args.no_flight:
        print("\n[2] СВЕДЕНИЕ: полная миссия на заглушке, дрон сводится над знаком")
        # Урезаем тайминги: это проверка логики, а не полёт. Пороги детектора и
        # калибровку (уже прочитаны выше) правка не трогает.
        _ускорить(cfg)
        for pad in смещения:
            промах, строка = проверка_сведения(cfg, калибровка, детектор, pad)
            # Запас к допуску: последний шаг П-регулятора не доводит ровно в ноль,
            # плюс дискретность кадров. 5 см сверх tol — тот же порог, что в тесте.
            порог = допуск + 0.05
            метка = "  ok" if промах <= порог else "  ПРОВАЛ"
            print("  " + строка + метка)
            if промах > порог:
                провалов += 1

    print("\n" + "=" * 78)
    if ошибки_в_центре:
        print("локализация знака в центре кадра: макс %.1f см, средняя %.1f см"
              % (100.0 * max(ошибки_в_центре),
                 100.0 * (sum(ошибки_в_центре) / len(ошибки_в_центре))))
    if не_найдено:
        print("знак не найден в %d статических кадрах — проверить пороги landing.* (цвет!)"
              % не_найдено)
    if args.no_flight:
        print("ИТОГ: статика пройдена (сведение пропущено ключом --no-flight)")
        return 1 if не_найдено else 0
    if провалов == 0:
        print("ИТОГ: дрон свёлся над знаком во всех прогонах, в допуске — OK")
        return 0
    print("ИТОГ: провалов сведения — %d FAIL (подобрать landing.* в config.yaml, RUN.md)"
          % провалов)
    return 1


if __name__ == "__main__":
    sys.exit(main())
