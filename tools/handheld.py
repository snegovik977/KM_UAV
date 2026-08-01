#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Камера дрона с рамками станций — БЕЗ ВЗЛЁТА. Дрон носят на руках.

Зачем. Всё, что мы знаем про детекцию, проверено на заглушке, которая рисует идеальный
прямоугольник на однородном фоне. Настоящий пол — тёмно-серый, с белой разметкой,
логотипами, швами и тенями, а панель тёмная с яркой радужной рамкой. Совпадёт ли
это с порогами из config.yaml, на заглушке узнать нельзя, а тратить на это попытку
(регламент 1.3 — время полигона идёт) незачем.

Что делает: берёт кадр с камеры, гонит его через ТОТ ЖЕ конвейер, что и в полёте
(та же калибровка, тот же детектор, те же пороги из config.yaml), рисует результат
и отдаёт в трансляцию. Смотреть в браузере:

    http://172.17.49.2:8889/main_camera

⚠️ ПОЛЁТНЫХ КОМАНД ЗДЕСЬ НЕТ ВООБЩЕ: ни arm, ни takeoff, ни go_to_local_point.
Винты не крутятся. Дрон можно держать в руках и водить над станциями.

Высота берётся с нижнего дальномера — на руках он работает так же, как в воздухе.
Если дальномер молчит, высоту задают ключом --height.

Координаты станций считаются ОТНОСИТЕЛЬНО САМОГО ДРОНА (он же и есть начало отсчёта,
раз никуда не летел): «+0.40 вперёд, -0.12 влево» читается как «станция в 40 см перед
вами и в 12 см правее». Это и есть проверка локализации рулеткой из RUN.md §7.6,
только вживую.

Запуск на борту:
    python3 tools/handheld.py                  # смотреть в браузере
    python3 tools/handheld.py --height 1.5     # если дальномер не отвечает
    python3 tools/handheld.py --record         # ещё и записать видео + телеметрию
    python3 tools/handheld.py --quiet          # без построчного отчёта в консоль

Второй режим, БЕЗ ДРОНА ВООБЩЕ: прогнать те же пороги по готовым снимкам.

    python tools/handheld.py --image "фото станции/чистые/*.jpg" --height 1.5 --out разбор/

⚠️ У снимков с телефона другая оптика, поэтому площадь в м² по ним считается неверно —
доверять можно тому, СОБРАЛАСЬ ЛИ станция в одно пятно и не лезет ли в кандидаты
разметка, логотипы и тени. Пороги площади проверяются только кадром с камеры дрона.

На своей машине живой режим гоняется на заглушке, но станций в нём не видно:
заглушка камеры рисует их от позиции СВОЕГО дрона, а он никуда не взлетал. Проверять
логику детекции на заглушке нужно через `квадрокоптер/main.py`, здесь — только то,
что инструмент поднимается и не падает.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import imgio                            # чтение/запись по путям с кириллицей

_ЗДЕСЬ = os.path.dirname(os.path.abspath(__file__))
_КОРЕНЬ = os.path.dirname(_ЗДЕСЬ)
# Раскладка двойная: в репозитории модули лежат в «квадрокоптер/», на борту — плоско
# рядом с main.py. Ищем в обеих, как это делают main.py и tests/conftest.py.
for _путь in (os.path.join(_КОРЕНЬ, "квадрокоптер"),
              os.path.join(_КОРЕНЬ, "распределительный хаб"), _КОРЕНЬ, _ЗДЕСЬ):
    if os.path.isdir(_путь) and _путь not in sys.path:
        sys.path.insert(0, _путь)

try:
    import cv2
except ImportError:
    cv2 = None


ЦВЕТ_ПРИНЯТО = (80, 220, 80)
ЦВЕТ_ОТВЕРГНУТО = (60, 60, 235)
ЦВЕТ_HUD = (240, 240, 240)


def открыть_sdk():
    """Камера, трансляция и сервопривод. Настоящие или из заглушки — по PIONEER_MOCK."""
    if os.environ.get("PIONEER_MOCK"):
        from mock_pioneer import Camera, CameraType, ImageViewer, ServoCamera
    else:
        from pioneer_sdk2 import Camera, CameraType, ImageViewer, ServoCamera
    return Camera, CameraType, ImageViewer, ServoCamera


def подписать(frame, строки):
    """Несколько строк HUD в левом верхнем углу. Латиницей: cv2.putText не умеет
    кириллицу и вместо букв рисует «???»."""
    if cv2 is None:
        return
    for номер, текст in enumerate(строки):
        cv2.putText(frame, текст, (8, 20 + 18 * номер),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, текст, (8, 20 + 18 * номер),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, ЦВЕТ_HUD, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(
        description="Камера дрона с рамками станций, без взлёта")
    parser.add_argument("--config", default=None, help="путь к config.yaml")
    parser.add_argument("--height", type=float, default=None,
                        help="высота камеры над полом, м — если дальномер молчит")
    parser.add_argument("--record", action="store_true",
                        help="писать records/flight_*.mp4 и .jsonl для tools/replay.py")
    parser.add_argument("--quiet", action="store_true",
                        help="не печатать построчный отчёт по детекциям")
    parser.add_argument("--every", type=float, default=1.0,
                        help="как часто печатать отчёт, с")
    parser.add_argument("--image", default=None, metavar="МАСКА",
                        help="разобрать готовые снимки вместо камеры (без дрона). "
                             "Маска в кавычках, например \"фото станции/чистые/*.jpg\"")
    parser.add_argument("--out", default=None,
                        help="куда сложить размеченные снимки в режиме --image")
    parser.add_argument("--mjpeg", type=int, default=8080, metavar="ПОРТ",
                        help="своя трансляция в браузер, не зависящая от SDK "
                             "(0 — выключить). По умолчанию 8080")
    parser.add_argument("--no-viewer", action="store_true",
                        help="не трогать ImageViewer из SDK вообще")
    parser.add_argument("--set", action="append", metavar="КЛЮЧ=ЗНАЧЕНИЕ",
                        help="подменить настройку на время запуска, например "
                             "--set detector.close_px=21. Можно несколько раз. "
                             "config.yaml при этом не трогается")
    args = parser.parse_args()

    # Те же модули и теми же именами, что поднимает main.py -> поднять_перцепцию():
    # инструмент обязан проверять боевой конвейер, а не свою копию.
    import config as config_module
    from flight import Flight, create_pioneer
    from fusion import StationRegistry
    from mission import MissionState, SURVEY
    from perception.calib import from_config as калибровка_из_конфига
    from perception.detector import create_detector
    from perception.pipeline import Perception

    # Подбор порогов идёт ключами, а не правкой config.yaml на борту: иначе
    # подобранное «на попробовать» рано или поздно уедет в зачётную попытку.
    # Функция живёт в config.py — единственном модуле, который есть и на борту,
    # и в tools/. Импорт из соседнего инструмента здесь уже ломался: replay.py
    # на борт не заливается.
    cfg = config_module.load(args.config)
    config_module.применить_правки(cfg, getattr(args, "set", None), prefix="[руки]")
    log = print

    камера = viewer = pioneer = recorder = свой = None
    try:
        # --------------------------------------------------------------- перцепция
        калибровка = калибровка_из_конфига(cfg, log=log)
        детектор = create_detector(cfg, log=log)
        state = MissionState()
        # SURVEY, потому что перцепция принимает наблюдения только во время облёта.
        # Здесь облёта нет, но смотреть мы хотим именно то, что увидел бы дрон в нём.
        state.update(state=SURVEY)
        реестр = StationRegistry(cfg, log=log)
        перцепция = Perception(cfg, state, реестр, калибровка, детектор, log=log)
        перцепция.every_n = 1            # на руках спешить некуда, считаем каждый кадр
        перцепция.check(float(cfg["flight"]["h_survey"]))

        # ------------------------------------------ режим снимков: дрон не нужен
        if args.image:
            return разобрать_снимки(args, cfg, state, перцепция, детектор,
                                    калибровка, SURVEY, log)

        # ------------------------------------------------------------------- дрон
        # Flight нужен ровно для двух чтений — дальномер и углы. Ни одна команда
        # движения отсюда не вызывается, и вызываться не должна.
        try:
            pioneer = create_pioneer(log=log)
        except ImportError:
            # Голый traceback про pioneer_sdk2 на площадке читается как «всё сломалось».
            # На деле это просто запуск не на борту, и вариантов ровно три.
            log("")
            log("[руки] pioneer_sdk2 нет — значит это не борт, а обычный компьютер.")
            log("[руки] Живой режим работает ТОЛЬКО на дроне:")
            log("       python launch.py --push")
            log("       ssh pioneermini@172.17.49.2")
            log("       cd workspace && python3 tools/handheld.py")
            log("")
            log("[руки] Отсюда можно другое — прогнать те же пороги по снимкам:")
            log("       python tools/handheld.py --image \"фото станции/*/*.jpg\" "
                "--height 1.5 --out разбор/")
            log("")
            log("[руки] И есть заглушка, но станций в ней не видно (она рисует их")
            log("       от позиции своего дрона, а он не взлетал):")
            log("       PIONEER_MOCK=1 python tools/handheld.py --height 1.5")
            return 1
        flight = Flight(pioneer, cfg, log=log)
        raw = flight._raw_orientation()
        if raw:
            # То же, что делает arm(): принять текущий курс за нулевой. Тогда
            # координаты станций читаются относительно того, как вы держите дрон.
            flight.yaw0 = raw[2]

        Camera, CameraType, ImageViewer, ServoCamera = открыть_sdk()
        try:
            ServoCamera().set_angle(int(cfg["camera"]["servo_angle"]))
            log("[руки] сервопривод камеры: %s" % cfg["camera"]["servo_angle"])
        except Exception as e:
            log("[руки] сервопривод не отработал: %s: %s" % (type(e).__name__, e))
        камера = Camera(camera_type=CameraType.MAIN)

        # ImageViewer из SDK — вещь ненадёжная: по документации Geoscan он доступен
        # ТОЛЬКО при драйвере камеры gstreamer, отдаёт RTSP и молчит, если что-то
        # не так. Поэтому он больше не обязателен, а рядом всегда поднимается своя
        # трансляция (tools/mjpeg.py), которой хватает обычного браузера.
        if not args.no_viewer:
            try:
                viewer = ImageViewer()
                log("[руки] ImageViewer из SDK поднялся: %s"
                    % type(viewer).__module__)
            except Exception as e:
                viewer = None
                log("[руки] ImageViewer не поднялся (%s: %s). Скорее всего, драйвер "
                    "камеры не gstreamer. Смотрите свою трансляцию ниже."
                    % (type(e).__name__, e))

        if args.mjpeg:
            from mjpeg import MjpegServer
            свой = MjpegServer(port=args.mjpeg, log=log)
            if not свой.start():
                свой = None

        if args.record:
            from record import RecordWriter
            recorder = RecordWriter(out_dir=str(cfg["mission"]["record_dir"]),
                                    fps=float(cfg["mission"]["record_fps"]), log=log)

        имя_потока = str(cfg["camera"]["viewer_name"])
        таймаут = float(cfg["camera"]["frame_timeout"])
        слить = int(cfg["camera"].get("drain_frames", 0))
        мин_площадь = float(cfg["detector"]["min_area_m2"])
        макс_площадь = float(cfg["detector"]["max_area_m2"])

        log("")
        log("[руки] ВЗЛЁТА НЕ БУДЕТ: полётных команд в этом инструменте нет.")
        if свой is not None:
            log("[руки] СМОТРЕТЬ ЗДЕСЬ:  http://%s:%d/" % (_адрес_борта(), args.mjpeg))
            log("[руки] это своя трансляция, ей не нужны ни gstreamer, ни RTSP")
        if viewer is not None:
            log("[руки] через SDK, если работает: http://%s:8889/%s  (браузер, WebRTC)"
                % (_адрес_борта(), имя_потока))
            log("[руки]                           rtsp://%s:8554/%s  (VLC)"
                % (_адрес_борта(), имя_потока))
            log("[руки] оба адреса живы, ТОЛЬКО пока крутится эта программа")
        log("[руки] пороги площади: %.3f-%.3f м2, выход — Ctrl+C"
            % (мин_площадь, макс_площадь))
        log("")

        кадров = 0
        печатали = 0.0
        ошибки = 0
        while True:
            try:
                for _ in range(слить):
                    камера.get_cv_frame(timeout=таймаут)
                frame = камера.get_cv_frame(timeout=таймаут)
            except Exception as e:
                ошибки += 1
                if ошибки % 30 == 1:
                    log("[руки] нет кадра (%d-я ошибка): %s: %s"
                        % (ошибки, type(e).__name__, e))
                time.sleep(0.1)
                continue
            if frame is None:
                continue
            кадров += 1

            # ------------------------------------------------------------- поза
            высота = args.height
            if высота is None:
                высота = flight.range_down()
            if высота is None or высота <= 0:
                подписать(frame, ["NO HEIGHT: dalnomer molchit, zadaite --height"])
                _показать(viewer, свой, имя_потока, frame)
                if not args.quiet and time.time() - печатали > args.every:
                    печатали = time.time()
                    log("[руки] высоты нет: дальномер молчит. Задайте --height")
                continue
            углы = flight.orientation() or (0.0, 0.0, 0.0)
            state.update(position=(0.0, 0.0, float(высота)),
                         roll=углы[0], pitch=углы[1], yaw=углы[2])

            поза = перцепция.pose()
            отвергнутые = []
            детекции = детектор.detect(frame, area_m2=перцепция._площадь(поза),
                                       rejected=отвергнутые)

            # ---------------------------------------------------------- отрисовка
            перцепция.draw(frame, детекции)
            for детекция, причина, число in отвергнутые:
                x1, y1, x2, y2 = детекция.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), ЦВЕТ_ОТВЕРГНУТО, 1)
                cv2.putText(frame, "%s %.3g" % (_латиницей(причина), число),
                            (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            ЦВЕТ_ОТВЕРГНУТО, 1, cv2.LINE_AA)
            подписать(frame, [
                "h=%.2f m  roll=%+.1f pitch=%+.1f" % (высота, углы[0], углы[1]),
                "prinyato %d   otvergnuto %d" % (len(детекции), len(отвергнутые)),
                "porog ploshadi %.3f-%.3f m2" % (мин_площадь, макс_площадь),
            ])
            _показать(viewer, свой, имя_потока, frame)
            if recorder is not None:
                recorder.frame(frame)
                recorder.event({"t": time.time(), "state": SURVEY,
                                "position": [0.0, 0.0, высота],
                                "roll": углы[0], "pitch": углы[1], "yaw": углы[2],
                                "origin": [0.0, 0.0, 0.0], "frame": кадров})

            # ------------------------------------------------------------- отчёт
            if args.quiet or time.time() - печатали < args.every:
                continue
            печатали = time.time()
            log("--- кадр %d, высота %.2f м, крен %+.1f, тангаж %+.1f"
                % (кадров, высота, углы[0], углы[1]))
            if not детекции and not отвергнутые:
                log("    пусто: ни одного кандидата. Станция вне кадра, либо она "
                    "не отличается от фона (detector.contrast_*)")
            for детекция in детекции:
                _напечатать(log, "СТАНЦИЯ", детекция, поза, калибровка, frame.shape)
            for детекция, причина, число in отвергнутые:
                _напечатать(log, "отвергнут (%s %.3g)" % (причина, число),
                            детекция, поза, калибровка, frame.shape)

    except KeyboardInterrupt:
        print("\n[руки] остановлено оператором")
    finally:
        # Камера отдаётся через shared memory: не освободишь — следующий запуск
        # получит TimeoutError и будет выглядеть как «дрон сломался».
        for имя, объект, метод in (("камера", камера, "stop"),
                                   ("трансляция", свой, "close"),
                                   ("запись", recorder, "close")):
            try:
                if объект is not None:
                    getattr(объект, метод)()
            except Exception as e:
                print("[руки] %s не закрылась: %s: %s" % (имя, type(e).__name__, e))
    return 0


def разобрать_снимки(args, cfg, state, перцепция, детектор, калибровка, SURVEY, log):
    """Те же пороги по готовым снимкам. Дрона, камеры и полигона не требует.

    Нужен, чтобы посмотреть на детекцию по фотографиям настоящих станций ещё до того,
    как борт окажется под рукой: собирается ли панель в одно пятно, не лезут ли
    в кандидаты разметка, логотипы и тени.
    """
    import glob

    высота = args.height or float(cfg["flight"]["h_survey"])
    state.update(state=SURVEY, position=(0.0, 0.0, высота),
                 roll=0.0, pitch=0.0, yaw=0.0)
    поза = перцепция.pose()

    пути = sorted(glob.glob(args.image))
    if not пути:
        log("[снимки] по маске %r ничего не нашлось" % args.image)
        return 1
    if args.out and not os.path.isdir(args.out):
        os.makedirs(args.out)

    log("")
    log("[снимки] %d файлов, высота для геометрии %.2f м" % (len(пути), высота))
    log("[снимки] ⚠️ площадь в м² по телефонным снимкам НЕВЕРНА — у них другая оптика. "
        "Смотреть надо на то, собралась ли станция в одно пятно.")
    log("")

    всего_принято = всего_отвергнуто = 0
    for путь in пути:
        кадр = imgio.imread(путь)
        if кадр is None:
            log("%s: не читается" % путь)
            continue
        отвергнутые = []
        детекции = детектор.detect(кадр, area_m2=перцепция._площадь(поза),
                                   rejected=отвергнутые)
        всего_принято += len(детекции)
        всего_отвергнуто += len(отвергнутые)

        фон, порог = (детектор.background(cv2.cvtColor(кадр, cv2.COLOR_BGR2GRAY))
                      if hasattr(детектор, "background") else (0, 0))
        log("%s  (%dx%d, фон %.0f, порог отклонения %.0f)"
            % (os.path.basename(путь), кадр.shape[1], кадр.shape[0], фон, порог))
        if not детекции and not отвергнутые:
            log("    ни одного кандидата: ничего не отличается от фона сильнее порога")
        for детекция in детекции:
            _напечатать(log, "ПРИНЯТО", детекция, поза, калибровка, кадр.shape)
        for детекция, причина, число in отвергнутые:
            _напечатать(log, "отвергнут (%s %.3g)" % (причина, число),
                        детекция, поза, калибровка, кадр.shape)

        if args.out:
            перцепция.draw(кадр, детекции)
            for детекция, причина, число in отвергнутые:
                x1, y1, x2, y2 = детекция.bbox
                cv2.rectangle(кадр, (x1, y1), (x2, y2), ЦВЕТ_ОТВЕРГНУТО, 2)
                cv2.putText(кадр, "%s %.3g" % (_латиницей(причина), число),
                            (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            ЦВЕТ_ОТВЕРГНУТО, 2, cv2.LINE_AA)
            куда = os.path.join(args.out, os.path.basename(путь))
            if not imgio.imwrite(куда, кадр):
                log("    не удалось записать %s" % куда)

    log("")
    log("[снимки] итого принято %d, отвергнуто %d" % (всего_принято, всего_отвергнуто))
    if args.out:
        log("[снимки] размеченные файлы: %s" % args.out)
    return 0


def _адрес_борта():
    """IP борта в сети его точки доступа — для подсказки в логе.

    Маршрут «наружу» здесь не годится: на ноутбуке он уводит на Docker или VPN,
    а на борту в режиме точки доступа его может не быть вовсе. Поэтому ищем адрес
    именно в сети дрона (172.17.49.x у нашего экземпляра), а не «любой свой».
    """
    import socket
    try:
        _, _, адреса = socket.gethostbyname_ex(socket.gethostname())
    except Exception:
        адреса = []
    for адрес in адреса:
        if адрес.startswith("172.17.49."):
            return адрес
    # Не нашли — значит запуск не на борту либо сеть другая. Отдаём известный адрес
    # нашего экземпляра: при подключении к другому борту смотреть шлюз Wi-Fi.
    return "172.17.49.2"


def _показать(viewer, свой, имя, frame):
    """Отдать кадр в обе трансляции. Ни одна не имеет права уронить инструмент:
    смысл всей затеи в том, чтобы картинка была хоть где-то."""
    if свой is not None:
        свой.publish(frame)
    if viewer is None:
        return
    try:
        viewer.imshow(name=имя, frame=frame)
    except Exception:
        pass


_ПРИЧИНЫ_ЛАТИНИЦЕЙ = {
    "площадь м2": "ploshad m2",
    "вытянутость": "vytyanutost",
    "мало пикселей": "malo pikseley",
    "площадь не считается": "net ploshadi",
}


def _латиницей(причина):
    return _ПРИЧИНЫ_ЛАТИНИЦЕЙ.get(причина, причина)


def _напечатать(log, метка, детекция, поза, калибровка, shape):
    """Одна строка про кандидата: где он, какого размера и что о нём думает геометрия."""
    from localization import ground_area, pixel_to_ground

    площадь = ground_area(детекция.corners, поза, калибровка.intrinsics,
                          калибровка.r_mount)
    точка = pixel_to_ground(детекция.cx, детекция.cy, поза, калибровка.intrinsics,
                            калибровка.r_mount)
    x1, y1, x2, y2 = детекция.bbox
    куда = ("вперёд %+.2f влево %+.2f м" % (точка[0], точка[1])
            if точка else "луч мимо земли")
    # Стороны в сантиметрах: рулетка меряет их же, и сравнивать удобнее в см.
    стороны = ""
    if площадь is not None and площадь > 0:
        w, h = abs(x2 - x1), abs(y2 - y1)
        if w > 0 and h > 0:
            масштаб = (площадь / float(w * h)) ** 0.5
            стороны = ", ~%.0fx%.0f см" % (w * масштаб * 100, h * масштаб * 100)
    log("    %-28s %s, площадь %s м2%s"
        % (метка, куда, "?" if площадь is None else "%.4f" % площадь, стороны))


if __name__ == "__main__":
    sys.exit(main())
