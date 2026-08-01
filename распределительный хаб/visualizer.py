#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Визуализатор: карта полигона со станциями и следом дрона.

Без него подзадача 2.3.2 не сдаётся вообще: 8 баллов даются за то, что «визуализатор
в режиме реального времени отображает на карте обнаруженные станции» (регламент 2.3.2),
а 2.3.3 требует ещё и статусы.

За основу взят рабочий пример организаторов (source/2_step/MapExample.py): окно OpenCV
поверх картинки полигона, станции — кружки в ПИКСЕЛЬНЫХ координатах карты. Два отличия,
оба существенные:

1. ОН НЕ ПОДНИМАЕТ СВОЙ СЕРВЕР. Пример слушает POST на порту 5001 — ровно том, на котором
   живёт наш хаб, и на одной машине они бы столкнулись. Здесь визуализатор работает
   КЛИЕНТОМ хаба (GET /msg?since=N) через тот же protocol/transport.py, что дрон
   и автомобиль. Порядок запуска при этом перестаёт иметь значение.
2. СТАНЦИИ ВЕДУТСЯ ПО id, а не по порядку прихода сообщений. Пример красит станции
   счётчиком stations_counter, и один потерянный пакет сдвигает все последующие статусы
   на одну станцию. У нас id приходит в пакете, повторный station_new уточняет запись,
   а не заводит новую.

Координаты в пакетах — метры нашей СК; перевод в пиксели карты делает
protocol/legacy.py -> MapAnchor по секции map из config.yaml.

Запуск:
    python3 visualizer.py                            # хаб из config.yaml
    python3 visualizer.py --hub http://127.0.0.1:5001
    python3 visualizer.py --map assets/map.jpg --headless   # без окна, только лог
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_ЗДЕСЬ = os.path.dirname(os.path.abspath(__file__))
_КОРЕНЬ = os.path.dirname(_ЗДЕСЬ)
for _путь in (_ЗДЕСЬ, os.path.join(_КОРЕНЬ, "квадрокоптер")):
    if os.path.isdir(_путь) and _путь not in sys.path:
        sys.path.insert(0, _путь)

from protocol import HttpTransport, MapAnchor  # noqa: E402

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

# Цвета статусов, BGR. Те же, что рисует дрон на отладочной трансляции
# (квадрокоптер/perception/pipeline.py): оператор смотрит на два экрана сразу.
ЦВЕТ_СТАТУСА = {
    "ok": (80, 220, 80),        # зелёный — исправна
    "dust": (0, 165, 255),      # оранжевый — покрыта пылью
    "broken": (60, 60, 235),    # красный — неисправна
}
ЦВЕТ_ДРОНА = (255, 200, 0)
ЦВЕТ_СЛЕДА = (200, 160, 60)

РАДИУС_СТАНЦИИ = 12
ДЛИНА_СЛЕДА = 400               # точек следа дрона на карте


class MapView(object):
    """Состояние карты: станции по id, след дрона, счётчики. Без OpenCV."""

    def __init__(self, anchor, log=None):
        self.anchor = anchor
        self._log = log or (lambda text: print(text))
        self.stations = {}          # id -> {"x", "y", "status", "conf", "src"}
        self.trail = []             # [(x, y)] в метрах
        self.telemetry = None
        self.recon_done = None
        self.messages = 0
        self.changed = True

    def apply(self, msg):
        """Принять пакет протокола. Возвращает True, если карту надо перерисовать."""
        self.messages += 1
        тип = msg["type"]
        data = msg["data"]

        if тип == "station_new":
            прежняя = self.stations.get(data["id"])
            self.stations[data["id"]] = {
                "x": data["x"], "y": data["y"], "status": data["status"],
                "conf": data["conf"], "src": msg["src"]}
            if прежняя is None:
                self._log("[карта] станция %s: x=%+.2f y=%+.2f %s"
                          % (data["id"], data["x"], data["y"], data["status"]))
            self.changed = True

        elif тип == "status_update":
            станция = self.stations.setdefault(
                data["id"], {"x": None, "y": None, "src": msg["src"]})
            станция["status"] = data["status"]
            станция["conf"] = data["conf"]
            self._log("[карта] станция %s -> %s" % (data["id"], data["status"]))
            self.changed = True

        elif тип == "telemetry":
            self.telemetry = data
            self.trail.append((data["x"], data["y"]))
            if len(self.trail) > ДЛИНА_СЛЕДА:
                del self.trail[0]
            self.changed = True

        elif тип == "recon_done":
            self.recon_done = data["count"]
            self._log("[карта] разведка завершена: станций %d" % data["count"])
            self.changed = True

        return self.changed

    def counters(self):
        по_статусам = {}
        for станция in self.stations.values():
            статус = станция.get("status") or "?"
            по_статусам[статус] = по_статусам.get(статус, 0) + 1
        return по_статусам

    def summary(self):
        части = ["станций: %d" % len(self.stations)]
        for статус, сколько in sorted(self.counters().items()):
            части.append("%s %d" % (статус, сколько))
        if self.recon_done is not None:
            части.append("разведка завершена (%d)" % self.recon_done)
        return ", ".join(части)


def загрузить_карту(path, width=1200, height=900):
    """Картинка полигона или пустой холст, если её нет.

    Отсутствие файла не должно ронять визуализатор: на площадке фотография карты может
    появиться позже, а станции надо видеть уже сейчас — хотя бы на сетке.
    """
    if cv2 is not None and path and os.path.exists(path):
        карта = cv2.imread(path)
        if карта is not None:
            return карта
        print("[карта] %s не читается — рисую на сетке" % path)
    elif path:
        print("[карта] файла %s нет — рисую на сетке" % path)
    if np is None:
        return None
    холст = np.full((height, width, 3), 40, dtype=np.uint8)
    if cv2 is not None:
        for x in range(0, width, 50):
            cv2.line(холст, (x, 0), (x, height), (60, 60, 60), 1)
        for y in range(0, height, 50):
            cv2.line(холст, (0, y), (width, y), (60, 60, 60), 1)
    return холст


def нарисовать(основа, view):
    """Кадр карты: след дрона, станции, сам дрон, счётчики."""
    if cv2 is None or основа is None:
        return None
    кадр = основа.copy()
    anchor = view.anchor

    # След дрона — по нему на разборе видно, где он реально прошёл и что мог не снять.
    точки = [anchor.to_px(x, y) for x, y in view.trail]
    for первая, вторая in zip(точки, точки[1:]):
        cv2.line(кадр, первая, вторая, ЦВЕТ_СЛЕДА, 1, cv2.LINE_AA)

    for имя, станция in sorted(view.stations.items()):
        if станция.get("x") is None:
            continue                 # статус пришёл раньше координат
        u, v = anchor.to_px(станция["x"], станция["y"])
        цвет = ЦВЕТ_СТАТУСА.get(станция.get("status"), (200, 200, 200))
        cv2.circle(кадр, (u, v), РАДИУС_СТАНЦИИ, цвет, -1)
        cv2.circle(кадр, (u, v), РАДИУС_СТАНЦИИ, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(кадр, имя, (u + РАДИУС_СТАНЦИИ + 4, v + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if view.telemetry is not None:
        u, v = anchor.to_px(view.telemetry["x"], view.telemetry["y"])
        cv2.circle(кадр, (u, v), 7, ЦВЕТ_ДРОНА, -1)
        cv2.circle(кадр, (u, v), 7, (0, 0, 0), 1, cv2.LINE_AA)

    подпись = view.summary()
    if view.telemetry is not None:
        подпись += "   дрон: %s z=%.2f batt=%.2fB" % (
            view.telemetry["state"], view.telemetry["z"], view.telemetry["batt"])
    cv2.rectangle(кадр, (0, 0), (кадр.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(кадр, подпись, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return кадр


def main():
    parser = argparse.ArgumentParser(description="Визуализатор карты станций")
    parser.add_argument("--hub", default=None, help="адрес хаба")
    parser.add_argument("--config", default=None, help="путь к config.yaml")
    parser.add_argument("--map", default=None, help="картинка карты полигона")
    parser.add_argument("--headless", action="store_true",
                        help="без окна: только лог принятых станций")
    parser.add_argument("--save", default=None,
                        help="куда сохранить итоговую карту при выходе (png)")
    args = parser.parse_args()

    import config as config_module
    cfg = config_module.load(args.config or os.path.join(
        _КОРЕНЬ, "квадрокоптер", "config.yaml"))
    url = args.hub or cfg["hub"]["url"]

    путь_карты = args.map or str(cfg.get("map", {}).get("image", ""))
    if путь_карты and not os.path.isabs(путь_карты):
        путь_карты = os.path.join(_ЗДЕСЬ, путь_карты)
    основа = загрузить_карту(путь_карты)
    if основа is None:
        print("[карта] нет ни OpenCV, ни numpy — работаю только в текстовом режиме")
        args.headless = True
        высота = ширина = None
    else:
        высота, ширина = основа.shape[:2]

    anchor = MapAnchor.from_config(cfg, width=ширина, height=высота)
    view = MapView(anchor)
    print("[карта] хаб %s, привязка: %.1f px/м, ноль в (%.0f, %.0f), поворот %.0f°"
          % (url, anchor.px_per_m, anchor.origin_px[0], anchor.origin_px[1],
             anchor.yaw_deg))

    transport = HttpTransport(url, src="visualizer",
                              poll_interval=float(cfg["hub"]["poll_interval"]),
                              timeout=float(cfg["hub"]["timeout"]),
                              retries=int(cfg["hub"]["retries"])).start()
    здоровье = transport.health()
    if not здоровье.get("ok"):
        print("[карта] хаб не отвечает (%s) — жду, он может подняться позже"
              % здоровье.get("error"))

    if not args.headless:
        cv2.namedWindow("Карта станций", cv2.WINDOW_NORMAL)
    кадр = None
    try:
        while True:
            for msg in transport.recv():
                view.apply(msg)

            if args.headless:
                time.sleep(0.2)
                continue

            if view.changed:
                кадр = нарисовать(основа, view)
                view.changed = False
            if кадр is not None:
                cv2.imshow("Карта станций", кадр)
            # waitKey и есть пауза цикла: отдельный sleep только добавил бы задержку.
            if (cv2.waitKey(100) & 0xFF) == 27:      # Esc — как в примере организаторов
                break
    except KeyboardInterrupt:
        print("\n[карта] остановлено оператором")
    finally:
        if args.save and кадр is not None and cv2 is not None:
            cv2.imwrite(args.save, кадр)
            print("[карта] сохранено: %s" % args.save)
        if not args.headless and cv2 is not None:
            cv2.destroyAllWindows()
        transport.stop()
        print("[карта] итог: %s (пакетов принято %d)" % (view.summary(), view.messages))
        # Итоговые координаты — то, что сверяют рулеткой после попытки. Дрон дошлёт
        # уточнённую медиану, и последнее значение отличается от того, что печаталось
        # при первом появлении станции.
        for имя, станция in sorted(view.stations.items()):
            if станция.get("x") is None:
                print("  %s: координат не было, статус %s"
                      % (имя, станция.get("status")))
                continue
            print("  %s: x=%+.2f y=%+.2f %s"
                  % (имя, станция["x"], станция["y"], станция.get("status")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
