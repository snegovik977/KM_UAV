#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Распределительный хаб — ретранслятор пакетов роя.

Регламент 2.9 требует директорию «распределительный хаб», но нигде её не определяет
(docs/NOTES.md §3). Раздатка организаторов даёт разгадку: HTTP-обмен «один отправитель —
один получатель» не умеет доставить пакет сразу и автомобилю, и визуализатору
(docs/organizer_handouts.md §2.1). Хаб закрывает ровно эту дыру: принимает пакеты
от кого угодно и раздаёт их всем остальным по опросу.

Живёт на компьютере оператора. Дрону не нужен ни Flask, ни requests — он общается
с хабом через stdlib (protocol/transport.py).

    POST /msg              принять пакет (тело — JSON пакета протокола)
    GET  /msg?since=N      отдать пакеты начиная с индекса N
                           &exclude_src=drone — не возвращать свои же
    GET  /state            снимок: последняя телеметрия + реестр станций
    GET  /health           жив ли хаб (проверка связности перед попыткой)

Сервер поднимается на Flask, если он есть, иначе на http.server из стандартной
библиотеки — маршруты и ответы одинаковые. Это не перестраховка: по регламенту 1.4
личные компьютеры на площадке запрещены, а что установлено на выданной машине,
заранее неизвестно.

Запуск:
    python hub.py                       # 0.0.0.0:5001
    python hub.py --port 5001 --log records/hub_log.jsonl
"""
from __future__ import annotations

import argparse
import io
import json
import os
import queue
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from protocol.legacy import MapAnchor, to_legacy  # noqa: E402
from protocol.messages import ProtocolError, STATUSES, parse  # noqa: E402

DEFAULT_PORT = 5001          # порт из примеров организаторов


class LegacyForwarder(object):
    """Мост наружу: дублирует наши пакеты в текстовом формате примера организаторов.

    Нужен ровно на один случай: если на площадке визуализатор пишет смежный трек
    по присланному организаторами шаблону (source/2_step/MapExample.py), он ждёт
    POST /fly с телом "x,y" в пикселях и POST /eyecar с "Good"/"Broken", а нашего
    JSON не понимает. Восемь баллов подзадачи 2.3.2 требуют работающего визуализатора,
    и упереться в несовпадение форматов на площадке дороже, чем держать этот мост.

    По умолчанию ВЫКЛЮЧЕН: хаб обязан оставаться тупым ретранслятором, а лишний
    исходящий HTTP на несуществующий адрес — это задержки в потоке приёма.
    Отправка идёт в отдельном потоке и ошибки только считает.
    """

    def __init__(self, url, anchor, log=None):
        self.url = url.rstrip("/")
        self.anchor = anchor
        self._log = log or (lambda text: print(text))
        self.sent = 0
        self.errors = 0
        self._queue = queue.Queue(maxsize=200)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="legacy-forward",
                                        daemon=True)
        self._thread.start()

    def offer(self, msg):
        """Поставить пакет в очередь. Не блокирует: переполненную очередь роняем —
        мост вторичен, приём пакетов важнее."""
        for маршрут, тело in to_legacy(msg, self.anchor):
            try:
                self._queue.put_nowait((маршрут, тело))
            except queue.Full:
                self.errors += 1

    def _loop(self):
        from urllib import error, request

        while not self._stop.is_set():
            try:
                маршрут, тело = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            запрос = request.Request(self.url + маршрут,
                                     data=тело.encode("utf-8"), method="POST")
            запрос.add_header("Content-Type", "text/plain; charset=utf-8")
            try:
                request.urlopen(запрос, timeout=1.0).read()
                self.sent += 1
            except (error.URLError, OSError) as e:
                self.errors += 1
                if self.errors % 20 == 1:
                    self._log("[мост] %s не принял (%d-я ошибка): %s"
                              % (self.url, self.errors, e))

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)

    def summary(self):
        return "[мост] на %s отправлено %d, ошибок %d" % (self.url, self.sent, self.errors)


class HubState:
    """Журнал пакетов и производная от него карта. Вся логика хаба — здесь,
    чтобы Flask и http.server оставались тонкими обёртками."""

    def __init__(self, log_path=None, forwarder=None):
        self.forwarder = forwarder      # мост в формат организаторов, обычно None
        self._lock = threading.Lock()
        self._log = []                  # все принятые пакеты по порядку
        self._stations = {}             # id -> {id, x, y, status, conf, src, ts}
        self._telemetry = None          # последний telemetry.data + ts
        self._recon_done = None         # {"src": ..., "count": ..., "ts": ...}
        self._rejected = 0
        self._started = time.time()
        self._log_path = log_path
        self._log_file = None
        if log_path:
            try:
                папка = os.path.dirname(log_path)
                if папка and not os.path.isdir(папка):
                    os.makedirs(папка)
                self._log_file = io.open(log_path, "a", encoding="utf-8")
            except Exception as e:
                print("[hub] не удалось открыть журнал %s: %s" % (log_path, e))

    # ------------------------------------------------------------------ приём

    def add(self, raw):
        """Принять пакет. Возвращает (код ответа, тело ответа)."""
        try:
            msg = parse(raw)
        except ProtocolError as e:
            with self._lock:
                self._rejected += 1
            return 400, {"ok": False, "error": str(e)}

        with self._lock:
            self._log.append(msg)
            self._apply(msg)
            index = len(self._log)
        self._write_log(msg)
        if self.forwarder is not None:
            # Вне замка и без ожидания: мост вторичен, приём пакетов важнее.
            self.forwarder.offer(msg)
        return 200, {"ok": True, "index": index}

    def _apply(self, msg):
        """Обновить производную карту. Вызывается под замком."""
        data = msg["data"]
        if msg["type"] == "telemetry":
            self._telemetry = dict(data, ts=msg["ts"], src=msg["src"])
        elif msg["type"] == "station_new":
            # Слияние «дрон и автомобиль увидели одну станцию» — забота потребителя
            # (по координатам, порог 40 см). Хаб ничего не склеивает: он обязан
            # оставаться тупым ретранслятором, иначе его логика разойдётся с логикой
            # автомобиля и визуализатора.
            self._stations[data["id"]] = {
                "id": data["id"], "x": data["x"], "y": data["y"],
                "status": data["status"], "conf": data["conf"],
                "src": msg["src"], "ts": msg["ts"],
            }
        elif msg["type"] == "status_update":
            station = self._stations.get(data["id"])
            if station is None:
                # status_update раньше station_new: пакет мог потеряться или прийти
                # не по порядку. Заводим запись без координат — лучше, чем потерять статус.
                self._stations[data["id"]] = {
                    "id": data["id"], "x": None, "y": None,
                    "status": data["status"], "conf": data["conf"],
                    "src": msg["src"], "ts": msg["ts"],
                }
            else:
                station.update(status=data["status"], conf=data["conf"], ts=msg["ts"])
        elif msg["type"] == "recon_done":
            self._recon_done = {"src": msg["src"], "count": data["count"], "ts": msg["ts"]}

    def _write_log(self, msg):
        """Дозапись в jsonl: разбор попытки вечером не требует полигона (регламент 1.3)."""
        if self._log_file is None:
            return
        try:
            self._log_file.write(json.dumps(msg, ensure_ascii=False) + "\n")
            self._log_file.flush()
        except Exception:
            pass

    # ------------------------------------------------------------------ выдача

    def since(self, index, exclude_src=None):
        with self._lock:
            total = len(self._log)
            index = max(0, min(int(index), total))
            msgs = self._log[index:]
        if exclude_src:
            msgs = [m for m in msgs if m["src"] != exclude_src]
        # next всегда полная длина журнала, даже если что-то отфильтровано, —
        # иначе курсор клиента залипнет на отфильтрованном пакете.
        return {"next": total, "msgs": msgs}

    def snapshot(self):
        with self._lock:
            stations = [dict(s) for s in self._stations.values()]
            by_status = {s: 0 for s in STATUSES}
            for station in stations:
                if station["status"] in by_status:
                    by_status[station["status"]] += 1
            return {
                "telemetry": dict(self._telemetry) if self._telemetry else None,
                "stations": sorted(stations, key=lambda s: s["id"]),
                "count": len(stations),
                "by_status": by_status,
                "recon_done": dict(self._recon_done) if self._recon_done else None,
                "messages": len(self._log),
            }

    def health(self):
        with self._lock:
            return {
                "ok": True,
                "uptime": round(time.time() - self._started, 1),
                "messages": len(self._log),
                "rejected": self._rejected,
                "stations": len(self._stations),
                "log": self._log_path,
            }

    def close(self):
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass


# ------------------------------------------------------------------ два сервера

def run_flask(state, host, port):
    from flask import Flask, request, jsonify

    app = Flask(__name__)

    @app.route("/msg", methods=["POST"])
    def post_msg():
        code, body = state.add(request.get_data())
        return jsonify(body), code

    @app.route("/msg", methods=["GET"])
    def get_msg():
        return jsonify(state.since(request.args.get("since", 0),
                                   request.args.get("exclude_src")))

    @app.route("/state", methods=["GET"])
    def get_state():
        return jsonify(state.snapshot())

    @app.route("/health", methods=["GET"])
    def get_health():
        return jsonify(state.health())

    print("[hub] Flask на http://%s:%d" % (host, port))
    # use_reloader=True перезапускает процесс и убивает фоновые потоки
    # (docs/organizer_handouts.md §2.2) — на площадке это выглядит как «хаб иногда
    # теряет пакеты».
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)


def make_stdlib_server(state, host, port):
    """Собрать сервер, но не запускать. Отдельно от run_stdlib, чтобы тесты могли
    поднять хаб на свободном порту (port=0) и узнать, какой достался."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlparse

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _reply(self, code, body):
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/msg":
                self._reply(404, {"ok": False, "error": "нет такого маршрута"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
            except Exception as e:
                self._reply(400, {"ok": False, "error": "тело не прочитано: %s" % e})
                return
            code, body = state.add(raw)
            self._reply(code, body)

        def do_GET(self):
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == "/msg":
                since = query.get("since", ["0"])[0]
                exclude = query.get("exclude_src", [None])[0]
                try:
                    since_index = int(since)
                except ValueError:
                    since_index = 0
                self._reply(200, state.since(since_index, exclude))
            elif parsed.path == "/state":
                self._reply(200, state.snapshot())
            elif parsed.path == "/health":
                self._reply(200, state.health())
            else:
                self._reply(404, {"ok": False, "error": "нет такого маршрута"})

        def log_message(self, fmt, *args):
            pass  # опрос идёт 5 раз в секунду от каждого клиента — консоль не нужна

    return ThreadingHTTPServer((host, port), Handler)


def run_stdlib(state, host, port):
    server = make_stdlib_server(state, host, port)
    print("[hub] http.server на http://%s:%d (Flask не найден — работаем на stdlib)"
          % (host, port))
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Распределительный хаб — ретранслятор пакетов")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    # Журнал лежит рядом с записями попыток (records/), а не в корне: разбор вечером
    # берёт всё из одной папки, и корень репозитория остаётся читаемым.
    parser.add_argument("--log", default=os.path.join("records", "hub_log.jsonl"),
                        help="журнал принятых пакетов (пусто — не вести)")
    parser.add_argument("--server", choices=("auto", "flask", "stdlib"), default="auto")
    parser.add_argument("--forward", default=None, metavar="URL",
                        help="дублировать пакеты в текстовом формате организаторов "
                             "(POST /fly, /eyecar) на чужой визуализатор. "
                             "По умолчанию выключено")
    parser.add_argument("--config", default=None,
                        help="config.yaml — из него берётся привязка карты для --forward")
    args = parser.parse_args()

    forwarder = None
    if args.forward:
        # Привязка «метры -> пиксели карты» нужна только мосту: наш собственный
        # визуализатор берёт её сам и в метрах не теряется.
        cfg = {}
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            os.pardir, "квадрокоптер"))
            import config as config_module
            cfg = config_module.load(args.config)
        except Exception as e:
            print("[мост] config.yaml не прочитан (%s) — привязка карты по умолчанию" % e)
        forwarder = LegacyForwarder(args.forward, MapAnchor.from_config(cfg))
        print("[мост] дублирую станции в текстовом формате на %s" % args.forward)

    state = HubState(log_path=args.log or None, forwarder=forwarder)

    use_flask = args.server == "flask"
    if args.server == "auto":
        try:
            import flask  # noqa: F401
            use_flask = True
        except ImportError:
            use_flask = False

    print("[hub] журнал: %s" % (args.log or "не ведётся"))
    print("[hub] проверка связности:  curl http://<адрес хаба>:%d/health" % args.port)
    try:
        if use_flask:
            run_flask(state, args.host, args.port)
        else:
            run_stdlib(state, args.host, args.port)
    except KeyboardInterrupt:
        print("\n[hub] остановлен оператором")
    finally:
        snapshot = state.snapshot()
        print("[hub] принято пакетов: %d, станций в карте: %d"
              % (snapshot["messages"], snapshot["count"]))
        if forwarder is not None:
            print(forwarder.summary())
            forwarder.stop()
        state.close()


if __name__ == "__main__":
    main()
