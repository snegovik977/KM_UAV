# -*- coding: utf-8 -*-
"""Своя трансляция кадров в браузер. Только стандартная библиотека плюс cv2.

Зачем своя, если в SDK есть ImageViewer. Он работает не всегда и молча:

  - по документации Geoscan это «трансляция numpy-кадров по RTSP через GStreamer»,
    и класс доступен ТОЛЬКО при драйвере камеры gstreamer. Если драйвер другой,
    imshow() не обязан ни ругаться, ни показывать;
  - порт 8889 при RTSP — это, судя по всему, WebRTC-выход медиасервера (у mediamtx
    по умолчанию RTSP 8554, HLS 8888, WebRTC 8889). Тогда браузер и VLC хотят
    РАЗНЫЕ адреса, и «не работает» может означать просто не тот протокол;
  - поток существует, только пока процесс публикует кадры. Открыть адрес заранее
    и увидеть пустоту — нормальное поведение, которое читается как поломка.

Разбираться с этим на площадке — потеря времени, а картинка нужна прямо сейчас:
без неё нельзя настроить детекцию (RUN.md §7.6а). Поэтому здесь простой
MJPEG-сервер: multipart/x-mixed-replace, который умеет показать любой браузер
без плагинов, кодеков и WebRTC.

Только для отладки: MJPEG прожорлив по трафику и не годится для боевой телеметрии.
Бортовой код (`квадрокоптер/`) его не импортирует — это инструмент.
"""
from __future__ import annotations

import threading

try:
    import cv2
except ImportError:                      # pragma: no cover
    cv2 = None

try:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
except ImportError:                      # pragma: no cover — Python < 3.7
    ThreadingHTTPServer = None

СТРАНИЦА = b"""<!doctype html><html><head><meta charset="utf-8">
<title>Kamera drona</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px sans-serif;text-align:center}
img{max-width:100%;height:auto}</style></head>
<body><p>tools/handheld.py &mdash; zelenym prinyato, krasnym otvergnuto</p>
<img src="/stream"></body></html>"""


class MjpegServer(object):
    """Отдаёт последний опубликованный кадр всем подключённым браузерам.

    Кадр хранится в единственном экземпляре: отстающий клиент получит свежий кадр,
    а не очередь старых. Это то же правило, по которому устроена телеметрия
    в protocol/transport.py.
    """

    def __init__(self, port=8080, quality=70, log=None):
        self.port = int(port)
        self.quality = int(quality)
        self._log = log or (lambda text: print(text))
        self._кадр = None                # уже сжатый JPEG: жмём один раз на всех
        self._условие = threading.Condition()
        self._номер = 0
        self._сервер = None
        self._поток = None

    # ------------------------------------------------------------------- запуск

    def start(self):
        """Поднять сервер в фоновом потоке. False, если не вышло — трансляция
        никогда не должна мешать основной работе инструмента."""
        if cv2 is None or ThreadingHTTPServer is None:
            self._log("[mjpeg] нет cv2 или http.server — трансляции не будет")
            return False
        владелец = self

        class Обработчик(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *args):
                pass                     # иначе каждый кадр пишется в консоль

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(СТРАНИЦА)))
                    self.end_headers()
                    self.wfile.write(СТРАНИЦА)
                    return
                if self.path != "/stream":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                последний = -1
                try:
                    while True:
                        кадр, последний = владелец._ждать(последний)
                        if кадр is None:
                            continue
                        self.wfile.write(b"--frame\r\n"
                                         b"Content-Type: image/jpeg\r\n"
                                         b"Content-Length: " +
                                         str(len(кадр)).encode() + b"\r\n\r\n")
                        self.wfile.write(кадр)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass                 # браузер закрыли — это норма, не ошибка

        try:
            self._сервер = ThreadingHTTPServer(("0.0.0.0", self.port), Обработчик)
        except OSError as e:
            self._log("[mjpeg] порт %d занять не удалось: %s. Попробуйте --mjpeg 0 "
                      "(выключить) или другой номер" % (self.port, e))
            return False
        self._сервер.daemon_threads = True
        self._поток = threading.Thread(target=self._сервер.serve_forever,
                                       name="mjpeg", daemon=True)
        self._поток.start()
        return True

    # -------------------------------------------------------------------- кадры

    def publish(self, frame):
        """Положить кадр как последний. Сжатие — здесь, один раз на всех клиентов."""
        if cv2 is None or frame is None or self._сервер is None:
            return
        try:
            ok, буфер = cv2.imencode(".jpg", frame,
                                     [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        except Exception:
            return
        if not ok:
            return
        with self._условие:
            self._кадр = буфер.tobytes()
            self._номер += 1
            self._условие.notify_all()

    def _ждать(self, был):
        """Дождаться кадра новее, чем `был`. Со таймаутом, чтобы закрытие сервера
        не оставляло висеть потоки клиентов."""
        with self._условие:
            if self._номер == был:
                self._условие.wait(timeout=1.0)
            return self._кадр, self._номер

    def close(self):
        if self._сервер is not None:
            try:
                self._сервер.shutdown()
                self._сервер.server_close()
            except Exception:
                pass
            self._сервер = None
