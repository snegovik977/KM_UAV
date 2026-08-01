# -*- coding: utf-8 -*-
"""Транспорт роя: HTTP через распределительный хаб. Только стандартная библиотека.

Почему HTTP, а не UDP-broadcast из первой редакции плана: раздатка организаторов
(docs/organizer_handouts.md §2) прямо показывает Flask + requests как ожидаемый способ
связи, а необходимость раздавать пакет сразу двум получателям — автомобилю
и визуализатору — объясняет «распределительный хаб» из регламента 2.9. Совпадение
с материалами организаторов проще защищать на разборе решения (регламент 2.5/2.7).

Почему urllib, а не requests: на борту нет ни requests, ни flask, а чтобы их поставить,
дрон надо перевести в режим клиента Wi-Fi — после чего он перестаёт быть на 172.17.49.2
(docs/organizer_handouts.md §2.4). Отправитель на stdlib снимает эту зависимость
целиком: Flask нужен только хабу, который живёт на компьютере оператора.

Три свойства, ради которых это не просто обёртка над urlopen:

1. send() НЕ блокирует. Отправка живёт в своём потоке, вызывающий кладёт пакет
   в очередь и идёт дальше. Поток перцепции не имеет права ждать сеть.
2. Телеметрия и события разделены. Устаревшая телеметрия бесполезна — в её очереди
   хранится только последний пакет. А station_new терять нельзя: «количество
   обнаруженных станций соответствует реальному» — условие всех 8 баллов подзадачи
   2.3.2, поэтому события идут отдельной очередью и телеметрия их не вытесняет.
3. Недоступный хаб не роняет полёт. Ошибки сети считаются и печатаются с прореживанием
   (docs/lessons_from_archipelago.md §3), наружу не всплывают.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections import deque

from .messages import ProtocolError, SeqDedup, dumps, parse

# Пакеты этого типа устаревают мгновенно: важен только последний.
_VOLATILE = ("telemetry",)


class HttpTransport:
    """Клиент хаба: отправка в фоне + опрос входящих.

        t = HttpTransport("http://172.17.49.5:5001", src="drone")
        t.start()
        t.send(factory.telemetry(...))      # не блокирует
        for msg in t.recv():                # разобранные, без повторов
            ...
        t.stop()

    recv() — одноразовое чтение: пакеты забираются из буфера. Потребитель должен быть
    один (у нас это полётный поток), иначе сообщения размажутся между читателями.
    """

    def __init__(self, hub_url, src, poll=True, poll_interval=0.2, timeout=0.5,
                 retries=2, queue_size=500, exclude_self=True, log=None):
        self.hub_url = hub_url.rstrip("/")
        self.src = src
        self.poll = poll
        self.poll_interval = float(poll_interval)
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.exclude_self = bool(exclude_self)
        self._log = log or (lambda text: print(text))

        self._events = deque(maxlen=int(queue_size))   # важные пакеты, порядок сохраняется
        self._events_lock = threading.Lock()
        self._latest_volatile = {}                     # type -> последний пакет
        self._wake = threading.Event()                 # будит отправителя без busy-loop

        self._inbox = deque()
        self._inbox_lock = threading.Lock()
        self._dedup = SeqDedup()
        self._since = 0

        self._stop = threading.Event()
        self._threads = []

        self.stats = {"sent": 0, "dropped": 0, "send_errors": 0,
                      "received": 0, "poll_errors": 0, "rejected": 0}

    # ------------------------------------------------------------ жизненный цикл

    def start(self):
        if self._threads:
            return self
        self._stop.clear()
        self._threads.append(threading.Thread(target=self._sender_loop,
                                              name="hub-send", daemon=True))
        if self.poll:
            self._threads.append(threading.Thread(target=self._poll_loop,
                                                  name="hub-poll", daemon=True))
        for t in self._threads:
            t.start()
        return self

    def stop(self, timeout=1.5):
        self._stop.set()
        self._wake.set()
        for t in self._threads:
            t.join(timeout=timeout)
        self._threads = []

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ---------------------------------------------------------------- отправка

    def send(self, msg):
        """Поставить пакет в очередь. Возврат мгновенный, сеть — в фоне."""
        if msg["type"] in _VOLATILE:
            # Копится только последний: догонять дрон устаревшей телеметрией бессмысленно.
            self._latest_volatile[msg["type"]] = msg
        else:
            with self._events_lock:
                if len(self._events) == self._events.maxlen:
                    # deque с maxlen вытеснит сам, но нам нужен счётчик — потери
                    # событий должны быть видны в логе, а не молча съедены.
                    self.stats["dropped"] += 1
                self._events.append(msg)
        self._wake.set()

    def _take_next(self):
        """Событие важнее телеметрии: сначала очередь событий, потом свежая телеметрия."""
        with self._events_lock:
            if self._events:
                return self._events.popleft()
        for msg_type in _VOLATILE:
            msg = self._latest_volatile.pop(msg_type, None)
            if msg is not None:
                return msg
        return None

    def _sender_loop(self):
        while not self._stop.is_set():
            msg = self._take_next()
            if msg is None:
                # Ждём события, а не крутим пустой цикл: CPU нужен инференсу.
                self._wake.wait(timeout=0.2)
                self._wake.clear()
                continue
            self._post(msg)

    def _post(self, msg):
        payload = dumps(msg)
        for attempt in range(self.retries + 1):
            if self._stop.is_set():
                return False
            try:
                request = urllib.request.Request(
                    self.hub_url + "/msg", data=payload,
                    headers={"Content-Type": "application/json"}, method="POST")
                resp = urllib.request.urlopen(request, timeout=self.timeout)
                resp.read()
                resp.close()
                self.stats["sent"] += 1
                return True
            except Exception as e:
                if attempt >= self.retries:
                    self.stats["send_errors"] += 1
                    # Прореживание: при упавшем хабе иначе зальём консоль и потеряем
                    # настоящие ошибки полёта.
                    if self.stats["send_errors"] % 30 == 1:
                        self._log("[hub] отправка %s не удалась (%d-я ошибка): %s: %s"
                                  % (msg["type"], self.stats["send_errors"],
                                     type(e).__name__, e))
                else:
                    time.sleep(0.05)
        return False

    # ------------------------------------------------------------------ приём

    def _poll_loop(self):
        while not self._stop.is_set():
            self._poll_once()
            self._stop.wait(self.poll_interval)

    def _poll_once(self):
        url = "%s/msg?since=%d" % (self.hub_url, self._since)
        if self.exclude_self:
            # Свои же пакеты обратно не забираем: хаб раздаёт всем подряд.
            url += "&exclude_src=" + self.src
        try:
            resp = urllib.request.urlopen(url, timeout=self.timeout)
            body = resp.read()
            resp.close()
            payload = json.loads(body.decode("utf-8"))
        except Exception as e:
            self.stats["poll_errors"] += 1
            if self.stats["poll_errors"] % 30 == 1:
                self._log("[hub] опрос не удался (%d-я ошибка): %s: %s"
                          % (self.stats["poll_errors"], type(e).__name__, e))
            return

        self._since = int(payload.get("next", self._since))
        fresh = []
        for raw in payload.get("msgs", []):
            try:
                msg = parse(raw)
            except ProtocolError as e:
                # Чужой кривой пакет не должен ронять приём: считаем и живём дальше.
                self.stats["rejected"] += 1
                if self.stats["rejected"] % 30 == 1:
                    self._log("[hub] отвергнут пакет (%d-й): %s" % (self.stats["rejected"], e))
                continue
            if self._dedup.is_new(msg):
                fresh.append(msg)
        if fresh:
            with self._inbox_lock:
                self._inbox.extend(fresh)
            self.stats["received"] += len(fresh)

    def recv(self):
        """Забрать накопленные входящие пакеты (и очистить буфер)."""
        with self._inbox_lock:
            out = list(self._inbox)
            self._inbox.clear()
        return out

    # ----------------------------------------------------------- диагностика

    def health(self):
        """Жив ли хаб. Синхронный вызов — только для проверок перед попыткой."""
        try:
            resp = urllib.request.urlopen(self.hub_url + "/health", timeout=self.timeout)
            body = resp.read()
            resp.close()
            return json.loads(body.decode("utf-8"))
        except Exception as e:
            return {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}

    def summary(self):
        """Строка для HUD и финального лога попытки."""
        s = self.stats
        return ("hub: отправлено %d, потеряно %d, ошибок отправки %d | "
                "принято %d, ошибок опроса %d, отвергнуто %d"
                % (s["sent"], s["dropped"], s["send_errors"],
                   s["received"], s["poll_errors"], s["rejected"]))
