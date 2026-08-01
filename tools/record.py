# -*- coding: utf-8 -*-
"""Запись попытки: видео + телеметрия с общими таймстемпами.

Прямое следствие механики лимита времени (регламент 1.3): время утекает, даже когда
полигон простаивает, поэтому каждая попытка обязана привозить материал, по которому
вечером можно работать без полигона. Одна попытка = вход для разметки датасета
(пункт 2 плана), для tools/replay.py и для разбора «почему дрон повёл себя так».

Файлы кладутся парой с одинаковым именем в папку `mission.record_dir` (по умолчанию
`records/` — и на ноутбуке, и на борту; в корне репозитория они тонули):

    records/flight_20260731_142530.mp4      кадры, как их видела перцепция
    records/flight_20260731_142530.jsonl    снимки состояния миссии, по строке на запись

Запись никогда не роняет полёт: не открылся VideoWriter — летим без видео, не пишется
кадр — считаем ошибку и продолжаем.

Просмотр записи без полигона:
    python tools/record.py records/flight_20260731_142530.jsonl
"""
from __future__ import annotations

import io
import json
import os
import time

try:
    import cv2
except ImportError:
    cv2 = None


class RecordWriter(object):
    """Пишет кадры и телеметрию одной попытки."""

    def __init__(self, out_dir=".", fps=15, prefix="flight", log=None):
        self.out_dir = out_dir
        self.fps = float(fps)
        self._log = log or (lambda text: print(text))
        self.started = time.time()
        self.stamp = time.strftime("%Y%m%d_%H%M%S")
        self.base = os.path.join(out_dir, "%s_%s" % (prefix, self.stamp))
        self.video_path = self.base + ".mp4"
        self.telemetry_path = self.base + ".jsonl"

        self.frames = 0
        self.records = 0
        self.errors = 0
        self._writer = None
        self._telemetry = None

        try:
            if out_dir and not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            self._telemetry = io.open(self.telemetry_path, "w", encoding="utf-8")
            self._log("[запись] телеметрия: %s" % self.telemetry_path)
        except Exception as e:
            self._log("[запись] телеметрия недоступна: %s: %s" % (type(e).__name__, e))

    # ------------------------------------------------------------------------ видео

    def frame(self, image):
        """Записать кадр. VideoWriter открывается по первому кадру — до него
        неизвестно разрешение (а оно у камеры борта до сих пор не подтверждено)."""
        if image is None:
            return False
        if self._writer is None:
            if not self._open_writer(image):
                return False
        try:
            self._writer.write(image)
            self.frames += 1
            return True
        except Exception as e:
            self.errors += 1
            if self.errors % 30 == 1:
                self._log("[запись] кадр не записан (%d-я ошибка): %s: %s"
                          % (self.errors, type(e).__name__, e))
            return False

    def _open_writer(self, image):
        if cv2 is None:
            self._log("[запись] нет OpenCV — пишем только телеметрию")
            self._writer = False
            return False
        if self._writer is False:
            return False
        try:
            высота, ширина = image.shape[:2]
            writer = cv2.VideoWriter(self.video_path,
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     self.fps, (ширина, высота))
            if not writer.isOpened():
                self._log("[запись] VideoWriter не открылся — летим без видео")
                self._writer = False
                return False
            self._writer = writer
            self._log("[запись] видео: %s (%dx%d @ %.0f fps)"
                      % (self.video_path, ширина, высота, self.fps))
            return True
        except Exception as e:
            self._log("[запись] видео недоступно: %s: %s" % (type(e).__name__, e))
            self._writer = False
            return False

    # ------------------------------------------------------------------ телеметрия

    def telemetry(self, snapshot):
        """Снимок MissionState -> строка jsonl. Ключ t — секунды от начала записи,
        общие с номером кадра: по ним запись и телеметрия совмещаются в replay."""
        if self._telemetry is None:
            return False
        запись = dict(snapshot)
        запись["t"] = round(time.time() - self.started, 3)
        запись["frame"] = self.frames
        try:
            self._telemetry.write(json.dumps(запись, ensure_ascii=False, default=str) + "\n")
            self.records += 1
            return True
        except Exception as e:
            self.errors += 1
            if self.errors % 30 == 1:
                self._log("[запись] телеметрия не записана: %s: %s" % (type(e).__name__, e))
            return False

    # ------------------------------------------------------------------- завершение

    def close(self):
        if self._writer not in (None, False):
            try:
                self._writer.release()
            except Exception:
                pass
        if self._telemetry is not None:
            try:
                self._telemetry.close()
            except Exception:
                pass
        self._log("[запись] сохранено: %d кадров, %d записей телеметрии, %d ошибок"
                  % (self.frames, self.records, self.errors))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------- просмотр

def описать(path):
    """Краткая сводка по jsonl-логу попытки: что произошло и когда."""
    записи = []
    with io.open(path, "r", encoding="utf-8") as f:
        for строка in f:
            строка = строка.strip()
            if строка:
                try:
                    записи.append(json.loads(строка))
                except ValueError:
                    continue
    if not записи:
        print("Пустой лог: %s" % path)
        return

    print("Лог: %s" % path)
    print("Записей: %d, длительность: %.1f с" % (len(записи), записи[-1].get("t", 0.0)))

    предыдущее = None
    print("\nСмена состояний:")
    for запись in записи:
        состояние = запись.get("state")
        if состояние != предыдущее:
            print("  %7.1f с  %s" % (запись.get("t", 0.0), состояние))
            предыдущее = состояние

    высоты = [з["position"][2] for з in записи if з.get("position")]
    заряды = [з["battery"] for з in записи if з.get("battery")]
    if высоты:
        print("\nВысота: мин %.2f м, макс %.2f м" % (min(высоты), max(высоты)))
    if заряды:
        print("Заряд: %.2f -> %.2f В" % (заряды[0], заряды[-1]))
    причина = [з.get("abort_reason") for з in записи if з.get("abort_reason")]
    if причина:
        print("Аварийное завершение: %s" % причина[-1])


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Просмотр записи попытки")
    parser.add_argument("log", help="файл flight_*.jsonl")
    описать(parser.parse_args().log)


if __name__ == "__main__":
    main()
