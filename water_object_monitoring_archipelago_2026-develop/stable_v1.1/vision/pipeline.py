"""
ShipMonitor — боевая логика анализа кадра (детекция-агностик).

Поток на кадр:
  1. ROI бассейна (pool_roi) -> подавление ложных срабатываний вне воды.
  2. Детектор (YOLO на борту / ColorBlob для offline-теста) -> боксы.
  3. Фильтр: центр бокса ВНУТРИ ROI.
  4. Классификация цвета кропа (если детектор не дал класс) -> registered/unregistered.
  5. ArUco (если метка видна) -> marker_id, привязка к треку.
  6. Трекинг -> мостит объект между кадрами (стабильный бокс/класс).
  7. ПОДСЧЁТ УНИКАЛЬНЫХ = по РАЗНЫМ прочитанным ArUco-id (require_aruco=True).

Логика уникальности (как задумано командой):
  - Прочитан ArUco-id, которого нет в seen -> НОВАЯ лодка (+ фотофиксация, лог).
  - Прочитан id, который уже есть -> старая, не считаем.
  - id НЕ прочитан в этом кадре -> объект "обнаружен, но не опознан": рисуем и ведём
    трекером, но В ПОДСЧЁТ НЕ ИДЁТ. ("нет id" != "новая" — иначе переучёт, т.к. метка
    3 см читается через раз.)
Класс объекта фиксируем голосованием по треку (устойчивее одного кадра).

require_aruco=False — offline-фолбэк для видео БЕЗ меток: считает подтверждённые
визуальные треки (приблизительно, переучитывает на панорамах). Только для теста.

Начисление баллов: +10 обнаружение (YOLO), +20 классификация, +5 фотофиксация,
+5 ArUco-id, +10 совпадение числа, +20 стабильность без ложных (ROI), штраф -10 за ложное.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import cv2
import numpy as np

from .pool_roi import pool_mask, center_in_mask, pool_coverage
from .color import classify_color
from .tracker import CentroidTracker

CLASS_COLORS = {"registered": (0, 200, 0), "unregistered": (0, 140, 255)}
PENDING_COLOR = (200, 200, 200)


@dataclass
class ShipMonitor:
    detector: object
    aruco: object | None = None
    require_aruco: bool = True           # уникальность по ArUco-id (боевой режим)
    use_roi: bool = True
    stride: int = 3                      # тяжёлую обработку (детект/ArUco/ROI) — раз в N кадров
    roi_downscale: int = 3               # маску бассейна считаем на кадре /N (дешевле CPU)
    min_pool_cov: float = 0.05           # бассейна почти нет в кадре -> пропуск (взлёт/пол)
    confirm_conf: float = 0.35
    aruco_min_votes: int = 3             # id привязываем только после N одинаковых чтений
    aruco_max_id: int = 50               # id >= этого считаем мисдекодом (метки оргов id<=36)
    # Пороги уверенности ПО КЛАССАМ: зелёный (registered) под-уверен -> ниже порог.
    # Детектор (Yolo) должен отдавать всё >= min(порогов); финальный фильтр здесь.
    conf_by_class: dict = field(default_factory=lambda: {"registered": 0.25,
                                                          "unregistered": 0.40})
    tracker: CentroidTracker = field(default_factory=CentroidTracker)
    seen: dict = field(default_factory=dict)   # aruco_id | track-key -> запись объекта
    log: list = field(default_factory=list)
    _events: list = field(default_factory=list)
    _fi: int = 0                               # счётчик кадров (для прореживания)

    def process_frame(self, frame: np.ndarray, t: float = 0.0):
        self._events = []
        annotated = frame.copy()
        self._fi += 1

        # ТЯЖЁЛОЕ (детект + ArUco + ROI + подсчёт) — раз в stride кадров
        if self._fi % self.stride == 0:
            self._detect_and_count(frame, t)

        # ДЕШЁВОЕ (каждый кадр): рамки по живым трекам + HUD -> плавная трансляция,
        # рамки держатся между тяжёлыми кадрами. Пропускаем 1-кадровые блипы.
        for tid, track in self.tracker.tracks.items():
            if track.hits < 2 and track.aruco_id is None:
                continue
            self._draw_track(annotated, track, self.tracker.dominant_class(tid))

        self._draw_hud(annotated)
        return annotated, self._events

    def _detect_and_count(self, frame, t):
        """Тяжёлая часть: ROI -> детект -> ArUco в боксах -> трекер -> подсчёт."""
        mask = pool_mask(frame, downscale=self.roi_downscale) if self.use_roi else None
        if self.use_roi and pool_coverage(mask) < self.min_pool_cov:
            return                                          # бассейна нет в кадре

        kept = []
        for d in self.detector.detect(frame):
            if self.use_roi and not center_in_mask(d.bbox, mask, self.roi_downscale):
                continue                                    # ложное вне бассейна
            if d.label is None:
                x1, y1, x2, y2 = d.bbox
                d.label, d.score = classify_color(frame[y1:y2, x1:x2])
            if d.label is None:
                continue
            if d.score < self.conf_by_class.get(d.label, 0.0):
                continue                                    # порог по классу
            kept.append(d)

        for tid, d in self.tracker.update(kept):
            track = self.tracker.tracks[tid]
            self.tracker.vote_class(tid, d.label)
            # ArUco читаем ВНУТРИ бокса лодки, пока id не привязан. Голосуем: привязываем
            # id только после aruco_min_votes одинаковых чтений и БОЛЬШЕ не меняем —
            # так один мисдекод не плодит фантомные уникальные (было id542/id11 на одной лодке).
            if self.aruco is not None and track.aruco_id is None:
                mid = self.aruco.read_bbox(frame, d.bbox)
                if mid is not None and 0 <= mid < self.aruco_max_id:
                    track.aruco_votes[mid] = track.aruco_votes.get(mid, 0) + 1
                    best, cnt = max(track.aruco_votes.items(), key=lambda kv: kv[1])
                    if cnt >= self.aruco_min_votes:
                        track.aruco_id = best              # привязка один раз
            if self.require_aruco:
                self._count_by_aruco(track, d, t)
            else:
                self._count_by_track(tid, d, t)

    # ---- подсчёт ----
    def _count_by_aruco(self, track, d, t):
        """Уникальность = разные прочитанные ArUco-id."""
        mid = track.aruco_id
        if mid is None or mid in self.seen:
            return                                         # не опознан или уже учтён
        cls = self.tracker.dominant_class(track.tid)
        rec = {"aruco_id": mid, "class": cls, "t": round(t, 2)}
        self.seen[mid] = rec
        self.log.append(rec)
        self._events.append(("new", d, rec))               # -> фотофиксация вызывающим

    def _count_by_track(self, tid, d, t):
        """Offline-фолбэк без меток: считаем подтверждённые треки (приблизительно)."""
        key = f"track:{tid}"
        if not self.tracker.confirmed(tid) or key in self.seen or d.score < self.confirm_conf:
            return
        cls = self.tracker.dominant_class(tid)
        rec = {"aruco_id": None, "key": key, "class": cls, "t": round(t, 2)}
        self.seen[key] = rec
        self.log.append(rec)
        self._events.append(("new", d, rec))

    # ---- отрисовка ----
    def _draw_track(self, img, track, label):
        """Рисует рамку по треку (последний bbox) — держится через пропущенные кадры."""
        x1, y1, x2, y2 = track.bbox
        identified = track.aruco_id is not None
        pending = self.require_aruco and not identified     # опознаётся (ждём ArUco)
        col = PENDING_COLOR if pending else CLASS_COLORS.get(label, (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        tag = {"registered": "REG", "unregistered": "UNREG"}.get(label, "?")
        if identified:
            tag += f" id{track.aruco_id}"
        elif pending:
            tag += " ?"
        cv2.putText(img, tag, (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

    def _draw_hud(self, img):
        s = self.summary()
        pending = sum(1 for tid in self.tracker.tracks
                      if self.tracker.tracks[tid].aruco_id is None) if self.require_aruco else 0
        lines = [f"Unique (by ArUco): {s['total']}",
                 f"  registered:   {s['registered']}",
                 f"  unregistered: {s['unregistered']}"]
        if self.require_aruco:
            lines.append(f"  identifying:  {pending}")
        y = 24
        for ln in lines:
            cv2.putText(img, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(img, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y += 26

    def summary(self):
        reg = sum(1 for v in self.seen.values() if v["class"] == "registered")
        unreg = sum(1 for v in self.seen.values() if v["class"] == "unregistered")
        return {"total": reg + unreg, "registered": reg, "unregistered": unreg,
                "log": self.log}
