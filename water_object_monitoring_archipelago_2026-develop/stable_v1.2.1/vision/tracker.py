"""
Простой centroid-трекер для дедупа детекций между кадрами (offline-фолбэк).

На борту УНИКАЛЬНАЯ идентичность = ArUco marker_id (когда метка прочитана); при
неподвижных лодках можно дедупить и по МИРОВОЙ координате (из телеметрии дрона).
Здесь — пиксельный трекер для теста логики на видео БЕЗ телеметрии/ArUco: связывает
детекции по близости центров, выдаёт стабильные track_id и считает уникальные треки.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Track:
    tid: int
    cx: float
    cy: float
    label: str | None
    hits: int = 1
    misses: int = 0
    aruco_id: int | None = None            # ПРИВЯЗАННЫЙ ArUco-id (после голосования, не меняется)
    aruco_votes: dict = field(default_factory=dict)  # голоса за id (защита от мисдекодов)
    class_votes: dict = field(default_factory=dict)  # голоса за класс по кадрам
    bbox: tuple = (0, 0, 0, 0)             # последний известный bbox (для стабильной отрисовки)


class CentroidTracker:
    def __init__(self, max_dist: float = 120.0, max_misses: int = 8, min_hits: int = 3):
        self.max_dist = max_dist
        self.max_misses = max_misses
        self.min_hits = min_hits
        self._next = 0
        self.tracks: dict[int, Track] = {}

    def update(self, detections) -> list[tuple[int, object]]:
        """detections: list[Detection]. -> list[(track_id, Detection)] для текущего кадра."""
        centers = []
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            centers.append(((x1 + x2) / 2, (y1 + y2) / 2))

        assigned, used = [], set()
        for d, (cx, cy) in zip(detections, centers):
            best, bestd = None, self.max_dist
            for tid, t in self.tracks.items():
                if tid in used:
                    continue
                dist = ((t.cx - cx) ** 2 + (t.cy - cy) ** 2) ** 0.5
                if dist < bestd:
                    best, bestd = tid, dist
            if best is None:
                best = self._next; self._next += 1
                self.tracks[best] = Track(best, cx, cy, d.label, bbox=d.bbox)
            else:
                t = self.tracks[best]
                t.cx, t.cy, t.hits, t.misses, t.bbox = cx, cy, t.hits + 1, 0, d.bbox
                if d.label:
                    t.label = d.label
            used.add(best)
            assigned.append((best, d))

        for tid, t in list(self.tracks.items()):
            if tid not in used:
                t.misses += 1
                if t.misses > self.max_misses:
                    del self.tracks[tid]
        return assigned

    def confirmed(self, tid: int) -> bool:
        t = self.tracks.get(tid)
        return t is not None and t.hits >= self.min_hits

    def vote_class(self, tid: int, label: str | None):
        t = self.tracks.get(tid)
        if t is not None and label is not None:
            t.class_votes[label] = t.class_votes.get(label, 0) + 1

    def dominant_class(self, tid: int) -> str | None:
        t = self.tracks.get(tid)
        if not t or not t.class_votes:
            return t.label if t else None
        return max(t.class_votes.items(), key=lambda kv: kv[1])[0]
