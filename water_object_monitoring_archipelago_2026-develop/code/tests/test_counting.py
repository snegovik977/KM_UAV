"""
Детерминированный тест логики уникального подсчёта по ArUco-id.

Синтетика: две "лодки" (id5 зелёная=registered, id7 оранжевая=unregistered) в
фиксированных позициях. Метка читается НЕ в каждом кадре — проверяем, что:
  - уникальных ровно 2 (а не по детекции на кадр);
  - классы верные;
  - объект без прочитанной метки НЕ увеличивает счётчик (статус "опознаётся");
  - ложная детекция без цвета отбрасывается и не считается.

Запуск:  python code/tests/test_counting.py   (или pytest)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cv2
import numpy as np
from vision.pipeline import ShipMonitor
from vision.aruco_id import ArucoReader, DEFAULT_DICT
from vision.detector import Detection

ADICT = cv2.aruco.getPredefinedDictionary(DEFAULT_DICT)
H, W = 400, 640

# лодки: (box, accent_bgr, aruco_id)
BOAT_A = ((90, 140, 150, 90), (55, 135, 55), 5)    # зелёный акцент (G доминирует)
BOAT_B = ((360, 170, 150, 90), (30, 90, 210), 7)   # оранжевый акцент (R доминирует)
FP = ((560, 300, 40, 40), (110, 110, 110), None)   # серый ложный -> classify None


def _marker_patch(mid, inner=60, pad=12):
    m = cv2.aruco.generateImageMarker(ADICT, mid, inner)
    m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    return cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def build_frame(present):
    """present: list of (boat_spec, draw_marker_bool). -> (frame, detections)."""
    f = np.full((H, W, 3), 60, np.uint8)
    dets = []
    for (box, accent, mid), draw_marker in present:
        x, y, w, h = box
        cv2.rectangle(f, (x, y), (x + w, y + h), (180, 120, 60), -1)   # голубой корпус
        f[y + 45:y + h - 5, x + 5:x + w - 5] = accent                  # акцент (низ = корма)
        if draw_marker and mid is not None:
            p = _marker_patch(mid)
            ph, pw = p.shape[:2]
            f[y + 3:y + 3 + ph, x + 3:x + 3 + pw] = p                  # метка на носу
        dets.append(Detection((x, y, x + w, y + h)))
    return f, dets


class ScriptedDetector:
    def __init__(self):
        self._dets = []
    def set(self, dets):
        self._dets = dets
    def detect(self, frame):
        return list(self._dets)


def run():
    det = ScriptedDetector()
    mon = ShipMonitor(detector=det, aruco=ArucoReader(), require_aruco=True, use_roi=False)

    # сценарий по кадрам: (лодки в кадре, читается ли метка)
    script = (
        [([(BOAT_A, True)], )] * 3 +          # A с меткой -> учтётся
        [([(BOAT_A, False)], )] * 4 +         # A без метки -> не новая, счётчик не растёт
        [([(BOAT_A, False), (BOAT_B, False)], )] * 3 +  # B появилась, метка НЕ читается -> pending
        [([(BOAT_A, True), (BOAT_B, True)], )] * 5      # обе с метками -> B учтётся
    )
    totals = []
    for i, (present,) in enumerate(script):
        present = [p if isinstance(p, tuple) and len(p) == 2 else (p, True) for p in present]
        frame, dets = build_frame(present + [(FP, False)])
        det.set(dets)
        mon.process_frame(frame, t=i / 8.0)
        totals.append(mon.summary()["total"])

    s = mon.summary()
    print("итоговый summary:", {k: v for k, v in s.items() if k != "log"})
    print("лог уникальных:", s["log"])
    print("динамика total по кадрам:", totals)

    assert s["total"] == 2, f"ожидали 2 уникальных, получили {s['total']}"
    assert s["registered"] == 1 and s["unregistered"] == 1, "неверное распределение классов"
    ids = {r["aruco_id"] for r in s["log"]}
    assert ids == {5, 7}, f"ожидали id {{5,7}}, получили {ids}"
    cls = {r["aruco_id"]: r["class"] for r in s["log"]}
    assert cls[5] == "registered" and cls[7] == "unregistered", f"классы перепутаны: {cls}"
    # A учтена рано (кадр 0..2), счётчик не рос в кадрах 3..9 без метки B
    assert totals[2] == 1 and totals[9] == 1, f"переучёт до чтения метки B: {totals}"
    assert totals[-1] == 2, "B не учтена после чтения метки"
    print("\n✅ TEST PASSED: уникальность по ArUco-id, без переучёта, классы верны, ложное отброшено")


def test_counting():   # pytest-совместимо
    run()


if __name__ == "__main__":
    run()
