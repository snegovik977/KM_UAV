"""
Offline-прогон боевой логики на видео (тест до появления обученного YOLO и телеметрии).

    python code/run_on_video.py <input.mp4> [--out DIR] [--green] [--stride N]

Пишет: аннотированное видео, detections.json (лог уникальных объектов), photofix/ (кропы).
Детектор — ColorBlobDetector (заглушка). На борту заменяется на YoloRknnDetector, а
пиксельный трекер — на ArUco-ID/мировую координату; остальная логика не меняется.
"""
from __future__ import annotations
import argparse, json, os, sys
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from vision import ColorBlobDetector, ShipMonitor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", default="run_output")
    ap.add_argument("--green", action="store_true", help="включить ненадёжный цветовой зелёный")
    ap.add_argument("--stride", type=int, default=1, help="обрабатывать каждый N-й кадр")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Не открыть видео: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(args.out, exist_ok=True)
    photo_dir = os.path.join(args.out, "photofix")
    os.makedirs(photo_dir, exist_ok=True)
    writer = cv2.VideoWriter(os.path.join(args.out, "annotated.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps / args.stride, (w, h))

    # У отснятого видео НЕТ ArUco-меток (крепят позже) -> offline-фолбэк по трекам.
    # Уникальный счётчик тут ПРИБЛИЗИТЕЛЬНЫЙ (переучитывает на панорамах); на борту
    # require_aruco=True даёт точный подсчёт по прочитанным меткам.
    mon = ShipMonitor(detector=ColorBlobDetector(use_green=args.green),
                      require_aruco=False)
    print("[note] offline-режим без ArUco: уникальный счётчик приблизительный")

    idx = written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.stride != 0:
            idx += 1
            continue
        t = idx / fps
        annotated, events = mon.process_frame(frame, t=t)
        for kind, d, rec in events:               # фотофиксация нового объекта
            x1, y1, x2, y2 = d.bbox
            pad = 25
            crop = frame[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
            fn = f"{rec['key'].replace(':', '_')}_{rec['class']}_t{rec['t']}.jpg"
            cv2.imwrite(os.path.join(photo_dir, fn), crop)
        writer.write(annotated)
        written += 1
        idx += 1

    cap.release(); writer.release()
    summary = mon.summary()
    with open(os.path.join(args.out, "detections.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"кадров обработано: {written}")
    print(f"уникальных объектов: {summary['total']} "
          f"(registered={summary['registered']}, unregistered={summary['unregistered']})")
    print(f"фотофиксаций: {len(os.listdir(photo_dir))}")
    print(f"вывод: {args.out}/annotated.mp4, detections.json, photofix/")


if __name__ == "__main__":
    main()
