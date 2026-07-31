"""
VisionRuntime — «обвязка» нейросети для борта БВС.

Инкапсулирует ВСЮ vision-часть, чтобы код автономного полёта её не касался:
детекция (YOLO .rknn на NPU) → классификация → чтение ArUco → дедуп/подсчёт
уникальных → отрисовка рамок → вывод в терминал (форматы оргов) → запись видео →
фотофиксация → JSON-лог.

Интеграция в код полёта — две точки:
    vr = VisionRuntime(...)              # до взлёта
    ...
    frame = camera.get_cv_frame(timeout=5.0)
    annotated = vr.process(frame)        # в каждом кадре полётного цикла
    viewer.imshow("main_camera", annotated, fps=30)
    ...
    vr.finish()                          # после посадки

Требования регламента (обновление 26.07.2026), которые закрывает этот модуль:
- рамки по классу: оранжевые — незарегистрированные, зелёные — зарегистрированные;
- запись видеопотока в память оборудования (+5);
- вывод в терминал: «Обнаружено … судно», «Обнаружено судно с id …»,
  «Обнаружено … зарегистрированных/незарегистрированных судов», «Общее количество суден: …»;
- уникальность объектов по ArUco-метке (словарь 4x4-1000);
- фотофиксация каждого объекта.
"""
from __future__ import annotations
import os, sys, json

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../code
from vision.pipeline import ShipMonitor
from vision.aruco_id import ArucoReader

CLASS_RU = {"registered": "зарегистрированное", "unregistered": "незарегистрированное"}
CLASS_RU_PL = {"registered": "зарегистрированных", "unregistered": "незарегистрированных"}


class VisionRuntime:
    def __init__(self, detector, *,
                 out_dir="/home/pioneermini/workspace",
                 video_name="record.mp4", fps=30, frame_size=(1080, 720),
                 conf_registered=0.25, conf_unregistered=0.40,
                 use_roi=True, require_aruco=True):
        """
        detector — объект с методом .detect(frame)->[Detection] (YoloRknnDetector на борту).
        Пороги по классам: зелёный ниже (он под-уверен).
        require_aruco=True — боевой режим (уникальность по ArUco). False — offline-тест
        обвязки на видео без меток (подсчёт по трекам, приблизительно).
        """
        os.makedirs(out_dir, exist_ok=True)
        self.photo_dir = os.path.join(out_dir, "photofix")
        os.makedirs(self.photo_dir, exist_ok=True)
        self.log_path = os.path.join(out_dir, "detections.json")

        self.aruco = ArucoReader()  # DICT_4X4_1000 (по регламенту)
        self.monitor = ShipMonitor(
            detector=detector, aruco=self.aruco, require_aruco=require_aruco, use_roi=use_roi,
            conf_by_class={"registered": conf_registered,
                           "unregistered": conf_unregistered},
        )
        # Запись видеопотока (в память оборудования) — критерий +5.
        self._video_path = os.path.join(out_dir, video_name)
        self.writer = cv2.VideoWriter(self._video_path,
                                      cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
        self._frame_size = frame_size
        self._n = 0

    def process(self, frame, t: float = 0.0):
        """Обработать один кадр. Возвращает аннотированный кадр для viewer.imshow."""
        if frame is None:
            return None
        annotated, events = self.monitor.process_frame(frame, t=t)

        for _kind, d, rec in events:              # НОВЫЙ уникальный объект
            cls = rec["class"]
            # Ведущий '\n' и flush — чтобы сообщение не «влипло» в строку таймера
            # обратного отсчёта (тот пишет через '\r' из другого потока).
            lines = [f"Обнаружено {CLASS_RU[cls]} судно"]
            if rec.get("aruco_id") is not None:
                lines.append(f"Обнаружено судно с id {rec['aruco_id']}")
            # форматы вывода в терминал строго по регламенту
            sys.stdout.write("\n" + "\n".join(lines) + "\n")
            sys.stdout.flush()
            self._save_photofix(frame, d, rec)

        # запись кадра в видео (приводим к размеру writer при необходимости)
        if (annotated.shape[1], annotated.shape[0]) != self._frame_size:
            annotated = cv2.resize(annotated, self._frame_size)
        self.writer.write(annotated)
        self._n += 1
        return annotated

    def _save_photofix(self, frame, d, rec):
        x1, y1, x2, y2 = d.bbox
        pad = 25
        crop = frame[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
        mid = rec.get("aruco_id")
        name = f"ship_id{mid}_{rec['class']}.jpg" if mid is not None \
            else f"ship_{rec['key'].replace(':', '_')}_{rec['class']}.jpg"
        if crop.size:
            cv2.imwrite(os.path.join(self.photo_dir, name), crop)

    def finish(self):
        """Финальный вывод + сохранение лога и видео. Вызвать после посадки."""
        s = self.monitor.summary()
        # чистый блок итогов (по форматам регламента)
        sys.stdout.write(
            "\n=== ИТОГИ ===\n"
            f"Обнаружено {s['registered']} {CLASS_RU_PL['registered']} судов\n"
            f"Обнаружено {s['unregistered']} {CLASS_RU_PL['unregistered']} судов\n"
            f"Общее количество суден: {s['total']}\n"
        )
        sys.stdout.flush()

        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
        self.writer.release()
        print(f"[vision] видео: {self._video_path} ({self._n} кадров), лог: {self.log_path}")
        return s
