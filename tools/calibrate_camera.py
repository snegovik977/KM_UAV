#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Калибровка камеры по шахматной доске: интринсики для config.yaml.

15 минут, ПОЛИГОН НЕ НУЖЕН — а закрывает 4 балла подзадачи 2.3.2. Пока
camera.intrinsics в конфиге пусты, локализация работает по грубой оценке из
zone.fov_h_deg, и «координаты отличаются менее чем на 30 см» по ней не проверить
(perception/calib.py предупреждает об этом при каждом старте).

Порядок:
  1. распечатать шахматку (подойдёт любая, важно знать число ВНУТРЕННИХ углов
     и размер клетки в миллиметрах);
  2. снять 20-30 кадров с борта: доска целиком в кадре, под разными углами и на разных
     расстояниях, обязательно с наклоном — по фронтальным кадрам дисторсия не считается;
  3. прогнать этот скрипт и вставить напечатанный блок в config.yaml.

Снять кадры можно двумя способами:
    python3 tools/calibrate_camera.py --grab кадры/    # прямо с камеры борта
    python3 tools/calibrate_camera.py кадры/           # посчитать по готовым

⚠️ Калибровать надо ТУ ЖЕ камеру в ТОМ ЖЕ разрешении, что идёт в перцепцию
(1080x720 на нашем борту). Интринсики привязаны к разрешению: посчитанные на
уменьшенном кадре, они дадут смещение координат в метры.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys


def снять_кадры(куда, сколько, углы, servo):
    """Съёмка шахматки с борта: кадр сохраняется, только если доска найдена."""
    import cv2

    from pioneer_sdk2 import Camera, CameraType
    try:
        from pioneer_sdk2 import ServoCamera
        ServoCamera().set_angle(servo)
    except Exception as e:
        print("сервопривод не отработал: %s: %s" % (type(e).__name__, e))

    if not os.path.isdir(куда):
        os.makedirs(куда)
    камера = Camera(camera_type=CameraType.MAIN)
    сохранено = 0
    try:
        while сохранено < сколько:
            кадр = камера.get_cv_frame(timeout=5.0)
            if кадр is None:
                continue
            серый = cv2.cvtColor(кадр, cv2.COLOR_BGR2GRAY)
            найдено, _ = cv2.findChessboardCorners(серый, углы, None)
            if not найдено:
                continue
            путь = os.path.join(куда, "calib_%03d.png" % сохранено)
            cv2.imwrite(путь, кадр)
            сохранено += 1
            print("[%d/%d] %s" % (сохранено, сколько, путь))
    finally:
        # Камера отдаётся через shared memory: не остановить — следующий запуск
        # получит TimeoutError.
        камера.stop()
    return сохранено


def посчитать(пути, углы, клетка_мм):
    import cv2
    import numpy as np

    # Углы доски в её собственной СК. Масштаб в миллиметрах роли не играет —
    # интринсики от него не зависят, — но с ним понятнее ошибка репроекции.
    образец = np.zeros((углы[0] * углы[1], 3), np.float32)
    образец[:, :2] = np.mgrid[0:углы[0], 0:углы[1]].T.reshape(-1, 2) * клетка_мм

    в_мире, в_кадре = [], []
    размер = None
    принято = 0
    критерий = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for путь in пути:
        кадр = cv2.imread(путь)
        if кадр is None:
            print("  %s: не читается" % путь)
            continue
        серый = cv2.cvtColor(кадр, cv2.COLOR_BGR2GRAY)
        if размер is None:
            размер = серый.shape[::-1]
        elif серый.shape[::-1] != размер:
            print("  %s: другое разрешение %s — пропускаю (интринсики привязаны "
                  "к разрешению)" % (путь, серый.shape[::-1]))
            continue
        найдено, точки = cv2.findChessboardCorners(серый, углы, None)
        if not найдено:
            print("  %s: доска не найдена" % путь)
            continue
        точки = cv2.cornerSubPix(серый, точки, (11, 11), (-1, -1), критерий)
        в_мире.append(образец)
        в_кадре.append(точки)
        принято += 1

    if принято < 5:
        print("Принято кадров: %d. Нужно хотя бы 5, лучше 20-30 под разными углами."
              % принято)
        return None

    ошибка, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        в_мире, в_кадре, размер, None, None)

    # Средняя ошибка репроекции — единственная честная оценка качества калибровки.
    # Больше ~1 пикселя означает, что кадры сняты плохо (мало наклонов, смазано).
    сумма = 0.0
    for i in range(len(в_мире)):
        спроецировано, _ = cv2.projectPoints(в_мире[i], rvecs[i], tvecs[i], K, dist)
        сумма += cv2.norm(в_кадре[i], спроецировано, cv2.NORM_L2) / len(спроецировано)
    ошибка_репроекции = сумма / len(в_мире)

    return {"K": K, "dist": dist.ravel(), "размер": размер, "кадров": принято,
            "ошибка": ошибка_репроекции}


def напечатать(итог):
    import math

    K = итог["K"]
    dist = list(итог["dist"]) + [0.0] * 5
    ширина, высота = итог["размер"]
    fx, fy = K[0][0], K[1][1]
    cx, cy = K[0][2], K[1][2]

    print()
    print("Кадров принято: %d, разрешение %dx%d" % (итог["кадров"], ширина, высота))
    print("Ошибка репроекции: %.3f px %s"
          % (итог["ошибка"],
             "— хорошо" if итог["ошибка"] < 1.0 else "— МНОГО, переснять кадры"))
    print("Угол обзора: горизонтальный %.1f°, вертикальный %.1f°"
          % (2 * math.degrees(math.atan(ширина / (2 * fx))),
             2 * math.degrees(math.atan(высота / (2 * fy)))))
    print()
    print("Вставить в квадрокоптер/config.yaml вместо блока camera.intrinsics:")
    print()
    print("  intrinsics:")
    print("    # Измерено tools/calibrate_camera.py: %d кадров, ошибка репроекции %.3f px"
          % (итог["кадров"], итог["ошибка"]))
    print("    fx: %.2f" % fx)
    print("    fy: %.2f" % fy)
    print("    cx: %.2f" % cx)
    print("    cy: %.2f" % cy)
    print("    k1: %.6f" % dist[0])
    print("    k2: %.6f" % dist[1])
    print("    p1: %.6f" % dist[2])
    print("    p2: %.6f" % dist[3])
    print("    k3: %.6f" % dist[4])
    print()
    print("И заодно в zone.fov_h_deg: %.1f  (шаг змейки считается из него)"
          % (2 * math.degrees(math.atan(ширина / (2 * fx)))))
    print()
    print("⚠️ Осталось измерить camera.mount.* — механический перекос крепления. "
          "Он меряется ОТДЕЛЬНО и при рабочем угле сервопривода: положить метку на "
          "известном расстоянии, снять кадр, подобрать поправки так, чтобы "
          "tools/replay.py вернул реальные координаты.")


def main():
    parser = argparse.ArgumentParser(description="Интринсики камеры по шахматной доске")
    parser.add_argument("dir", nargs="?", default="calib",
                        help="директория с кадрами шахматки")
    parser.add_argument("--grab", metavar="ДИР",
                        help="снять кадры с камеры борта в эту директорию")
    parser.add_argument("--frames", type=int, default=25, help="сколько кадров снять")
    parser.add_argument("--cols", type=int, default=9,
                        help="внутренних углов по горизонтали (у доски 10x7 это 9)")
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--square", type=float, default=25.0, help="сторона клетки, мм")
    parser.add_argument("--servo", type=int, default=-80,
                        help="угол камеры при съёмке; тот же, что в config.yaml")
    args = parser.parse_args()

    углы = (args.cols, args.rows)

    if args.grab:
        снято = снять_кадры(args.grab, args.frames, углы, args.servo)
        print("Снято кадров с найденной доской: %d" % снято)
        args.dir = args.grab

    пути = sorted(glob.glob(os.path.join(args.dir, "*.png"))
                  + glob.glob(os.path.join(args.dir, "*.jpg")))
    if not пути:
        print("В %s нет кадров. Снять их: --grab %s" % (args.dir, args.dir))
        return 1
    print("Кадров найдено: %d, ищу доску %dx%d внутренних углов"
          % (len(пути), углы[0], углы[1]))

    итог = посчитать(пути, углы, args.square)
    if итог is None:
        return 1
    напечатать(итог)
    return 0


if __name__ == "__main__":
    sys.exit(main())
