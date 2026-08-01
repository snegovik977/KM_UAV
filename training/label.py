#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Разметка кадров под YOLO: рамка мышью, класс цифрой, файл рядом с картинкой.

Второй шаг конвейера (docs/YOLO_TRAINING.md §3) и самый дорогой по времени: 300 кадров
размечаются вечером всей командой параллельно — каждый берёт свой клип
(`python training/label.py --clip 083847`).

Почему свой разметчик, а не labelImg/CVAT: на выданной машине (регламент 1.4) ничего,
кроме уже нужного нам opencv, может не поставиться — ни pip-пакета с Qt, ни доступа
в интернет, ни прав администратора. Здесь нужен только cv2, который на этой машине
и так обязан быть, иначе не полетит и сам дрон.

Формат — YOLO, файл `<кадр>.txt` рядом с `<кадр>.jpg`, строка на объект:

    <класс> <x_центра> <y_центра> <ширина> <высота>      всё нормировано 0..1

⚠️ ПУСТОЙ .txt — ЭТО НЕ «НЕ РАЗМЕЧЕНО», А «ПРОВЕРЕНО, СТАНЦИЙ НЕТ». Такие кадры нужны
сети не меньше положительных: белый логотип «ТЕХНОБОТ» и серые комки мусора — это
подтверждённые ложные срабатывания классического детектора (CLAUDE.md), и учить сеть
не считать их станцией можно только на кадрах, где они размечены как фон. Кадр без
.txt в датасет не попадает вообще (build_dataset.py), поэтому пролистывание ничего
не портит: пустой файл создаётся только клавишей X.

Управление (напоминание всегда есть в самом окне):

    ЛКМ протянуть   новая рамка текущим классом
    ПКМ по рамке    удалить рамку
    1..9            текущий класс; C — переназначить класс рамке под курсором
    U               убрать последнюю рамку
    X               кадр без станций (пустая разметка) и вперёд
    D               снять разметку с кадра (вернуть в «не размечено»)
    ПРОБЕЛ / N      сохранить и вперёд        B / P     назад
    S               сохранить                 Q / ESC   выход (сохранив текущий)

Запуск:

    python training/label.py                       # все клипы подряд
    python training/label.py --clip organizers     # один клип
    python training/label.py --only-unlabeled      # продолжить с неразмеченных
    python training/label.py --review              # пройти уже размеченное глазами
    python training/label.py --prelabel training/runs/stations/weights/best.pt
                                                   # предзаполнить прошлой моделью
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common                                                      # noqa: E402

try:
    import cv2
except ImportError:                       # pragma: no cover
    cv2 = None

ОКНО = "labeler"

# Цвета классов (BGR). Первый — исправная станция, второй — запылённая; дальше по кругу.
ЦВЕТА = ((80, 220, 80), (60, 160, 255), (200, 120, 255), (255, 200, 60))


def цвет(класс):
    return ЦВЕТА[класс % len(ЦВЕТА)]


class Кадр(object):
    """Одна картинка и её разметка в пикселях (в файл уходит нормированная)."""

    def __init__(self, путь, число_классов):
        self.путь = путь
        self.txt = os.path.splitext(путь)[0] + ".txt"
        self.число_классов = число_классов
        self.изображение = None
        self.боксы = []                   # (класс, x1, y1, x2, y2) в пикселях
        self.черновик = None              # предсказания прошлой модели, --prelabel
        self.изменён = False

    @property
    def размечен(self):
        return os.path.exists(self.txt)

    def загрузить(self):
        self.изображение = common.imread(self.путь)
        if self.изображение is None:
            return False
        высота, ширина = self.изображение.shape[:2]
        self.боксы = []
        for класс, xc, yc, w, h in common.читать_разметку(self.txt):
            self.боксы.append((класс,
                               (xc - w / 2.0) * ширина, (yc - h / 2.0) * высота,
                               (xc + w / 2.0) * ширина, (yc + h / 2.0) * высота))
        # Черновик модели показывается только там, где разметки ещё нет, и сам
        # по себе не сохраняется: изменён остаётся False, пока рамки не тронули.
        if not self.боксы and self.черновик:
            self.боксы = list(self.черновик)
        self.изменён = False
        return True

    def сохранить(self):
        высота, ширина = self.изображение.shape[:2]
        нормированные = []
        for класс, x1, y1, x2, y2 in self.боксы:
            xc = (x1 + x2) / 2.0 / ширина
            yc = (y1 + y2) / 2.0 / высота
            w = abs(x2 - x1) / float(ширина)
            h = abs(y2 - y1) / float(высота)
            бокс = (класс, min(max(xc, 0.0), 1.0), min(max(yc, 0.0), 1.0),
                    min(w, 1.0), min(h, 1.0))
            беда = common.проверить_бокс(бокс, self.число_классов)
            if беда:
                print("  пропущена рамка: %s" % беда)
                continue
            нормированные.append(бокс)
        common.писать_разметку(self.txt, нормированные)
        self.изменён = False

    def снять_разметку(self):
        if os.path.exists(self.txt):
            os.remove(self.txt)
        self.боксы = []
        self.изменён = False

    def под_курсором(self, x, y):
        """Индекс самой мелкой рамки, накрывающей точку: вложенные рамки иначе
        не выбрать — крупная всегда перехватывала бы клик."""
        подходящие = [(abs((x2 - x1) * (y2 - y1)), i)
                      for i, (_, x1, y1, x2, y2) in enumerate(self.боксы)
                      if min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)]
        return min(подходящие)[1] if подходящие else None


class Разметчик(object):
    def __init__(self, кадры, имена_классов, масштаб_экрана=1200):
        self.кадры = кадры
        self.имена = имена_классов
        self.масштаб_экрана = масштаб_экрана
        self.индекс = 0
        self.класс = 0
        self.масштаб = 1.0
        self.тянем = None                 # (x, y) начала протяжки, в пикселях кадра
        self.курсор = (0, 0)
        self.подсказка = True

    @property
    def кадр(self):
        return self.кадры[self.индекс]

    # ------------------------------------------------------------------ мышь

    def мышь(self, событие, x, y, флаги, _):
        точка = (x / self.масштаб, y / self.масштаб)
        self.курсор = точка
        if событие == cv2.EVENT_LBUTTONDOWN:
            self.тянем = точка
        elif событие == cv2.EVENT_LBUTTONUP and self.тянем is not None:
            x1, y1 = self.тянем
            x2, y2 = точка
            self.тянем = None
            # Клик без протяжки — не рамка, а промах: рамка в пару пикселей ломает
            # обучение молча (объект нулевой площади в лоссе).
            if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
                return
            self.кадр.боксы.append((self.класс, min(x1, x2), min(y1, y2),
                                    max(x1, x2), max(y1, y2)))
            self.кадр.изменён = True
        elif событие == cv2.EVENT_RBUTTONDOWN:
            i = self.кадр.под_курсором(*точка)
            if i is not None:
                self.кадр.боксы.pop(i)
                self.кадр.изменён = True

    # --------------------------------------------------------------- отрисовка

    def нарисовать(self):
        холст = self.кадр.изображение.copy()
        for класс, x1, y1, x2, y2 in self.кадр.боксы:
            c = цвет(класс)
            cv2.rectangle(холст, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
            cv2.putText(холст, self.имена[класс], (int(x1), max(14, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
        if self.тянем is not None:
            x1, y1 = self.тянем
            x2, y2 = self.курсор
            cv2.rectangle(холст, (int(x1), int(y1)), (int(x2), int(y2)),
                          цвет(self.класс), 1)

        высота, ширина = холст.shape[:2]
        self.масштаб = min(1.0, self.масштаб_экрана / float(ширина))
        if self.масштаб < 1.0:
            холст = cv2.resize(холст, (int(ширина * self.масштаб),
                                       int(высота * self.масштаб)))

        состояние = "размечен" if self.кадр.размечен else "НЕ размечен"
        if self.кадр.изменён:
            состояние = "изменён, не сохранён"
        шапка = "%d/%d  %s  [%s]  рамок: %d  класс: %d %s" % (
            self.индекс + 1, len(self.кадры), os.path.basename(self.кадр.путь),
            состояние, len(self.кадр.боксы), self.класс, self.имена[self.класс])
        подписи = [шапка]
        if self.подсказка:
            подписи += ["ЛКМ - рамка, ПКМ - удалить, 1..9 - класс, C - сменить класс",
                        "ПРОБЕЛ вперёд, B назад, X пусто, D снять, S сохранить, Q выход",
                        "H - убрать подсказку"]
        for i, текст in enumerate(подписи):
            y = 18 + i * 18
            cv2.putText(холст, текст, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(холст, текст, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
        return холст

    # ------------------------------------------------------------------- цикл

    def перейти(self, шаг):
        if self.кадр.изменён:
            self.кадр.сохранить()
        новый = self.индекс + шаг
        if not 0 <= новый < len(self.кадры):
            return False
        self.индекс = новый
        while not self.кадр.загрузить():
            print("не читается: %s" % self.кадр.путь)
            новый += шаг if шаг else 1
            if not 0 <= новый < len(self.кадры):
                return False
            self.индекс = новый
        return True

    def работать(self):
        cv2.namedWindow(ОКНО, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(ОКНО, self.мышь)
        if not self.кадр.загрузить():
            self.перейти(1)

        while True:
            cv2.imshow(ОКНО, self.нарисовать())
            клавиша = cv2.waitKey(20) & 0xFF
            if клавиша == 255:
                # Окно закрыли крестиком — считаем это выходом, иначе цикл повиснет.
                if cv2.getWindowProperty(ОКНО, cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue

            if клавиша in (ord("q"), 27):
                break
            elif клавиша in (ord(" "), ord("n")):
                if not self.перейти(1):
                    print("это последний кадр")
            elif клавиша in (ord("b"), ord("p")):
                if not self.перейти(-1):
                    print("это первый кадр")
            elif клавиша == ord("s"):
                self.кадр.сохранить()
            elif клавиша == ord("x"):
                self.кадр.боксы = []
                self.кадр.сохранить()
                self.перейти(1)
            elif клавиша == ord("d"):
                self.кадр.снять_разметку()
            elif клавиша == ord("u"):
                if self.кадр.боксы:
                    self.кадр.боксы.pop()
                    self.кадр.изменён = True
            elif клавиша == ord("c"):
                i = self.кадр.под_курсором(*self.курсор)
                if i is not None:
                    старый = self.кадр.боксы[i]
                    self.кадр.боксы[i] = (self.класс,) + старый[1:]
                    self.кадр.изменён = True
            elif клавиша == ord("h"):
                self.подсказка = not self.подсказка
            elif ord("1") <= клавиша <= ord("9"):
                номер = клавиша - ord("1")
                if номер < len(self.имена):
                    self.класс = номер

        if self.кадр.изменён:
            self.кадр.сохранить()
        cv2.destroyAllWindows()


def предразметить(кадры, модель, порог, число_классов):
    """Предсказания прошлой модели как черновик разметки.

    Окупается со второго круга: разметив 50 кадров и обучив на них, дальше правишь
    рамки вместо того, чтобы рисовать их с нуля. Черновик СОХРАНЯЕТСЯ ТОЛЬКО РУКАМИ —
    иначе ошибки модели молча уедут в датасет и закрепятся при следующем обучении.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("--prelabel просит ultralytics: pip install -r training/requirements.txt")
        return
    сеть = YOLO(модель)
    сделано = 0
    for кадр in кадры:
        if кадр.размечен:
            continue
        изображение = common.imread(кадр.путь)
        if изображение is None:
            continue
        итог = сеть.predict(изображение, conf=порог, verbose=False)[0]
        кадр.черновик = []
        for бокс in итог.boxes:                        # noqa: перебор боксов ultralytics
            класс = int(бокс.cls.item())
            if класс >= число_классов:
                continue
            x1, y1, x2, y2 = (float(v) for v in бокс.xyxy[0])
            кадр.черновик.append((класс, x1, y1, x2, y2))
        сделано += 1
    print("предразмечено черновиком: %d кадров (проверить и сохранить руками)" % сделано)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default=common.ПАПКА_РАЗМЕТКИ,
                        help="папка с клипами (по умолчанию training/images_to_label)")
    parser.add_argument("--clip", default=None, help="размечать только этот клип")
    parser.add_argument("--only-unlabeled", action="store_true",
                        help="показывать только кадры без .txt")
    parser.add_argument("--review", action="store_true",
                        help="показывать только уже размеченные кадры")
    parser.add_argument("--prelabel", default=None,
                        help="путь к .pt: предзаполнить рамки прошлой моделью")
    parser.add_argument("--prelabel-conf", type=float, default=0.25)
    parser.add_argument("--width", type=int, default=1200,
                        help="ширина окна в пикселях (кадр ужимается под неё)")
    аргументы = parser.parse_args()

    if cv2 is None:
        print("нужен opencv-python: pip install -r training/requirements.txt")
        return 1

    имена = common.классы()
    клипы = [аргументы.clip] if аргументы.clip else common.клипы(аргументы.dir)
    if not клипы:
        print("в %s нет ни одного клипа — сначала: python training/extract_frames.py ..."
              % аргументы.dir)
        return 1

    кадры = []
    for клип in клипы:
        for путь in common.файлы(os.path.join(аргументы.dir, клип),
                                 common.РАСШИРЕНИЯ_КАРТИНОК):
            кадр = Кадр(путь, len(имена))
            if аргументы.only_unlabeled and кадр.размечен:
                continue
            if аргументы.review and not кадр.размечен:
                continue
            кадры.append(кадр)
    if not кадры:
        print("нечего размечать по заданным условиям")
        return 1

    print("кадров: %d, из них размечено: %d"
          % (len(кадры), sum(1 for к in кадры if к.размечен)))
    print("классы: %s" % ", ".join("%d=%s" % (i + 1, имя) for i, имя in enumerate(имена)))

    if аргументы.prelabel:
        предразметить(кадры, аргументы.prelabel, аргументы.prelabel_conf, len(имена))

    Разметчик(кадры, имена, аргументы.width).работать()

    размечено = sum(1 for к in кадры if к.размечен)
    print("\nразмечено %d из %d. Дальше: python training/build_dataset.py"
          % (размечено, len(кадры)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
