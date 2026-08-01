# -*- coding: utf-8 -*-
"""Общее для скриптов обучения: классы, пути, чтение картинок с кириллицей.

Отдельный модуль, потому что список классов читают пять скриптов подряд
(разметчик, сборка датасета, обучение, экспорт, конвертация), и второй копии этого
списка в репозитории быть не должно — см. training/classes.txt.

Совместимость с Python 3.8 held намеренно: обучение гоняется на выданной машине
(регламент 1.4), состав которой заранее неизвестен.
"""
from __future__ import annotations

import os
import sys

КОРЕНЬ_ОБУЧЕНИЯ = os.path.dirname(os.path.abspath(__file__))
КОРЕНЬ = os.path.dirname(КОРЕНЬ_ОБУЧЕНИЯ)

# imgio живёт в tools/: cv2.imread молча не открывает пути с кириллицей, а у нас
# кириллица и в `фото станции/`, и в рабочих папках из RUN.md.
for _путь in (os.path.join(КОРЕНЬ, "tools"), os.path.join(КОРЕНЬ, "квадрокоптер")):
    if os.path.isdir(_путь) and _путь not in sys.path:
        sys.path.insert(0, _путь)

try:
    import imgio                          # noqa: E402
except ImportError:                       # pragma: no cover — только если запустили не отсюда
    imgio = None

ФАЙЛ_КЛАССОВ = os.path.join(КОРЕНЬ_ОБУЧЕНИЯ, "classes.txt")
# Кадры для разметки: одна папка на клип (пролёт, серия снимков). Сплит train/val
# делается ПО КЛИПАМ, поэтому клип — не украшение, а единица разбиения.
ПАПКА_РАЗМЕТКИ = os.path.join(КОРЕНЬ_ОБУЧЕНИЯ, "images_to_label")
ПАПКА_ДАТАСЕТА = os.path.join(КОРЕНЬ_ОБУЧЕНИЯ, "dataset")
ФАЙЛ_VAL = os.path.join(КОРЕНЬ_ОБУЧЕНИЯ, "val_clips.txt")
# Клипы, которые размечены, но в обучение не идут (снято не той камерой и т.п.).
ФАЙЛ_SKIP = os.path.join(КОРЕНЬ_ОБУЧЕНИЯ, "skip_clips.txt")

РАСШИРЕНИЯ_КАРТИНОК = (".jpg", ".jpeg", ".png", ".bmp")
РАСШИРЕНИЯ_ВИДЕО = (".mp4", ".avi", ".mov", ".mkv")


def классы(path=None):
    """Список классов из classes.txt. Порядок = номера классов в разметке."""
    path = path or ФАЙЛ_КЛАССОВ
    имена = []
    with open(path, "rb") as f:
        for строка in f.read().decode("utf-8").splitlines():
            строка = строка.strip()
            if строка and not строка.startswith("#"):
                имена.append(строка)
    if len(имена) < 2:
        raise ValueError(
            "в %s должно быть не меньше двух классов (сейчас %d): при одном классе "
            "классовая ветка модели неотличима от score-sum по числу каналов, "
            "и наш декодер на борту разберёт выход неверно" % (path, len(имена)))
    return имена


def метки_из_конфига():
    """detector.labels из квадрокоптер/config.yaml — для сверки с classes.txt.

    Возвращает None, если конфиг не прочитался: сверка полезна, но обучение из-за
    неё останавливать нельзя (конфига может не быть на машине, где только учат).
    """
    try:
        import config as конфиг_борта      # квадрокоптер/config.py, свой парсер YAML
        cfg = конфиг_борта.load()
    except Exception:
        return None
    строка = cfg.get_path("detector.labels", "")
    if not строка:
        return None
    return [часть.strip() for часть in str(строка).split(",") if часть.strip()]


def imread(path):
    """Картинка по пути с кириллицей. None, если не прочиталась."""
    if imgio is not None:
        return imgio.imread(path)
    import cv2
    return cv2.imread(path)


def imwrite(path, image):
    """Запись картинки по пути с кириллицей. True, если файл действительно записан."""
    if imgio is not None:
        return imgio.imwrite(path, image)
    import cv2
    return bool(cv2.imwrite(path, image))


def файлы(корень, расширения):
    """Отсортированный список файлов с нужными расширениями (без рекурсии)."""
    if not os.path.isdir(корень):
        return []
    итог = []
    for имя in sorted(os.listdir(корень)):
        if os.path.splitext(имя)[1].lower() in расширения:
            итог.append(os.path.join(корень, имя))
    return итог


def клипы(корень=None):
    """Папки-клипы в images_to_label, отсортированные по имени."""
    корень = корень or ПАПКА_РАЗМЕТКИ
    if not os.path.isdir(корень):
        return []
    return sorted(имя for имя in os.listdir(корень)
                  if os.path.isdir(os.path.join(корень, имя)))


def читать_разметку(path):
    """YOLO-txt -> список (класс, xc, yc, w, h). Пустой файл = проверенный негатив."""
    боксы = []
    if not os.path.exists(path):
        return боксы
    with open(path, "rb") as f:
        for строка in f.read().decode("utf-8").splitlines():
            строка = строка.strip()
            if not строка:
                continue
            части = строка.split()
            if len(части) != 5:
                raise ValueError("строка не из 5 полей: %r" % строка)
            боксы.append((int(части[0]),) + tuple(float(ч) for ч in части[1:]))
    return боксы


def писать_разметку(path, боксы):
    """Список (класс, xc, yc, w, h) -> YOLO-txt. Пустой список пишет пустой файл:
    это не «не размечено», а «проверено, объектов нет» — такие кадры нужны сети
    не меньше положительных."""
    строки = ["%d %.6f %.6f %.6f %.6f" % (int(к), xc, yc, w, h)
              for к, xc, yc, w, h in боксы]
    with open(path, "wb") as f:
        f.write(("\n".join(строки) + ("\n" if строки else "")).encode("utf-8"))


def проверить_бокс(бокс, число_классов):
    """Причина, по которой бокс негоден, либо None. Отдельной функцией — её зовут
    и разметчик, и сборка датасета, и тесты."""
    класс, xc, yc, w, h = бокс
    if not 0 <= класс < число_классов:
        return "класс %d вне 0..%d" % (класс, число_классов - 1)
    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
        return "центр вне кадра: %.3f %.3f" % (xc, yc)
    if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
        return "размер вне (0..1]: %.3f %.3f" % (w, h)
    return None
