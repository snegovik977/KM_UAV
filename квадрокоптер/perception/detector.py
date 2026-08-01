# -*- coding: utf-8 -*-
"""Детекция электростанций в кадре. Два пути, переключение в config.yaml.

  classic  Классический CV. Работает с первой минуты и без единого размеченного кадра.
           Это baseline и страховка: если обучение или конвертация в .rknn не успеют,
           подзадача 2.3.2 всё равно сдаётся (docs/DRONE_PLAN.md, приложение В).
  yolo     YOLOv8/11 на NPU борта. Основной путь, требует обученной модели.

Оба отдают один список Detection в координатах кадра, поэтому остальной конвейер
(локализация, реестр) их не различает.

Порог площади берётся В МЕТРАХ, а не в пикселях: площадь в пикселях падает с высотой
квадратично, и порог, подобранный на 1.5 м, отсечёт ту же станцию на 2.5 м. Порог
в м² — свойство самой станции. Порог яркости — адаптивный, по методу Оцу на самом
кадре, а не константа. Обе меры — прямое требование универсальности (регламент 2.5):
подогнанные под конкретную расстановку пороги считаются основанием не засчитать попытку.

cv2 и numpy импортируются мягко: перцепция не имеет права ронять полёт, а на выданной
машине состав пакетов заранее неизвестен (регламент 1.4).
"""
from __future__ import annotations

from collections import namedtuple

try:
    import cv2
except ImportError:                      # pragma: no cover — на борту cv2 есть
    cv2 = None

try:
    import numpy as np
except ImportError:                      # pragma: no cover
    np = None

# bbox   — (x1, y1, x2, y2) в пикселях, для отрисовки и обрезки;
# corners — 4 угла minAreaRect, по ним считается площадь на земле (наклон камеры
#           делает прямоугольник трапецией, и bbox для площади уже неточен);
# cx, cy — центроид, именно он идёт в локализацию;
# angle  — угол длинной стороны, нужен манёвру обдува (пункт 4);
# label  — статус станции: ok | dust | broken, либо None у детектора без классов.
Detection = namedtuple("Detection", "bbox corners cx cy angle score label")

BACKENDS = ("classic", "yolo")

# Метки классов модели. Порядок = порядок в data.yaml при обучении и совпадает
# с STATUSES протокола: одна сеть закрывает и 2.3.2 (бокс), и 2.3.3 (класс).
LABELS = ("ok", "dust", "broken")


class DetectorUnavailable(RuntimeError):
    """Выбранный детектор не поднялся: нет cv2, нет pioneer_rknn, нет модели."""


def _rect_to_detection(rect, score, label, frame_shape):
    """cv2.minAreaRect -> Detection с обрезкой bbox по границам кадра."""
    (cx, cy), (w, h), angle = rect
    углы = [(float(u), float(v)) for u, v in cv2.boxPoints(rect)]
    высота, ширина = frame_shape[:2]
    x1 = max(0, int(min(u for u, _ in углы)))
    y1 = max(0, int(min(v for _, v in углы)))
    x2 = min(ширина, int(max(u for u, _ in углы)))
    y2 = min(высота, int(max(v for _, v in углы)))
    # minAreaRect отдаёт угол короткой или длинной стороны в зависимости от версии
    # OpenCV. Приводим к «углу длинной стороны» сами, чтобы пункт 4 не зависел от версии.
    if w < h:
        angle += 90.0
    return Detection((x1, y1, x2, y2), углы, float(cx), float(cy),
                     float(angle), float(score), label)


POLARITIES = ("any", "dark", "bright")

# Робастная оценка разброса: MAD * 1.4826 = сигма для нормального распределения.
_MAD_В_СИГМУ = 1.4826


class ClassicDetector(object):
    """Пятно, непохожее на пол, — вариант A из docs/DRONE_PLAN.md §2.1.

    Порядок: яркостной канал -> отклонение от фона -> морфологическая чистка ->
    контуры -> minAreaRect -> фильтр по площади в м² и по вытянутости.

    ⚠️ Отступление от плана, и намеренное. План описывал порог Оцу по тёмному пятну
    на светлом полу. Оба допущения оказались негодными:

    1. Оцу делит кадр ровно надвое и физически не может взять в один проход тёмную
       исправную панель и СВЕТЛУЮ запылённую — а «покрыта пылью» и распознаётся именно
       по повышенной яркости (docs/DRONE_PLAN.md §3.2). Пропущенная запылённая станция
       рушит и счёт станций в 2.3.2, и наведение в 2.3.4.
    2. «Светлый пол» — вообще не наш случай: на карте полигона от организаторов
       (source/2_step/) покрытие ТЁМНОЕ, со светлой разметкой.

    Поэтому фон оценивается робастно (медиана и MAD по кадру), а объектом считается
    всё, что отклонилось от него сильнее порога, в любую сторону. Порог адаптивный:
    contrast_sigma разбросов фона, но не меньше contrast_min доли диапазона — иначе
    на идеально однородном полу разброс нулевой и «объектом» становится шум АЦП.

    Детектор классов не различает: он даёт бокс, а статус в подзадаче 2.3.3 ставит
    perception/status.py по признакам внутри маски. Здесь label всегда None, и реестр
    трактует это как «статус неизвестен».
    """

    def __init__(self, min_area_m2=0.03, max_area_m2=1.20, aspect_max=4.0,
                 close_px=9, min_pixels=200, contrast_sigma=6.0, contrast_min=0.12,
                 polarity="any", log=None):
        if cv2 is None or np is None:
            raise DetectorUnavailable("классическому детектору нужны cv2 и numpy")
        if polarity not in POLARITIES:
            raise ValueError("detector.polarity должно быть одним из %s, получено %r"
                             % (POLARITIES, polarity))
        self.min_area_m2 = float(min_area_m2)
        self.max_area_m2 = float(max_area_m2)
        self.aspect_max = float(aspect_max)
        self.close_px = int(close_px)
        self.min_pixels = int(min_pixels)
        self.contrast_sigma = float(contrast_sigma)
        self.contrast_min = float(contrast_min)
        self.polarity = polarity
        self._log = log or (lambda text: print(text))

    def background(self, gray):
        """Яркость фона и порог отклонения от неё, оба — по самому кадру.

        Отдельным методом, потому что это же число печатает tools/replay.py: на разборе
        решения (регламент 2.7) надо уметь показать, откуда порог взялся, а не сказать
        «подобрали».
        """
        фон = float(np.median(gray))
        разброс = float(np.median(np.abs(gray.astype("float32") - фон))) * _MAD_В_СИГМУ
        порог = max(self.contrast_sigma * разброс, self.contrast_min * 255.0)
        return фон, порог

    def _маска(self, frame_bgr):
        яркость = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        яркость = cv2.GaussianBlur(яркость, (5, 5), 0)
        фон, порог = self.background(яркость)

        отклонение = яркость.astype("float32") - фон
        if self.polarity == "dark":
            маска = отклонение < -порог
        elif self.polarity == "bright":
            маска = отклонение > порог
        else:
            маска = np.abs(отклонение) > порог
        маска = маска.astype("uint8") * 255

        ядро = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (self.close_px, self.close_px))
        маска = cv2.morphologyEx(маска, cv2.MORPH_OPEN, ядро)
        return cv2.morphologyEx(маска, cv2.MORPH_CLOSE, ядро)

    def detect(self, frame_bgr, area_m2=None):
        """Список Detection. area_m2 — функция углов в пиксели -> площадь в м²
        (её даёт локализация по текущей позе). Без неё фильтр по площади не работает,
        и остаётся только фильтр по вытянутости — так бывает при отладке по записи
        без телеметрии."""
        маска = self._маска(frame_bgr)
        найденное = cv2.findContours(маска, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        контуры = найденное[0] if len(найденное) == 2 else найденное[1]

        детекции = []
        for контур in контуры:
            if cv2.contourArea(контур) < self.min_pixels:
                continue
            rect = cv2.minAreaRect(контур)
            (_, _), (w, h), _ = rect
            if w <= 0 or h <= 0:
                continue
            вытянутость = max(w, h) / min(w, h)
            if вытянутость > self.aspect_max:
                continue

            детекция = _rect_to_detection(rect, 1.0, None, frame_bgr.shape)
            if area_m2 is not None:
                площадь = area_m2(детекция.corners)
                if площадь is None or not self.min_area_m2 <= площадь <= self.max_area_m2:
                    continue
                # Уверенность — насколько плотно контур заполняет свой minAreaRect.
                # Реестр взвешивает по ней голоса, и «клякса» должна весить меньше
                # аккуратного прямоугольника.
                заполнение = cv2.contourArea(контур) / max(w * h, 1.0)
                детекция = детекция._replace(score=min(1.0, заполнение))
            детекции.append(детекция)
        return детекции


# ------------------------------------------------------- YOLO на NPU борта (raw-branch)

_REG = 16          # DFL-бины: box-ветка это 4 стороны * 16 бинов = 64 канала


def _to_chw(тензор, nc):
    """Выход RKNN -> (C, H, W). Раскладку опознаём по числу каналов: RKNN отдаёт
    то CHW, то NHWC, и полагаться на порядок нельзя."""
    a = np.array(тензор)
    a = a[0] if a.ndim == 4 else a
    if a.ndim == 3 and a.shape[0] not in (4 * _REG, nc, 1) and a.shape[2] in (4 * _REG, nc, 1):
        a = a.transpose(2, 0, 1)
    return a


def decode_rawbranch(outputs, nc, img_size, conf, nms_thresh):
    """Декодирование сырых веток YOLO: 9 тензоров -> (боксы xyxy, классы, скоры).

    ⚠️ Это НЕ оптимизация, а единственный работающий путь на этом железе — проверено
    другой командой на том же дроне (docs/lessons_from_archipelago.md §2). Обычный
    экспорт ONNX с плоской головой (1x6x8400, decode внутри графа) после INT8 даёт
    на борту РОВНО НОЛЬ детекций, без единой ошибки: DFL-голова не квантуется.

    Рабочая последовательность: экспорт из форка airockchip/ultralytics_yolo11 (голова
    отдаёт сырые тензоры до decode), регистрация модели с архитектурой custom, а DFL,
    грид и NMS — сами на CPU, вот здесь.

    Ветки опознаются ПО ЧИСЛУ КАНАЛОВ, а не по порядку: RKNN выходы переставляет.
      box-DFL  [1, 64, H, W]
      классы   [1, nc, H, W]   уже после sigmoid
      score-sum[1, 1,  H, W]   не используется, decode считаем по классам
    """
    масштабы = {}
    for тензор in outputs:
        a = _to_chw(тензор, nc)
        каналов, высота, _ = a.shape
        stride = img_size // высота
        ветка = масштабы.setdefault(stride, {})
        if каналов == 4 * _REG:
            ветка["box"] = a
        elif каналов == nc:
            ветка["cls"] = a
        elif каналов == 1:
            ветка["sum"] = a

    боксы, скоры, классы = [], [], []
    for stride, ветка in масштабы.items():
        if "box" not in ветка or "cls" not in ветка:
            continue
        box, cls = ветка["box"], ветка["cls"]
        _, высота, ширина = box.shape

        b = box.reshape(4, _REG, высота, ширина)
        e = np.exp(b - b.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)                     # softmax по бинам
        расстояния = (p * np.arange(_REG).reshape(1, _REG, 1, 1)).sum(axis=1)

        gy, gx = np.meshgrid(np.arange(высота), np.arange(ширина), indexing="ij")
        cx, cy = gx + 0.5, gy + 0.5
        слева, сверху, справа, снизу = расстояния
        x1 = (cx - слева) * stride
        y1 = (cy - сверху) * stride
        x2 = (cx + справа) * stride
        y2 = (cy + снизу) * stride

        лучший = cls.max(axis=0)
        какой = cls.argmax(axis=0)
        отбор = лучший > conf
        if отбор.any():
            боксы.append(np.stack([x1[отбор], y1[отбор], x2[отбор], y2[отбор]], axis=1))
            скоры.append(лучший[отбор])
            классы.append(какой[отбор])

    if not боксы:
        return None, None, None
    боксы = np.concatenate(боксы)
    скоры = np.concatenate(скоры)
    классы = np.concatenate(классы)
    оставить = nms(боксы, скоры, nms_thresh)
    return боксы[оставить], классы[оставить], скоры[оставить]


def nms(boxes, scores, thresh):
    """Подавление немаксимумов. Отдельная функция: её же зовут оффлайн-инструменты."""
    x1, y1, x2, y2 = boxes.T
    площади = (x2 - x1) * (y2 - y1)
    порядок = scores.argsort()[::-1]
    оставить = []
    while порядок.size:
        i = порядок[0]
        оставить.append(i)
        xx1 = np.maximum(x1[i], x1[порядок[1:]])
        yy1 = np.maximum(y1[i], y1[порядок[1:]])
        xx2 = np.minimum(x2[i], x2[порядок[1:]])
        yy2 = np.minimum(y2[i], y2[порядок[1:]])
        пересечение = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        доля = пересечение / (площади[i] + площади[порядок[1:]] - пересечение + 1e-9)
        порядок = порядок[np.where(доля <= thresh)[0] + 1]
    return np.array(оставить, int)


def _собрать_модель(model_name, conf, nms_thresh, nc, img_size, log):
    """ModelContainer с архитектурой custom и нашим decode.

    Архитектура именно custom: с yolov8/yolov11 ModelContainer постобработает выход
    сам, ожидая плоскую голову, и сырые ветки до нас не дойдут.
    """
    from pioneer_rknn.base import ModelContainer

    class _СтанцииYolo(ModelContainer):
        напечатал = False

        def __init__(self):
            ModelContainer.__init__(self, model_name=model_name)

        def run(self, inputs):
            сырое = ModelContainer.run(self, inputs)
            if сырое is None:
                return None, None, None
            if not _СтанцииYolo.напечатал:
                # Разовая диагностика формы: «детекций ноль» разбирается по ней
                # в первую очередь (tools/diag_model.py).
                try:
                    log("[детектор] сырые выходы: %s"
                        % [getattr(o, "shape", type(o)) for o in сырое])
                except Exception:
                    pass
                _СтанцииYolo.напечатал = True
            return decode_rawbranch(сырое, nc, img_size, conf, nms_thresh)

    return _СтанцииYolo()


class YoloRknnDetector(object):
    """Боевой детектор на NPU борта. Отдаёт и бокс (2.3.2), и класс статуса (2.3.3).

    ⚠️ .rknn и этот код — ПАРА. Raw-branch модель с плоским декодером даёт мусор,
    а плоская модель с этим декодером — ничего, и оба случая проходят без единой
    ошибки. Версионировать и заливать только вместе (docs/lessons_from_archipelago.md §2).
    """

    def __init__(self, model_name="stations", conf=0.25, nms_thresh=0.45,
                 img_size=640, labels=LABELS, log=None):
        if cv2 is None or np is None:
            raise DetectorUnavailable("детектору на NPU нужны cv2 и numpy")
        self._log = log or (lambda text: print(text))
        self.img_size = int(img_size)
        self.labels = tuple(labels)
        try:
            self._model = _собрать_модель(model_name, float(conf), float(nms_thresh),
                                          len(self.labels), self.img_size, self._log)
        except Exception as e:
            raise DetectorUnavailable("модель %r не поднялась: %s: %s"
                                      % (model_name, type(e).__name__, e))

    def detect(self, frame_bgr, area_m2=None):
        высота, ширина = frame_bgr.shape[:2]
        вход = cv2.resize(frame_bgr, (self.img_size, self.img_size))
        # КРИТИЧНО: камера борта отдаёт BGR, модель обучена на RGB. Без этого сеть
        # работает, но заметно хуже — просадка, которую легко списать на «недоучена».
        вход = cv2.cvtColor(вход, cv2.COLOR_BGR2RGB)
        боксы, классы, скоры = self._model.run([np.expand_dims(вход, 0)])
        if боксы is None or len(боксы) == 0:
            return []

        по_x = ширина / float(self.img_size)
        по_y = высота / float(self.img_size)
        детекции = []
        for бокс, класс, скор in zip(боксы, классы, скоры):
            x1, y1, x2, y2 = (бокс[0] * по_x, бокс[1] * по_y,
                              бокс[2] * по_x, бокс[3] * по_y)
            углы = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            метка = self.labels[int(класс)] if int(класс) < len(self.labels) else None
            детекции.append(Detection(
                (max(0, int(x1)), max(0, int(y1)),
                 min(ширина, int(x2)), min(высота, int(y2))),
                углы, (x1 + x2) / 2.0, (y1 + y2) / 2.0, 0.0, float(скор), метка))
        return детекции


# ------------------------------------------------------------------------ выбор

def create_detector(cfg, log=None):
    """Детектор по config.yaml -> detector.backend.

    Если основной путь не поднялся (нет pioneer_rknn, нет модели в реестре), падаем
    на классический и говорим об этом громко. Молча остаться без детекции нельзя:
    условие всех 8 баллов — совпадение числа станций с реальным.
    """
    log = log or (lambda text: print(text))
    раздел = cfg.get("detector", {})
    backend = str(раздел.get("backend", "classic"))
    if backend not in BACKENDS:
        raise ValueError("detector.backend должно быть одним из %s, получено %r"
                         % (BACKENDS, backend))

    if backend == "yolo":
        try:
            детектор = YoloRknnDetector(
                model_name=str(раздел.get("model", "stations")),
                conf=float(раздел.get("conf", 0.25)),
                nms_thresh=float(раздел.get("nms", 0.45)),
                img_size=int(раздел.get("img_size", 640)),
                log=log)
            log("[детектор] YOLO на NPU: модель %r" % раздел.get("model", "stations"))
            return детектор
        except (DetectorUnavailable, ImportError) as e:
            log("[детектор] ВНИМАНИЕ: %s" % e)
            log("[детектор] падаю на классический CV — детекция будет, но хуже")

    детектор = ClassicDetector(
        min_area_m2=float(раздел.get("min_area_m2", 0.03)),
        max_area_m2=float(раздел.get("max_area_m2", 1.20)),
        aspect_max=float(раздел.get("aspect_max", 4.0)),
        close_px=int(раздел.get("close_px", 9)),
        min_pixels=int(раздел.get("min_pixels", 200)),
        contrast_sigma=float(раздел.get("contrast_sigma", 6.0)),
        contrast_min=float(раздел.get("contrast_min", 0.12)),
        polarity=str(раздел.get("polarity", "any")),
        log=log)
    log("[детектор] классический CV (отклонение от фона, полярность %s, "
        "площадь %.2f-%.2f м²)"
        % (детектор.polarity, детектор.min_area_m2, детектор.max_area_m2))
    return детектор
