"""
Детекторы корабликов с единым интерфейсом -> список Detection.

- ColorBlobDetector: заглушка на цвете/блобах. Работает СЕЙЧАС (до обучения YOLO),
  нужна для сквозного теста остальной логики на реальном видео. НЕ финальный детектор:
  липнет на жерди/метку старт/стоп (это отсекает pool_roi на уровне пайплайна).
- YoloRknnDetector: боевой детектор на борту (pioneer_rknn.Yolo, .rknn на NPU).
  Локально не запускается — интерфейс готов, подставим обученную модель позже.
"""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]      # x1, y1, x2, y2 (px)
    score: float = 1.0
    label: str | None = None             # заполняется классификатором цвета/YOLO


class ColorBlobDetector:
    """Кандидаты = регионы с оранжевыми/зелёными акцентными пикселями."""

    def __init__(self, min_area: int = 120, merge_gap: int = 22, pad: int = 6,
                 use_green: bool = False):
        # use_green=False по умолчанию: зелёный по цвету на бирюзовой воде НЕнадёжен
        # (см. валидацию) — это работа для YOLO. Заглушка держится на надёжном оранжевом.
        self.min_area = min_area
        self.merge_gap = merge_gap
        self.pad = pad
        self.use_green = use_green

    def _accent_mask(self, f: np.ndarray) -> np.ndarray:
        b, g, r = (f[..., i].astype(np.int16) for i in range(3))
        mask = (r > 60) & (r - g > 12) & (r - b > 22)          # оранжевый (надёжно)
        if self.use_green:
            mask = mask | ((g > 55) & (g < 190) & (g - r > 10) & (g - b > 12))
        return (mask.astype(np.uint8)) * 255

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        mask = self._accent_mask(frame_bgr)
        # Сомкнуть акценты одной лодки (корпус разбивает их на части)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.merge_gap, self.merge_gap))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        dets: list[Detection] = []
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_area:
                continue
            x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            x1 = max(0, x - self.pad); y1 = max(0, y - self.pad)
            x2 = min(w, x + bw + self.pad); y2 = min(h, y + bh + self.pad)
            dets.append(Detection((x1, y1, x2, y2), score=float(min(1.0, area / 400))))
        return dets


def _build_ship_yolo(model_name, object_thresh, nms_thresh):
    """Модель на базе ModelContainer с СОБСТВЕННЫМ пост-процессом плоского выхода.

    Наш ONNX -> RKNN даёт ПЛОСКИЙ декодированный выход (1, 4+2, 8400), как у их YoloFlat
    (drone/yolof.py), а НЕ 3-веточный сырой (который ждёт штатный Yolo). Поэтому:
      - модель надо регистрировать с архитектурой **custom** (как их «boat»);
      - декодим сами: (1,6,8400) -> boxes(xyxy 640), classes(argmax по 2), scores(max).
    """
    from pioneer_rknn.base import ModelContainer

    class _ShipYolo(ModelContainer):
        _printed = False

        def __init__(self):
            super().__init__(model_name=model_name)
            self.object_thresh = object_thresh
            self.nms_thresh = nms_thresh

        def run(self, inputs):
            raw = super().run(inputs)
            if raw is None:
                return None, None, None
            if not _ShipYolo._printed:          # разовая диагностика формы выхода
                try:
                    print("[detector] raw outputs:",
                          [getattr(o, "shape", type(o)) for o in raw])
                except Exception:
                    pass
                _ShipYolo._printed = True
            return self._post(np.array(raw[0]))

        def _post(self, output):
            # ждём (1, 6, 8400); подстрахуемся, если придёт (6, 8400)
            arr = output[0] if output.ndim == 3 else output   # (6, 8400)
            pred = arr.T                                       # (8400, 6)
            xywh = pred[:, :4]
            cls_scores = pred[:, 4:]                           # (8400, 2)
            classes = cls_scores.argmax(axis=1)
            scores = cls_scores.max(axis=1)

            xyxy = np.empty_like(xywh)
            xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
            xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
            xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
            xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

            keep = scores > self.object_thresh
            xyxy, scores, classes = xyxy[keep], scores[keep], classes[keep]
            if len(xyxy) == 0:
                return None, None, None
            idx = self._nms(xyxy, scores)
            return xyxy[idx], classes[idx], scores[idx]

        def _nms(self, boxes, scores):                        # как в их yolof.py
            x, y = boxes[:, 0], boxes[:, 1]
            w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
            areas = w * h
            order = scores.argsort()[::-1]
            keep = []
            while order.size > 0:
                i = order[0]; keep.append(i)
                xx1 = np.maximum(x[i], x[order[1:]]); yy1 = np.maximum(y[i], y[order[1:]])
                xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
                yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
                w1 = np.maximum(0.0, xx2 - xx1 + 1e-5); h1 = np.maximum(0.0, yy2 - yy1 + 1e-5)
                inter = w1 * h1
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
                order = order[np.where(ovr <= self.nms_thresh)[0] + 1]
            return np.array(keep)

    return _ShipYolo()


class YoloRknnDetector:
    """Боевой детектор на борту. Наш RKNN даёт плоский выход -> свой ModelContainer-декодер
    (модель регистрировать с архитектурой **custom**). Detection в координатах КАДРА.
    """

    def __init__(self, model_name: str = "yolo11nnew",
                 object_thresh: float = 0.25, nms_thresh: float = 0.45,
                 img_size: int = 640):
        self._img = img_size
        self._model = _build_ship_yolo(model_name, object_thresh, nms_thresh)
        self._labels = {0: "registered", 1: "unregistered"}  # по data.yaml

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        inp = cv2.resize(frame_bgr, (self._img, self._img))
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)  # КРИТИЧНО: модель обучена на RGB, камера даёт BGR
        inp = np.expand_dims(inp, 0)               # (1, 640, 640, 3)
        boxes, classes, scores = self._model.run([inp])
        if boxes is None or len(boxes) == 0:
            return []
        sx, sy = w / self._img, h / self._img      # 640 -> размер кадра
        dets: list[Detection] = []
        for box, cl, sc in zip(boxes, classes, scores):
            x1, y1, x2, y2 = box
            dets.append(Detection(
                (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)),
                score=float(sc), label=self._labels.get(int(cl)),
            ))
        return dets
