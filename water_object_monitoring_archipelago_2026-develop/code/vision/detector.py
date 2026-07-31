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


_REG = 16       # DFL-бины (box-ветка 64 канала = 4 стороны * 16 бинов)


def _to_chw(a, nc):
    """RKNN-выход -> (C,H,W). Опознаём раскладку по числу каналов (CHW или NHWC)."""
    a = np.array(a)
    a = a[0] if a.ndim == 4 else a
    if a.ndim == 3 and a.shape[0] not in (4 * _REG, nc, 1) and a.shape[2] in (4 * _REG, nc, 1):
        a = a.transpose(2, 0, 1)                       # NHWC -> CHW
    return a


def decode_rawbranch(outputs, nc, img, conf, nms_thresh):
    """RAW-BRANCH decode: 9 сырых веток (форк airockchip) -> (xyxy@img, classes, scores).

    Ветки: box-DFL [1,64,H,W], классы(sigmoid) [1,nc,H,W], score-sum [1,1,H,W] — по 3
    на масштаб (80/40/20). Опознаём по числу каналов, не по порядку (RKNN может
    переставлять выходы). DFL+грид считаем на CPU (в графе их нет -> INT8 не ломается).
    Проверено на симуляторе: INT8-боксы совпадают с FP16 (IoU 0.92–1.00).
    """
    scales = {}
    for o in outputs:
        a = _to_chw(o, nc)
        C, H, W = a.shape
        stride = img // H
        d = scales.setdefault(stride, {})
        if C == 4 * _REG:  d["box"] = a
        elif C == nc:      d["cls"] = a
        elif C == 1:       d["sum"] = a
    all_xyxy, all_sc, all_cl = [], [], []
    for stride, d in scales.items():
        if "box" not in d or "cls" not in d:
            continue
        box, cls = d["box"], d["cls"]                  # (64,H,W), (nc,H,W)
        _, H, W = box.shape
        b = box.reshape(4, _REG, H, W)
        e = np.exp(b - b.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)           # softmax по бинам
        dist = (p * np.arange(_REG).reshape(1, _REG, 1, 1)).sum(axis=1)  # (4,H,W): l,t,r,b
        gy, gx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        cx, cy = gx + 0.5, gy + 0.5
        l, t, r, bo = dist
        x1 = (cx - l) * stride; y1 = (cy - t) * stride
        x2 = (cx + r) * stride; y2 = (cy + bo) * stride
        smax = cls.max(axis=0); sarg = cls.argmax(axis=0)               # уже sigmoid, 0..1
        m = smax > conf
        if m.any():
            all_xyxy.append(np.stack([x1[m], y1[m], x2[m], y2[m]], axis=1))
            all_sc.append(smax[m]); all_cl.append(sarg[m])
    if not all_xyxy:
        return None, None, None
    xyxy = np.concatenate(all_xyxy)
    scores = np.concatenate(all_sc)
    classes = np.concatenate(all_cl)
    keep = _nms(xyxy, scores, nms_thresh)
    return xyxy[keep], classes[keep], scores[keep]


def _nms(boxes, scores, thr):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]; keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1); h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[np.where(ovr <= thr)[0] + 1]
    return np.array(keep, int)


def _build_ship_yolo(model_name, object_thresh, nms_thresh, nc, img_size):
    """Модель на базе ModelContainer с RAW-BRANCH пост-процессом.

    Модель — INT8 raw-branch RKNN (форк airockchip): decode-голова вынесена из графа,
    выход = 9 сырых веток (не плоский (1,6,8400)). Поэтому:
      - регистрировать с архитектурой **custom** (ModelContainer отдаёт сырые выходы);
      - decode (DFL+грид+NMS) делаем сами в decode_rawbranch() на CPU.
    Плоский INT8 раньше давал нули (DFL в графе не квантуется) — этот путь и есть фикс.
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
            if not _ShipYolo._printed:          # разовая диагностика формы выходов
                try:
                    print("[detector] raw outputs:",
                          [getattr(o, "shape", type(o)) for o in raw])
                except Exception:
                    pass
                _ShipYolo._printed = True
            return decode_rawbranch(raw, nc, img_size, self.object_thresh, self.nms_thresh)

    return _ShipYolo()


class YoloRknnDetector:
    """Боевой детектор на борту. RKNN даёт RAW-BRANCH выход (9 веток) -> decode_rawbranch
    (модель регистрировать с архитектурой **custom**). Detection в координатах КАДРА.
    """

    def __init__(self, model_name: str = "yolo11nnew",
                 object_thresh: float = 0.25, nms_thresh: float = 0.45,
                 img_size: int = 640):
        self._img = img_size
        self._labels = {0: "registered", 1: "unregistered"}  # по data.yaml
        self._model = _build_ship_yolo(model_name, object_thresh, nms_thresh,
                                       len(self._labels), img_size)

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
