"""
Полный decode raw-branch выхода + сравнение INT8 vs FP16 на симуляторе.

Доказываем: (1) наш CPU-decode из 9 веток даёт осмысленные боксы;
(2) INT8-боксы совпадают с FP16 (IoU) -> квантование не сломало детекцию.
Проверенную здесь функцию decode_rawbranch() затем переносим в code/vision/detector.py.

    python sim_rawbranch_decode.py <img1> [img2 ...]
"""
import sys
import cv2
import numpy as np
from rknn.api import RKNN

ONNX = "best_rawbranch.onnx"
NC = 2
IMG = 640
REG = 16          # DFL bins (64 = 4*16)


def _to_chw(a, nc):
    a = np.array(a)
    a = a[0] if a.ndim == 4 else a
    if a.ndim == 3 and a.shape[0] not in (4 * REG, nc, 1) and a.shape[2] in (4 * REG, nc, 1):
        a = a.transpose(2, 0, 1)              # NHWC -> CHW
    return a


def decode_rawbranch(outputs, nc=NC, img=IMG, conf=0.25, nms=0.45):
    """9 сырых веток -> (xyxy[640], classes, scores). Ветки опознаём по числу каналов."""
    scales = {}
    for o in outputs:
        a = _to_chw(o, nc)
        C, H, W = a.shape
        stride = img // H
        d = scales.setdefault(stride, {})
        if C == 4 * REG:   d["box"] = a
        elif C == nc:      d["cls"] = a
        elif C == 1:       d["sum"] = a
    all_xyxy, all_sc, all_cl = [], [], []
    for stride, d in scales.items():
        box, cls = d["box"], d["cls"]                 # (64,H,W), (nc,H,W)
        _, H, W = box.shape
        b = box.reshape(4, REG, H, W)
        e = np.exp(b - b.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        dist = (p * np.arange(REG).reshape(1, REG, 1, 1)).sum(axis=1)   # (4,H,W): l,t,r,b
        gy, gx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        cx, cy = gx + 0.5, gy + 0.5
        l, t, r, bo = dist
        x1 = (cx - l) * stride; y1 = (cy - t) * stride
        x2 = (cx + r) * stride; y2 = (cy + bo) * stride
        smax = cls.max(axis=0); sarg = cls.argmax(axis=0)              # (H,W)
        m = smax > conf
        if m.any():
            all_xyxy.append(np.stack([x1[m], y1[m], x2[m], y2[m]], axis=1))
            all_sc.append(smax[m]); all_cl.append(sarg[m])
    if not all_xyxy:
        return np.empty((0, 4)), np.empty(0, int), np.empty(0)
    xyxy = np.concatenate(all_xyxy); sc = np.concatenate(all_sc); cl = np.concatenate(all_cl)
    keep = _nms(xyxy, sc, nms)
    return xyxy[keep], cl[keep], sc[keep]


def _nms(boxes, scores, thr):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]; keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1); h = np.maximum(0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[np.where(ovr <= thr)[0] + 1]
    return np.array(keep, int)


def _iou(a, b):
    xx1 = max(a[0], b[0]); yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2]); yy2 = min(a[3], b[3])
    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0


def build(quant):
    rknn = RKNN()
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                target_platform="rk3576", quantized_dtype="asymmetric_quantized-8")
    rknn.load_onnx(model=ONNX)
    rknn.build(do_quantization=quant, dataset="dataset.txt")
    rknn.init_runtime()
    return rknn


if __name__ == "__main__":
    imgs = sys.argv[1:]
    int8, fp16 = build(True), build(False)
    for ip in imgs:
        img = cv2.resize(cv2.imread(ip), (IMG, IMG))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bi, ci, si = decode_rawbranch(int8.inference(inputs=[rgb]), conf=0.25)
        bf, cf, sf = decode_rawbranch(fp16.inference(inputs=[rgb]), conf=0.25)
        print(f"\n=== {ip.split('/')[-1]} ===")
        print(f"  INT8: {len(bi)} боксов | FP16: {len(bf)} боксов")
        for b, c, s in sorted(zip(bi.tolist(), ci.tolist(), si.tolist()), key=lambda z: -z[2])[:5]:
            best = max((_iou(b, fb) for fb in bf), default=0)
            lbl = {0: "reg", 1: "unreg"}[c]
            print(f"    INT8 {lbl:5s} score={s:.2f} box=[{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}] "
                  f"IoU_с_FP16={best:.2f}")
