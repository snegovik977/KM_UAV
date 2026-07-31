"""
Прогон .rknn на x86-СИМУЛЯТОРЕ rknn-toolkit2 (без NPU) для проверки выхода локально.
    python sim_infer.py <model.rknn> <img1> [img2 ...]
Печатает форму выхода и макс. классовый скор для RGB и BGR входа.
"""
import sys
import cv2
import numpy as np
from rknn.api import RKNN


def cls_max(o):
    o = np.array(o)
    if o.ndim == 3 and o.shape[1] in (5, 6):      # (1, 4+nc, 8400)
        return float(o[0, 4:, :].max()), o.shape
    if o.ndim == 3 and o.shape[2] in (5, 6):      # (1, 8400, 4+nc)
        return float(o[0, :, 4:].max()), o.shape
    return float("nan"), o.shape


def main():
    model = sys.argv[1]
    imgs = sys.argv[2:]
    rknn = RKNN()
    if rknn.load_rknn(model) != 0:
        print("load_rknn FAILED"); return
    if rknn.init_runtime() != 0:                    # target=None -> симулятор на PC
        print("init_runtime FAILED"); return
    print(f"=== {model} ===")
    for ip in imgs:
        img = cv2.resize(cv2.imread(ip), (640, 640))
        for tag, x in [("RGB", cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), ("BGR", img)]:
            out = rknn.inference(inputs=[x])
            cm, shp = cls_max(out[0])
            print(f"  {ip.split('/')[-1]:16s} {tag}: out={shp} class-max={cm:.4f}")
    rknn.release()


if __name__ == "__main__":
    main()
