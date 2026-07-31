"""
Собирает модель из ONNX (INT8 и FP16) и прогоняет на x86-СИМУЛЯТОРЕ rknn-toolkit2.
Позволяет локально увидеть, какой выход даёт квантованная/неквантованная модель.
    python sim_build_infer.py <img1> [img2 ...]
"""
import sys
import cv2
import numpy as np
from rknn.api import RKNN


def cls_max(o):
    o = np.array(o)
    if o.ndim == 3 and o.shape[1] in (5, 6):
        return float(o[0, 4:, :].max()), o.shape
    if o.ndim == 3 and o.shape[2] in (5, 6):
        return float(o[0, :, 4:].max()), o.shape
    return float("nan"), o.shape


def run(quant, imgs):
    rknn = RKNN()
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                target_platform="rk3576",
                quantized_dtype="asymmetric_quantized-8")
    rknn.load_onnx(model="best.onnx")
    rknn.build(do_quantization=quant, dataset="dataset.txt")
    rknn.init_runtime()                       # target=None -> PC-симулятор
    print(f"\n===== quant={quant} ({'INT8' if quant else 'FP16'}) =====")
    for ip in imgs:
        img = cv2.resize(cv2.imread(ip), (640, 640))
        for tag, x in [("RGB", cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), ("BGR", img)]:
            out = rknn.inference(inputs=[x])
            cm, shp = cls_max(out[0])
            print(f"  {ip.split('/')[-1]:16s} {tag}: out={shp} class-max={cm:.4f}")
    rknn.release()


if __name__ == "__main__":
    imgs = sys.argv[1:]
    run(True, imgs)     # INT8
    run(False, imgs)    # FP16
