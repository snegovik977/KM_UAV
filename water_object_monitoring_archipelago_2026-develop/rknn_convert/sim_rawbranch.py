"""
Проверка RAW-BRANCH экспорта на x86-СИМУЛЯТОРЕ rknn-toolkit2.

Собирает INT8 и FP16 из best_rawbranch.onnx (форк airockchip) и смотрит на выход
ветки классов (sigmoid, [1,2,H,W]). Цель — доказать, что INT8 теперь НЕ даёт нули
(в отличие от плоского best.onnx, где INT8 class-max = 0.0000 на всех кадрах).

    python sim_rawbranch.py <img1> [img2 ...]

Выходы raw-branch (9 шт., по 3 на масштаб 80/40/20):
    idx 0,3,6 — conv2d_* [1,64,H,W]  (box-DFL регрессия)
    idx 1,4,7 — sigmoid* [1,2,H,W]   (уверенность по 2 классам)  <-- смотрим сюда
    idx 2,5,8 — clamp*   [1,1,H,W]   (score-sum, ускорение порога)
"""
import sys
import cv2
import numpy as np
from rknn.api import RKNN

ONNX = "best_rawbranch.onnx"
CLS_IDX = [1, 4, 7]   # индексы sigmoid-веток классов


def cls_stats(outs):
    """Максимальная уверенность по классам среди всех масштабов."""
    vals = []
    for i in CLS_IDX:
        vals.append(float(np.array(outs[i]).max()))
    return max(vals), vals


def run(quant, imgs):
    rknn = RKNN()
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                target_platform="rk3576",
                quantized_dtype="asymmetric_quantized-8")
    rknn.load_onnx(model=ONNX)
    rknn.build(do_quantization=quant, dataset="dataset.txt")
    rknn.init_runtime()                       # target=None -> PC-симулятор
    print(f"\n===== quant={quant} ({'INT8' if quant else 'FP16'}) =====")
    for ip in imgs:
        img = cv2.resize(cv2.imread(ip), (640, 640))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        outs = rknn.inference(inputs=[rgb])
        cm, per = cls_stats(outs)
        per_s = "/".join(f"{v:.2f}" for v in per)
        print(f"  {ip.split('/')[-1]:18s} RGB: class-max={cm:.4f}  (по масштабам 80/40/20: {per_s})")
    rknn.release()


if __name__ == "__main__":
    imgs = sys.argv[1:]
    run(True, imgs)     # INT8  <-- ключевой тест
    run(False, imgs)    # FP16  <-- эталон
