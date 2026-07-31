"""
Сборка YOLO-датасета из размеченных кадров (Ultralytics-формат, для YOLOv11n).

Источник: images_to_label/<clip>/<name>.jpg + <name>.txt (per-file YOLO).
  - Кадр берётся, ТОЛЬКО если у него есть .txt (пустой .txt = проверенный негатив).
  - Кадры без .txt = ещё не размечены -> пропускаются.
Результат: dataset/images/{train,val} + dataset/labels/{train,val}  (+ используется data.yaml).

Сплит — ПО КЛИПАМ (не по кадрам!), иначе near-duplicate кадры протекут в val.
Идемпотентно: čистит dataset/ и пересобирает. Разметишь ещё — просто перезапусти.

    python training/build_dataset.py
"""
import os, glob, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "images_to_label")
OUT = os.path.join(HERE, "dataset")

# Клипы в VAL (целиком). Держим зелёно-богатые/сбалансированные клипы в TRAIN.
# 101741: целый дрон-клип, оба класса + много hard-negatives (оценка ложных). ~16%.
VAL_CLIPS = {"101741"}


def valid_label(path):
    """Проверка формата YOLO; пустой файл = валидный негатив."""
    for ln in open(path):
        ln = ln.strip()
        if not ln:
            continue
        p = ln.split()
        if len(p) != 5:
            return False, f"строк не из 5 полей: {ln!r}"
        try:
            c = int(p[0]); xc, yc, w, h = map(float, p[1:])
        except ValueError:
            return False, f"нечисловые значения: {ln!r}"
        if c not in (0, 1) or not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
            return False, f"вне диапазона: {ln!r}"
    return True, ""


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    for split in ("train", "val"):
        os.makedirs(os.path.join(OUT, "images", split))
        os.makedirs(os.path.join(OUT, "labels", split))

    stats = {"train": {}, "val": {}}
    bad = []
    for clip in sorted(os.listdir(SRC)):
        cdir = os.path.join(SRC, clip)
        if not os.path.isdir(cdir):
            continue
        split = "val" if clip in VAL_CLIPS else "train"
        for txt in sorted(glob.glob(os.path.join(cdir, "*.txt"))):
            stem = os.path.splitext(os.path.basename(txt))[0]
            jpg = os.path.join(cdir, stem + ".jpg")
            if not os.path.exists(jpg):
                bad.append((txt, "нет парного .jpg")); continue
            ok, why = valid_label(txt)
            if not ok:
                bad.append((txt, why)); continue
            shutil.copy2(jpg, os.path.join(OUT, "images", split, stem + ".jpg"))
            shutil.copy2(txt, os.path.join(OUT, "labels", split, stem + ".txt"))
            n = sum(1 for l in open(txt) if l.strip())
            s = stats[split].setdefault(clip, {"frames": 0, "boxes": 0})
            s["frames"] += 1; s["boxes"] += n

    print(f"{'split':6} {'clip':10} {'frames':>7} {'boxes':>6}")
    for split in ("train", "val"):
        for clip, s in sorted(stats[split].items()):
            print(f"{split:6} {clip:10} {s['frames']:>7} {s['boxes']:>6}")
    tf = sum(s['frames'] for s in stats['train'].values()); tb = sum(s['boxes'] for s in stats['train'].values())
    vf = sum(s['frames'] for s in stats['val'].values());   vb = sum(s['boxes'] for s in stats['val'].values())
    print(f"\nTRAIN: {tf} кадров, {tb} боксов | VAL: {vf} кадров, {vb} боксов | всего {tf+vf}")
    if bad:
        print(f"\n⚠ пропущено проблемных: {len(bad)}")
        for b in bad[:10]:
            print("  ", b)
    print(f"\nГотово: {OUT}/images|labels/{{train,val}} . Запуск обучения: cd training && python train.py")


if __name__ == "__main__":
    main()
