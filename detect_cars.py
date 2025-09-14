"""
在不污染人工标注的前提下，为数据追加 car(类索引=4) 的伪标签，并显示处理进度：
- 若检测到的 car 与任一已有GT(任意类) IoU >= SKIP_IOU(默认0.6)，则跳过（以人工标注为主）
- 仅将“与所有GT都不重叠”的car追加到同名 .txt
- 显示整体处理进度与累计统计（added / skipped）
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple

from ultralytics import YOLO

try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # 没装 tqdm 时回退到简单打印

# ----------------------------- 配置 -----------------------------
ROOT = Path(__file__).resolve().parent
MERGED_DIR = Path(os.environ.get("MERGED_DIR", ROOT / "merged_dataset"))
IMAGES_DIR = MERGED_DIR / "images"
LABELS_DIR = MERGED_DIR / "labels"

MODEL_PATH = Path(os.environ.get("DETECT_MODEL", ROOT / "yolov8m.pt"))
BATCH_SIZE = int(os.environ.get("BATCH", 32))
CONF_THRES = float(os.environ.get("CONF", 0.25))
IMG_SIZE = int(os.environ.get("IMG_SIZE", 640))

COCO_CAR_CLS = 2          # COCO car=2
DATASET_CAR_CLS = 4       # 你的数据集 car=4
SKIP_IOU = float(os.environ.get("SKIP_IOU", 0.6))  # 与任一GT IoU>=该值则不追加

# ----------------------------- 工具函数 -----------------------------
def list_images(images_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in exts]

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def xywhn_to_xyxy(xc: float, yc: float, w: float, h: float) -> Tuple[float, float, float, float]:
    x1 = clamp01(xc - w / 2)
    y1 = clamp01(yc - h / 2)
    x2 = clamp01(xc + w / 2)
    y2 = clamp01(yc + h / 2)
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2

def xyxy_to_xywhn(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    xc = clamp01(x1 + w / 2)
    yc = clamp01(y1 + h / 2)
    return xc, yc, w, h

def iou_xyxy(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def parse_label_file(p: Path) -> List[Tuple[int, Tuple[float,float,float,float]]]:
    """读取 YOLO标注行 -> [(cls_id, (x1,y1,x2,y2)), ...]"""
    out = []
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cid = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])
        except ValueError:
            continue
        x1, y1, x2, y2 = xywhn_to_xyxy(xc, yc, w, h)
        out.append((cid, (x1, y1, x2, y2)))
    return out

# ----------------------------- 主流程（带进度） -----------------------------
def main() -> None:
    if not IMAGES_DIR.exists():
        raise SystemExit(f"未找到图片目录: {IMAGES_DIR}")
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(IMAGES_DIR)
    if not image_paths:
        print(f"{IMAGES_DIR} 下未找到图片文件")
        return

    model = YOLO(str(MODEL_PATH))

    total = len(image_paths)
    print(f"开始批量推理：共 {total} 张，batch={BATCH_SIZE}，conf={CONF_THRES}，skip_iou={SKIP_IOU}")

    added_total = 0
    skipped_overlap = 0

    pbar = tqdm(total=total, unit="img", desc="Processing", ncols=0) if tqdm else None

    for start in range(0, total, BATCH_SIZE):
        batch_paths = image_paths[start:start + BATCH_SIZE]
        results = model.predict(
            [str(p) for p in batch_paths],
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            verbose=False,
            stream=False,
        )

        batch_added = 0
        batch_skipped = 0

        for img_path, result in zip(batch_paths, results):
            label_path = LABELS_DIR / f"{img_path.stem}.txt"

            gt = parse_label_file(label_path)
            gt_boxes = [b for (_cid, b) in gt]

            existing_lines = []
            if label_path.exists():
                existing_lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

            new_lines: List[str] = []

            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                classes = boxes.cls.int().tolist()
                xywhn = boxes.xywhn.tolist()
                confs = boxes.conf.tolist() if hasattr(boxes, "conf") and boxes.conf is not None else [1.0]*len(classes)

                for cls, (xc, yc, w, h), cf in zip(classes, xywhn, confs):
                    if cls != COCO_CAR_CLS or cf < CONF_THRES:
                        continue
                    x1, y1, x2, y2 = xywhn_to_xyxy(xc, yc, w, h)

                    overlap = any(iou_xyxy((x1, y1, x2, y2), g) >= SKIP_IOU for g in gt_boxes)
                    if overlap:
                        skipped_overlap += 1
                        batch_skipped += 1
                        continue

                    xc2, yc2, w2, h2 = xyxy_to_xywhn(x1, y1, x2, y2)
                    line = f"{DATASET_CAR_CLS} {xc2:.6f} {yc2:.6f} {w2:.6f} {h2:.6f}"
                    if line not in existing_lines and line not in new_lines:
                        new_lines.append(line)

            if new_lines:
                out = existing_lines + new_lines
                Path(label_path).write_text("\n".join(out) + "\n", encoding="utf-8")
                added_total += len(new_lines)
                batch_added += len(new_lines)

        # 更新进度
        if pbar:
            pbar.update(len(batch_paths))
            pbar.set_postfix(added=added_total, skipped=skipped_overlap)
        else:
            done = min(start + BATCH_SIZE, total)
            print(f"已处理 {done}/{total} | 本批新增 {batch_added}，本批跳过 {batch_skipped} | 累计新增 {added_total}，累计跳过 {skipped_overlap}")

    if pbar:
        pbar.close()

    print(f"完成。累计新增 {added_total} 个 car 伪标签，因与GT重叠跳过 {skipped_overlap} 个。")

if __name__ == "__main__":
    main()