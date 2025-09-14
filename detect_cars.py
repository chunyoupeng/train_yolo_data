"""
批量为数据添加小车(car)的标签。

- 使用 Ultralytics YOLO 进行推理，支持批处理加速。
- 读取 merged_dataset/images 下的图片，向同名 .txt 标签文件追加 "car" 框。
  数据集类别顺序为 [bus, coach, truck_large, truck_small, car]，其中 car 的索引为 4。
  预训练 COCO 权重中 car 的类别索引为 2。

可通过环境变量覆盖默认配置：
  MERGED_DIR   默认: <repo>/merged_dataset
  DETECT_MODEL 默认: <repo>/yolov8m.pt
  BATCH        默认: 32
  CONF         默认: 0.25
  IMG_SIZE     默认: 640
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from ultralytics import YOLO


# ----------------------------- 配置 -----------------------------
ROOT = Path(__file__).resolve().parent
MERGED_DIR = Path(os.environ.get("MERGED_DIR", ROOT / "merged_dataset"))
IMAGES_DIR = MERGED_DIR / "images"
LABELS_DIR = MERGED_DIR / "labels"

MODEL_PATH = Path(os.environ.get("DETECT_MODEL", ROOT / "yolov8m.pt"))
BATCH_SIZE = int(os.environ.get("BATCH", 32))
CONF_THRES = float(os.environ.get("CONF", 0.25))
IMG_SIZE = int(os.environ.get("IMG_SIZE", 640))

# COCO 中 car 的类别索引；数据集中 car 的索引为 4
COCO_CAR_CLS = 2
DATASET_CAR_CLS = 4


def list_images(images_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in exts]


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
    print(f"开始批量推理，总计 {total} 张图片，batch={BATCH_SIZE}，conf={CONF_THRES}")

    # 按批次推理
    for start in range(0, total, BATCH_SIZE):
        batch_paths = image_paths[start : start + BATCH_SIZE]
        results = model.predict(
            [str(p) for p in batch_paths],
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            verbose=False,
            stream=False,
        )

        # 为每张图片写入/追加标签
        for img_path, result in zip(batch_paths, results):
            # 读取现有标签（如果存在）
            label_path = LABELS_DIR / f"{img_path.stem}.txt"
            existing = []
            if label_path.exists():
                existing = label_path.read_text(encoding="utf-8").splitlines(keepends=True)

            car_lines: List[str] = []
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                classes = boxes.cls.int().tolist()
                xywhn = boxes.xywhn.tolist()  # 归一化 [x,y,w,h]
                for cls, (xc, yc, w, h) in zip(classes, xywhn):
                    if cls == COCO_CAR_CLS:
                        car_lines.append(
                            f"{DATASET_CAR_CLS} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"
                        )

            # 追加写入
            all_lines = existing + car_lines
            # 始终写入文件：即便无标签也会生成空文件，行为与旧脚本一致
            label_path.write_text("".join(all_lines), encoding="utf-8")

        done = min(start + BATCH_SIZE, total)
        print(f"已处理 {done}/{total}")

    print("所有图片的标签已更新，已追加检测到的 car 框。")


if __name__ == "__main__":
    main()
