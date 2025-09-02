"""
利用 Ultralytics YOLOv8 批量（并行加载/多进程可分片）检测 COCO 的 car 类别，并将结果以 YOLO 格式追加为自定义类别 4。

环境变量（可选）：
  MERGED_DIR  数据根目录，默认 <repo>/merged_dataset
  MODEL       模型权重，默认 yolov8m.pt
  BATCH       批大小，默认 16（总批次）
  WORKERS     DataLoader 并行 worker 数，默认 4
  DEVICE      设备，默认 0（或 cpu）
  CONF        置信度阈值，默认 0.25
  IMG         推理尺寸 imgsz，默认 640
  SHARDS      分片总数，默认 1（>1 时可多进程并行）
  SHARD_ID    当前分片序号 [0..SHARDS-1]，默认 0
"""

import os
from pathlib import Path
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
INPUT_DIR = Path(os.environ.get("MERGED_DIR", ROOT / "merged_dataset"))
IMAGES_DIR = INPUT_DIR / "images"
LABELS_DIR = INPUT_DIR / "labels"

MODEL_PATH = os.environ.get("MODEL", "yolov8m.pt")
BATCH = int(os.environ.get("BATCH", 16))
WORKERS = int(os.environ.get("WORKERS", 4))
DEVICE = os.environ.get("DEVICE", "0")  # e.g. "0" or "cpu"
CONF = float(os.environ.get("CONF", 0.25))
IMG = int(os.environ.get("IMG", 640))
SHARDS = int(os.environ.get("SHARDS", 1))
SHARD_ID = int(os.environ.get("SHARD_ID", 0))

if not IMAGES_DIR.exists():
    raise SystemExit(f"未找到图像目录: {IMAGES_DIR}")

model = YOLO(MODEL_PATH)

# 列表化全部图片，必要时进行分片，便于多进程并行
files = []
for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
    files.extend(sorted(IMAGES_DIR.glob(ext)))
if not files:
    raise SystemExit(f"未在 {IMAGES_DIR} 找到图片文件")

if SHARDS > 1:
    if not (0 <= SHARD_ID < SHARDS):
        raise SystemExit(f"SHARD_ID 必须在 [0,{SHARDS-1}]，当前: {SHARD_ID}")
    # 采用切片分片，天然无重叠
    files = files[SHARD_ID::SHARDS]

results = model.predict(
    source=[str(p) for p in files],
    batch=BATCH,
    workers=WORKERS,
    device=DEVICE,
    conf=CONF,
    imgsz=IMG,
    stream=True,   # 流式迭代，降低内存占用
    verbose=False,
)

num_images = 0
num_added = 0

LABELS_DIR.mkdir(parents=True, exist_ok=True)

for r in results:
    num_images += 1
    img_path = Path(r.path)
    label_path = LABELS_DIR / (img_path.stem + ".txt")

    if not r.boxes or len(r.boxes) == 0:
        continue

    lines = []
    for b in r.boxes:
        cls = int(b.cls[0])
        if cls == 2:  # COCO: car
            x, y, w, h = b.xywhn[0].tolist()
            lines.append(f"4 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    if lines:
        with open(label_path, "a", encoding="utf-8") as f:
            f.writelines(lines)
        num_added += len(lines)

print(f"处理完成：图像 {num_images} 张，新增 car 标签 {num_added} 条。")
