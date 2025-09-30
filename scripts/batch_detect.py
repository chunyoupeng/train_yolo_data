#!/usr/bin/env python3
"""Batch-detect all images within a folder using an Ultralytics YOLO model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO detection on every image inside a directory.",
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Folder that holds the images to be processed.",
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to the YOLO weights file (e.g. yolov8m.pt).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional directory to store detection results (defaults to runs/detect).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="batch_detect",
        help="Folder name for this run when --output is supplied.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size passed to the model (default: 640).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold used for filtering detections (default: 0.25).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for inference (default: 16).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device specifier such as '0', '0,1', 'cpu'.",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Also export detections as YOLO-format text files.",
    )
    parser.add_argument(
        "--save-conf",
        action="store_true",
        help="Include per-box confidence scores in the exported txt files.",
    )
    return parser.parse_args()


def collect_images(folder: Path) -> List[Path]:
    """Return every image file under the given folder (recursively)."""
    return sorted(
        p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
    )


def main() -> None:
    args = parse_args()

    image_dir = args.image_dir.expanduser().resolve()
    model_path = args.model.expanduser().resolve()

    if not image_dir.is_dir():
        raise SystemExit(f"Image directory not found: {image_dir}")
    if not model_path.is_file():
        raise SystemExit(f"Model file not found: {model_path}")

    images = collect_images(image_dir)
    if not images:
        print(f"No images with supported extensions found under {image_dir}")
        return

    model = YOLO(str(model_path))

    predict_kwargs = {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "batch": args.batch,
        "device": args.device,
        "save": True,
        "stream": False,
    }
    if args.save_txt:
        predict_kwargs["save_txt"] = True
    if args.save_conf:
        predict_kwargs["save_conf"] = True

    output_dir = None
    if args.output is not None:
        output_dir = args.output.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        predict_kwargs.update(
            {
                "project": str(output_dir),
                "name": args.run_name,
                "exist_ok": True,
            }
        )

    print(
        f"Running detection on {len(images)} images using model {model_path.name} "
        f"(imgsz={args.imgsz}, conf={args.conf}, batch={args.batch})"
    )

    results = model.predict([str(p) for p in images], **predict_kwargs)

    save_dir = None
    if results:
        first_result = results[0]
        save_dir_str = getattr(first_result, "save_dir", None)
        if save_dir_str:
            save_dir = Path(save_dir_str)

    if save_dir is None and output_dir is not None:
        save_dir = output_dir / args.run_name

    if save_dir is not None:
        print(f"Detections saved to {save_dir}")
    else:
        print("Detection complete. Check the default runs/detect directory for results.")


if __name__ == "__main__":
    main()
