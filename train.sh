#!/usr/bin/env bash
set -euo pipefail

# Configurable via environment variables
# Examples:
#   Single GPU: DEVICES=0 BATCH=16 bash train.sh
#   Two GPUs:   DEVICES=0,1 BATCH=32 bash train.sh

DEVICES=${DEVICES:-0}                # e.g. 0 or 0,1
BATCH=${BATCH:-16}                   # total batch (will be split across devices)
PROJECT=${PROJECT:-runs/train}

# DATA can be provided as env; if not, try common locations
DATA=${DATA:-}
MODEL=${MODEL:-yolo11m.pt}
EPOCHS=${EPOCHS:-100}
IMG=${IMG:-640}

if [[ -z "${DATA}" ]]; then
  if [[ -n "${MERGED_DIR:-}" && -f "${MERGED_DIR}/data.yaml" ]]; then
    DATA="${MERGED_DIR}/data.yaml"
  elif [[ -f "merged_dataset/data.yaml" ]]; then
    DATA="merged_dataset/data.yaml"
  elif [[ -f "data.yaml" ]]; then
    DATA="data.yaml"
  else
    echo "ERROR: DATA not set and no data.yaml found. Set DATA or MERGED_DIR." >&2
    exit 1
  fi
fi

dataset_slug=${DATA##*/}
dataset_slug=${dataset_slug%.yaml}
dataset_slug=${dataset_slug%.yml}
if [[ "${dataset_slug}" == "data" || -z "${dataset_slug}" ]]; then
  dataset_slug=$(basename "$(dirname "${DATA}")")
fi
dataset_slug=${dataset_slug:-dataset}
if [[ "${dataset_slug}" == "." ]]; then
  dataset_slug=dataset
fi
dataset_slug=${dataset_slug//[^a-zA-Z0-9_-]/_}

NAME=${NAME:-${dataset_slug}_${DEVICES//,/}}

echo "Using dataset: ${DATA}"
echo "Run name: ${NAME}"
echo "Devices: ${DEVICES} | Batch: ${BATCH} | Img: ${IMG} | Epochs: ${EPOCHS}"

# 固定多机位稳健版（单阶段就够用）
yolo detect train \
  data="${DATA}" \
  model="${MODEL}" epochs="${EPOCHS}" imgsz="${IMG}" batch="${BATCH}" device="${DEVICES}" \
  project="${PROJECT}" name="${NAME}" \
  augment=True multi_scale=True \
  mosaic=0.15 close_mosaic=$((EPOCHS/4)) \
  mixup=0.0 copy_paste=0.1 \
  hsv_h=0.015 hsv_s=0.40 hsv_v=0.25 \
  degrees=1.0 translate=0.03 scale=0.2 shear=0.3 perspective=0.0005 \
  fliplr=0.0 flipud=0.0 \
  erasing=0.03
