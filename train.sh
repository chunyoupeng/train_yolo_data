#!/usr/bin/env bash
set -euo pipefail

# Configurable via environment variables
# Examples:
#   Single GPU: DEVICES=0 BATCH=16 bash train.sh
#   Two GPUs:   DEVICES=0,1 BATCH=32 bash train.sh

DEVICES=${DEVICES:-0}                # e.g. 0 or 0,1
BATCH=${BATCH:-16}                   # total batch (will be split across devices)
PROJECT=${PROJECT:-runs/train}
NAME=${NAME:-exp_${DEVICES//,/}}

DATA=${DATA:-merged_dataset/data.yaml}
MODEL=${MODEL:-yolo11m.pt}
EPOCHS=${EPOCHS:-100}
IMG=${IMG:-640}

# 固定多机位稳健版（单阶段就够用）
yolo detect train \
  data="${DATA}" \
  model="${MODEL}" epochs="${EPOCHS}" imgsz="${IMG}" batch="${BATCH}" device="${DEVICES}" \
  project="${PROJECT}" name="${NAME}" \
  augment=True multi_scale=True \
  mosaic=0.4 close_mosaic=$((EPOCHS/5)) \
  mixup=0.05 copy_paste=0.0 \
  hsv_h=0.015 hsv_s=0.40 hsv_v=0.25 \
  degrees=1.0 translate=0.03 scale=0.25 shear=0.3 perspective=0.0005 \
  fliplr=0.0 flipud=0.0 \
  erasing=0.03
