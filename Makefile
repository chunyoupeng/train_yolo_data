.DEFAULT_GOAL := help

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.RECIPEPREFIX := >

PY ?= uv run
ZIPS_DIR ?= $(PWD)/all_zipfiles
MERGED_DIR ?= $(PWD)/merged_dataset
DATA ?= $(MERGED_DIR)/data.yaml

.PHONY: help setup check-zips merge detect split data train train-multi check-yolo run clean

help: ## Show available make targets
> @awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_.-]+:.*##/ {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Install dependencies into the local environment (uv)
> uv sync

check-zips: ## Verify there are ZIP files in $(ZIPS_DIR)
> @count=$$(ls -1 "$(ZIPS_DIR)"/*.zip 2>/dev/null | wc -l | tr -d ' '); \
> if [ "$$count" = "0" ]; then echo "No ZIP files in $(ZIPS_DIR)"; exit 1; else echo "Found $$count ZIP(s)."; fi

merge: check-zips ## Merge ZIPs into merged_dataset
> MERGED_DIR=$(MERGED_DIR) ZIPS_DIR=$(ZIPS_DIR) $(PY) merge_zip_files.py

detect: ## Append car labels with YOLOv8
> MERGED_DIR=$(MERGED_DIR) $(PY) detect_cars.py

DETECT_GPUS ?= 0,1
DETECT_BATCH ?= 4
DETECT_WORKERS ?= 2

detect-multi: ## Parallel detection across GPUs using shards (set DETECT_GPUS)
> IFS=',' read -r -a GPUS <<< '$(DETECT_GPUS)'; \
> N=$${#GPUS[@]}; \
> for i in $$(seq 0 $$((N-1))); do \
>   gpu=$${GPUS[$$i]}; \
>   echo "Launch shard $$i/$$((N-1)) on GPU $$gpu"; \
>   CUDA_VISIBLE_DEVICES=$$gpu \
>   MERGED_DIR=$(MERGED_DIR) \
>   DEVICE=0 SHARDS=$$N SHARD_ID=$$i \
>   BATCH=$(DETECT_BATCH) WORKERS=$(DETECT_WORKERS) \
>   $(PY) detect_cars.py & \
> done; \
> wait; \
> echo "Multi-GPU detection finished."

split: ## Create train/ and val/ splits
> MERGED_DIR=$(MERGED_DIR) $(PY) split_dataset.py

data: merge detect split ## Run full data pipeline

train: check-yolo ## Train YOLO with provided hyperparameters
> DATA=$(DATA) bash train.sh

TRAIN_MULTI_DEVICES ?= 0,1
TRAIN_MULTI_BATCH ?= 16

train-multi: check-yolo ## Train on two GPUs (override with TRAIN_MULTI_DEVICES/TRAIN_MULTI_BATCH)
> DATA=$(DATA) DEVICES=$(TRAIN_MULTI_DEVICES) BATCH=$(TRAIN_MULTI_BATCH) bash train.sh

check-yolo: ## Ensure yolo CLI is available
> @command -v yolo >/dev/null 2>&1 || { echo "yolo CLI not found. Run 'make setup' to install deps."; exit 1; }

run: ## Run main entry script
> $(PY) main.py

clean: ## Remove generated train/ and val/ splits (keeps merged data)
> @if [ -d "$(MERGED_DIR)" ]; then rm -rf "$(MERGED_DIR)"; fi
> @if [ -d "$(MERGED_DIR)/train" ]; then rm -rf "$(MERGED_DIR)/train"; fi
> @if [ -d "$(MERGED_DIR)/val" ]; then rm -rf "$(MERGED_DIR)/val"; fi
> @echo "Cleaned $(MERGED_DIR) train/ and val/."
