.DEFAULT_GOAL := help

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.RECIPEPREFIX := >

PY ?= uv run
ZIPS_DIR ?= /root/autodl-tmp/three_data/all_zipfiles
MERGED_FROM ?= $(ZIPS_DIR)/merged_dataset
MERGED_TO ?= /root/autodl-tmp/three_data/merged_dataset
ZIP_FILES := 757.zip 305.zip 222.zip 121.zip 118.zip 180.zip

.PHONY: help setup check-zips merge link-merged detect split data train check-yolo run clean

help: ## Show available make targets
> @awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_.-]+:.*##/ {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Install dependencies into the local environment (uv)
> uv sync

check-zips: ## Verify required ZIP files exist in $(ZIPS_DIR)
> @missing=0; \
> for z in $(ZIP_FILES); do \
>     if [ ! -f "$(ZIPS_DIR)/$$z" ]; then echo "Missing: $(ZIPS_DIR)/$$z"; missing=1; fi; \
> done; \
> if [ $$missing -eq 1 ]; then echo "Place ZIPs in $(ZIPS_DIR)"; exit 1; fi; \
> echo "All ZIPs present."

merge: check-zips ## Merge ZIPs into merged_dataset
> $(PY) merge_zip_files.py

link-merged: ## Ensure expected merged_dataset symlink exists
> @mkdir -p "$(dir $(MERGED_TO))"
> @if [ -d "$(MERGED_FROM)" ] && [ ! -e "$(MERGED_TO)" ]; then \
>     ln -s "$(MERGED_FROM)" "$(MERGED_TO)"; \
>     echo "Linked $(MERGED_TO) -> $(MERGED_FROM)"; \
> elif [ -e "$(MERGED_TO)" ] && [ ! -L "$(MERGED_TO)" ]; then \
>     echo "$(MERGED_TO) exists (not a symlink). Skipping."; \
> else \
>     echo "Using $(MERGED_TO)"; \
> fi

detect: ## Append car labels with YOLOv8
> $(PY) detect_cars.py

split: ## Create train/ and val/ splits
> $(PY) split_dataset.py

data: merge link-merged detect split ## Run full data pipeline

train: check-yolo ## Train YOLO with provided hyperparameters
> bash train.sh

check-yolo: ## Ensure yolo CLI is available
> @command -v yolo >/dev/null 2>&1 || { echo "yolo CLI not found. Run 'make setup' to install deps."; exit 1; }

run: ## Run main entry script
> $(PY) main.py

clean: ## Remove generated train/ and val/ splits (keeps merged data)
> @if [ -d "$(MERGED_TO)/train" ]; then rm -rf "$(MERGED_TO)/train"; fi
> @if [ -d "$(MERGED_TO)/val" ]; then rm -rf "$(MERGED_TO)/val"; fi
> @echo "Cleaned train/ and val/."

