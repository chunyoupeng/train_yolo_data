# Repository Guidelines

## Project Structure & Module Organization
- `merge_zip_files.py`: Merges dataset ZIPs into `merged_dataset/{images,labels}`.
- `detect_cars.py`: Adds car detections to YOLO label files using Ultralytics.
- `split_dataset.py`: Splits merged data into `train/` and `val/` via symlinks.
- `train.sh`: Trains YOLOv8 on the prepared dataset.
- `Makefile`: `data` target to run dataset prep (via `uv`).
- `all_zipfiles/`: Place source ZIPs here. Large data and weights are git-ignored.
- `pyproject.toml`: Python 3.12 project with `ultralytics` and `opencv-python`.

Note: Some scripts use absolute paths under `/root/autodl-tmp/three_data`. Adjust these paths for your environment before running.

## Build, Test, and Development Commands
- `uv sync`: Install deps into the local virtual env.
- `uv run merge_zip_files.py`: Extract and merge ZIPs into `merged_dataset/`.
- `uv run detect_cars.py`: Append car labels detected by YOLOv8.
- `uv run split_dataset.py`: Create `train/` and `val/` splits (symlinks).
- `bash train.sh`: Launch YOLO training with the provided hyperparams.
- `make data`: Convenience wrapper for the data pipeline (editable).

## Coding Style & Naming Conventions
- Indentation: 4 spaces; follow PEP 8 for Python.
- Names: `snake_case` for files/functions, `UPPER_SNAKE_CASE` for constants, clear module names.
- Imports: standard lib → third-party → local; prefer `pathlib.Path` over raw strings.
- Keep paths configurable (top-level constants); avoid hardcoding when feasible.

## Testing Guidelines
- Framework: none yet; prefer adding `pytest` for new logic.
- Conventions: name tests `test_*.py`, keep unit tests deterministic; mock filesystem/YOLO calls when possible.
- Quick checks: verify image/label counts after each step; spot-check labels render in a viewer.

## Commit & Pull Request Guidelines
- Commits: concise, descriptive, imperative mood; English or Chinese acceptable (current history is mixed/laconic).
- PRs: include purpose, steps to reproduce, sample commands, and before/after notes; attach small screenshots or logs where helpful.
- Data: do not commit ZIPs, model weights (`*.pt`), or generated datasets; paths are already in `.gitignore`.

## Security & Configuration Tips
- Keep secrets and API keys out of code. Use environment variables for configurable paths.
- Large data lives outside the repo; prefer symlinks over copies to save space.
