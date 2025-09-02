import os
import zipfile
from pathlib import Path

"""
将 all_zipfiles/ 目录下的所有 ZIP 合并解压到 merged_dataset/{images,labels}
可通过环境变量覆盖：
  ZIPS_DIR   (默认: <repo>/all_zipfiles)
  MERGED_DIR (默认: <repo>/merged_dataset)
"""

ROOT = Path(__file__).resolve().parent
ZIPS_DIR = Path(os.environ.get("ZIPS_DIR", ROOT / "all_zipfiles"))
OUTPUT_DIR = Path(os.environ.get("MERGED_DIR", ROOT / "merged_dataset"))

(OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "labels").mkdir(parents=True, exist_ok=True)

zip_files = sorted(ZIPS_DIR.glob("*.zip"))
if not zip_files:
    raise SystemExit(f"未找到 ZIP 文件，请将 *.zip 放入: {ZIPS_DIR}")

for zip_path in zip_files:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.startswith("images/") and file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                dest = OUTPUT_DIR / file_name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zip_ref.open(file_name) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
            elif file_name.startswith("labels/") and file_name.endswith(".txt"):
                dest = OUTPUT_DIR / file_name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zip_ref.open(file_name) as src, open(dest, "wb") as dst:
                    dst.write(src.read())

print(f"已合并 {len(zip_files)} 个 ZIP 至: {OUTPUT_DIR}")

