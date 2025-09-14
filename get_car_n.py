import os

def count_txt_without_class4(labels_dir="labels"):
    count = 0
    total = 0
    for root, _, files in os.walk(labels_dir):
        for fname in files:
            if not fname.lower().endswith(".txt"):
                continue
            total += 1
            path = os.path.join(root, fname)
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            # 判断是否包含 class 4
            has_class4 = any(line.strip().startswith("4 ") or line.strip() == "4" for line in lines)
            if not has_class4:
                count += 1
    print(f"总共有 {total} 个 .txt 文件，其中 {count} 个文件不包含类别 4。")

if __name__ == "__main__":
    count_txt_without_class4("/root/autodl-tmp/train_yolo_data/merged_dataset/labels")