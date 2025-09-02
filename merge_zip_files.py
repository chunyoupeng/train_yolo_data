import os
import zipfile

# 定义路径
zip_files = ["/root/autodl-tmp/three_data/all_zipfiles/757.zip", "/root/autodl-tmp/three_data/all_zipfiles/305.zip", "/root/autodl-tmp/three_data/all_zipfiles/222.zip", "/root/autodl-tmp/three_data/all_zipfiles/121.zip", "/root/autodl-tmp/three_data/all_zipfiles/118.zip", "/root/autodl-tmp/three_data/all_zipfiles/180.zip"]
output_dir = "/root/autodl-tmp/three_data/all_zipfiles/merged_dataset"

# 创建输出目录
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

# 解压所有zip文件
for zip_path in zip_files:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # 获取zip文件中的所有文件列表
        file_list = zip_ref.namelist()
        
        # 分别处理images和labels文件夹中的文件
        for file_name in file_list:
            if file_name.startswith("images/") and file_name.endswith(".jpg"):
                # 提取图片文件
                with zip_ref.open(file_name) as source, \
                     open(os.path.join(output_dir, file_name), "wb") as target:
                    target.write(source.read())
            elif file_name.startswith("labels/") and file_name.endswith(".txt"):
                # 提取标签文件
                with zip_ref.open(file_name) as source, \
                     open(os.path.join(output_dir, file_name), "wb") as target:
                    target.write(source.read())

print("所有zip文件已成功解压并合并到 merged_dataset 目录中。")