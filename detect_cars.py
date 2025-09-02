"""
有了数据之后，将小车的标签加上去
"""

from ultralytics import YOLO
import os
import cv2

# 加载预训练的YOLOv8模型
model = YOLO('yolov8m.pt')

# 定义路径
input_dir = "/root/autodl-tmp/three_data/merged_dataset"
images_dir = os.path.join(input_dir, "images")
labels_dir = os.path.join(input_dir, "labels")

# 类别映射
class_mapping = {
    0: "bus",
    1: "coach",
    2: "truck_large",
    3: "truck_small"
}

# 遍历所有图片文件
for image_file in os.listdir(images_dir):
    if image_file.endswith(".jpg"):
        # 构建图片和标签文件路径
        image_path = os.path.join(images_dir, image_file)
        label_file = image_file.replace(".jpg", ".txt")
        label_path = os.path.join(labels_dir, label_file)
        
        # 读取现有标签
        existing_labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                existing_labels = f.readlines()
        
        # 使用YOLO模型检测汽车
        results = model(image_path)
        
        # 提取检测到的汽车标签
        car_labels = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])  # 类别索引
                if cls == 2:  # 只处理汽车类别
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    # 转换为YOLO格式
                    img = cv2.imread(image_path)
                    h, w = img.shape[:2]
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    
                    # 添加新的标签 (类别4表示car)
                    car_labels.append(f"4 {x_center} {y_center} {width} {height}\n")
        
        # 将新的汽车标签添加到现有标签中
        all_labels = existing_labels + car_labels
        
        # 写入更新后的标签文件
        with open(label_path, "w") as f:
            f.writelines(all_labels)

print("所有图片的标签已更新，添加了检测到的汽车标签。")