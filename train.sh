yolo detect train \
  data=/root/autodl-tmp/three_data/merged_dataset/data.yaml \
  model=yolov8x.pt epochs=100 imgsz=640 batch=16 device=0 \
  augment=True multi_scale=True \
  mosaic=0.8 close_mosaic=15 \
  mixup=0.10 copy_paste=0.05 \
  hsv_h=0.015 hsv_s=0.50 hsv_v=0.35 \
  degrees=3.0 translate=0.05 scale=0.40 shear=1.0 perspective=0.001 \
  fliplr=0.0 flipud=0.0 \
  erasing=0.1
