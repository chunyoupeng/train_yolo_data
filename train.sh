yolo detect train \
  data=/root/autodl-tmp/three_data/merged_dataset/data.yaml \
  model=yolov8x.pt epochs=100 imgsz=640 batch=16 device=0 \
  augment=True \
  mosaic=1.0 close_mosaic=10 \
  mixup=0.15 copy_paste=0.05 \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  degrees=7.5 translate=0.10 scale=0.50 shear=2.0 perspective=0.0005 \
  fliplr=0.5 flipud=0.0 \
  erasing=0.2