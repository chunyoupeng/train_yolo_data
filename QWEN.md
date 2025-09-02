# 数据制作流程
- 首要目的：把所有.zip文件合并在一起
- 标签是
```yaml
  0: bus
  1: coach
  2: truck_large
  3: truck_small
```
- 每一个zip文件解压后有`./images/*.jpg` `./labels/*.txt`
- 需要写一个脚本把这些zip文件全部解压到一个目录下面，这个里面有`./images` `./labels`
- 之后需要再写一个脚本，调用yolov8m，根据每一个图片，得到car的label，并且作为4，加到原来的lables里面
-最后把他分成train和val，标准的yolo训练格式
