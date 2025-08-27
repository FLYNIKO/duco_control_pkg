# H型梁检测器使用说明

## 概述
这是一个基于雷达数据的H型梁检测器，能够识别H型钢梁结构，包括腹板(web)和翼缘(flange)。

## 主要修复和改进

### 1. 镜像问题修复
- 修复了图像左右镜像问题
- 将 `cartesian_to_image_coords` 函数中的 `center - y` 改为 `center + y`

### 2. U型特征检测优化
- 降低了Hough变换阈值以检测更多线段
- 增加了形态学操作强度以更好地连接U型结构
- 添加了专门的曲线检测策略
- 调整了边缘检测参数

### 3. 参数配置化
- 创建了 `config.py` 配置文件
- 所有关键参数都可以在配置文件中调整
- 便于调试和优化

## 参数调整指南

### 如果检测不到线段：
1. **降低Hough变换阈值**：
   - 减小 `HOUGH_THRESHOLD` (默认: 10)
   - 减小 `FLANGE_HOUGH_THRESHOLD` (默认: 15)
   - 减小 `CURVED_HOUGH_THRESHOLD` (默认: 8)

2. **降低最小线长要求**：
   - 减小 `MIN_LINE_LENGTH` (默认: 30)
   - 减小 `FLANGE_MIN_LINE_LENGTH` (默认: 15)

3. **增加间隙容忍度**：
   - 增大 `MAX_LINE_GAP` (默认: 60)
   - 增大 `FLANGE_MAX_LINE_GAP` (默认: 25)

### 如果检测到太多噪声：
1. **提高Hough变换阈值**：
   - 增大上述阈值参数

2. **提高最小线长要求**：
   - 增大最小线长参数

3. **调整边缘检测参数**：
   - 增大 `CANNY_LOW` (默认: 5)
   - 减小 `CANNY_HIGH` (默认: 80)

### 如果U型结构连接不好：
1. **增强形态学操作**：
   - 增大 `MORPH_LARGE_KERNEL` (默认: 7)
   - 增大 `MORPH_CONNECT_KERNEL` (默认: 4)
   - 增加 `MORPH_MEDIUM_ITERATIONS` (默认: 2)

## 调试功能

程序提供了多个调试窗口：
- **Occupancy Image**: 显示雷达数据转换后的占用图像
- **Edge Detection**: 显示边缘检测结果
- **Detected Lines**: 显示检测到的线段

## 常见问题解决

### 1. 图像仍然镜像
检查 `cartesian_to_image_coords` 函数中的y坐标计算

### 2. 检测不到U型结构
- 降低相关阈值参数
- 增加形态学操作强度
- 检查雷达数据质量

### 3. 检测结果不稳定
- 增加 `STABILITY_REQUIREMENT` (默认: 3)
- 调整 `TEMPORAL_BUFFER_SIZE` (默认: 5)

### 4. 性能问题
- 减小 `IMAGE_SIZE` (默认: 600)
- 减少形态学操作迭代次数

## 运行命令
```bash
cd src/duco_control_pkg/scripts
python3 cv2_H_detecter.py
```

## 注意事项
- 确保雷达数据正常发布
- 检查坐标系设置是否正确
- 根据实际环境调整参数
- 使用调试窗口观察检测过程
