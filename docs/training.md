# YOLO 硬盘分割训练说明

本项目的硬盘识别模型基于 Ultralytics YOLO segmentation 训练，标注数据来自 MakeSense 平台导出的 COCO 风格 `labels.json`，再转换为 YOLO 分割标签格式。

## 数据与标注

- 标注工具：MakeSense
- 任务类型：单类别实例分割
- 类别名称：`disk`
- 原始标注：`training/yolo-segmentation/annotations/makesense_labels.json`
- YOLO 标签：`training/yolo-segmentation/annotations/yolo_labels/`
- 数据配置：`training/yolo-segmentation/data.yaml`

为避免仓库体积过大，训练图片、增强后的完整数据集和模型权重没有上传。仓库中保留了标注文件、训练脚本、数据配置和训练日志，用于展示完整训练流程。

## 训练流程

1. 使用 MakeSense 对硬盘目标进行多边形分割标注，导出 COCO JSON。
2. 使用 `coco_to_yolo_converter.py` 将 COCO 标注转换为 YOLO segmentation 标签。
3. 使用 `split_dataset.py` 划分训练集和验证集。
4. 使用 `offline_augment_photometric.py` 做离线光照增强，包括亮度、对比度、gamma、阴影、模糊和噪声等变化。
5. 使用 `train.py` 启动 YOLO 分割训练。
6. 使用 `detect_segmentation.py` 对训练后的模型进行推理验证。

## 训练配置

训练配置保存在：

- `training/yolo-segmentation/train.py`
- `training/yolo-segmentation/runs/disk_seg_offline_light_plus_online_geo_v1/args.yaml`

关键参数：

| 参数 | 值 |
| --- | --- |
| 模型 | `yolo11s-seg.pt` |
| 任务 | segmentation |
| 输入尺寸 | 640 |
| batch | 8 |
| epochs | 100 |
| patience | 30 |
| workers | 2 |
| seed | 42 |
| AMP | false |
| 在线增强 | 关闭 |

## 训练结果

训练日志保存在：

- `training/yolo-segmentation/runs/disk_seg_offline_light_plus_online_geo_v1/results.csv`

从日志统计得到：

| 指标 | 结果 |
| --- | --- |
| 实际训练轮数 | 67 |
| 最佳 Box mAP50 | 0.995 |
| 最佳 Mask mAP50 | 0.995 |
| 最后一轮 Mask mAP50-95 | 0.93815 |

这些指标说明模型已经可以稳定识别工位中的硬盘目标，并为后续深度定位、位姿转换和机械臂抓取提供视觉输入。

## 未上传内容

以下内容不放入 Git 仓库：

- 原始照片和增强后的完整图片数据集
- `best.pt`、`last.pt` 等模型权重
- YOLO 自动生成的图片结果、曲线图和缓存文件
- IDE 数据库与本地环境文件

