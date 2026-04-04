import os
import torch
from ultralytics import YOLO    # type: ignore
import cv2
import numpy as np


def calculate_mask_center(mask):
    """
    计算掩码的中心点坐标

    Args:
        mask (numpy.ndarray): 二值化掩码图像

    Returns:
        tuple: 中心点坐标 (x, y)，如果无法计算则返回 None
    """
    # 边界检查
    if mask is None or mask.size == 0:
        return None

    # 确保掩码是二值化的
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    # 获取掩码的轮廓
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # 获取最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)

    # 计算轮廓的矩
    moments = cv2.moments(largest_contour)

    # 计算中心点坐标
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return (cx, cy)
    else:
        return None


def process_image_detection(model, image_path, output_dir, image_index, total_images):
    """
    处理单张图像的分割检测

    Args:
        model: YOLO模型实例
        image_path (str): 图像文件路径
        output_dir (str): 结果保存目录
        image_index (int): 当前图像索引
        total_images (int): 总图像数量
    """
    try:
        image_file = os.path.basename(image_path)
        print(f"处理图像 {image_index+1}/{total_images}: {image_file}")

        # 使用模型进行预测，设置置信度为0.9
        results = model(image_path, conf=0.8)

        # 显示结果
        result = results[0]  # 获取第一个（也是唯一一个）结果

        # 保存带注释的图像
        output_path = os.path.join(output_dir, f"result_{image_file}")
        result.save(output_path)  # 保存结果图像

        # 打印检测信息
        boxes = result.boxes
        if boxes is not None:
            print(f"  检测到 {len(boxes)} 个目标")
            for j, box in enumerate(boxes):
                cls = int(box.cls[0])  # 类别
                conf = float(box.conf[0])  # 置信度
                print(f"    目标 {j+1}: 类别={cls}, 置信度={conf:.2f}")

        # 如果有分割掩码
        if hasattr(result, 'masks') and result.masks is not None:
            print(f"  检测到 {len(result.masks)} 个分割掩码")

            # 计算每个掩码的中心点坐标
            masks = result.masks.data  # 修复：直接使用data属性
            # 计算每个掩码的长和宽像素
            for j, mask in enumerate(masks):
                # 将掩码转换为numpy数组
                mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask

                # 将掩码调整为原始图像大小
                mask_resized = cv2.resize(
                    mask_np,
                    (result.orig_shape[1], result.orig_shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

                # 计算掩码的长和宽
                mask_height, mask_width = mask_resized.shape
                print(f"    掩码 {j+1} 的尺寸: 宽={mask_width} 像素, 高={mask_height} 像素")

                # 计算中心点
                center_point = calculate_mask_center(mask_resized)

                if center_point:
                    print(f"    掩码 {j+1} 的中心点坐标: {center_point}")
                else:
                    print(f"    掩码 {j+1}: 未能计算中心点坐标")

    except Exception as e:
        print(f"处理图像 {image_file} 时出错: {e}")
        import traceback
        traceback.print_exc()


def initialize_model():
    """
    初始化分割模型
    
    Returns:
        YOLO: 加载的模型实例
    """
    # 解决OpenMP库冲突问题
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # 加载预训练的分割模型
    model = YOLO('/home/yyj/train/runs/segment/runs/segment/disk_seg_offline_light_plus_online_geo_v1/weights/best.pt')  # 加载提供的权重文件
    return model


def setup_directories():
    """
    设置必要的目录
    
    Returns:
        tuple: (验证集图像路径, 结果保存目录)
    """
    # 定义验证集路径
    val_images_path = 'dataset/val/images'
    
    # 创建保存结果的目录
    output_dir = 'detection_results_seg_s'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    return val_images_path, output_dir


def get_image_files(images_path):
    """
    获取图像文件列表
    
    Args:
        images_path (str): 图像目录路径
        
    Returns:
        list: 图像文件名列表
    """
    image_files = [
        f for f in os.listdir(images_path) 
        if f.endswith(('.jpg', '.jpeg', '.png'))
    ]
    return image_files


def main():
    """主函数"""
    try:
        # 初始化模型
        model = initialize_model()

        # 设置目录
        val_images_path, output_dir = setup_directories()

        # 获取验证集中的所有图像文件
        image_files = get_image_files(val_images_path)

        print(f"找到 {len(image_files)} 张验证图像")

        # 对每张图像进行检测
        for i, image_file in enumerate(image_files):
            # 构建完整的图像路径
            image_path = os.path.join(val_images_path, image_file)

            # 处理单张图像检测
            process_image_detection(model, image_path, output_dir, i, len(image_files))

        print(f"检测完成，结果保存在 {output_dir} 目录中")

    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()