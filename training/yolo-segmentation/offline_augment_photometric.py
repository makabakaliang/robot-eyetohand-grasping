#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
离线光照增强脚本（适用于 YOLO 分割 / 检测标签）
------------------------------------------------
特点：
1. 只做“不改变几何位置”的增强：
   - 亮度
   - 对比度
   - Gamma
   - 饱和度
   - 高斯模糊
   - 高斯噪声
   - 阴影
2. 因为不改目标位置/形状，所以标签文件可以直接复制
3. 推荐只对训练集 dataset/train/images 做增强
4. 不要对验证集 val 做增强

使用前提：
- 图片和标签同名，例如：
  dataset/train/images/0001.jpg
  dataset/train/labels/0001.txt

输出结果：
- 在原训练集目录中新增增强版图片
- 同时复制增强版标签
"""

import random
import shutil
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# 一、基础配置（你最常改的是这里）
# ============================================================

# 训练集图片目录
TRAIN_IMAGE_DIR = "dataset/train/images"

# 训练集标签目录
TRAIN_LABEL_DIR = "dataset/train/labels"

# 是否保留原图（通常保留）
KEEP_ORIGINAL = True

# 随机种子，保证可复现
SEED = 42

# 每张原图生成哪些增强版本
# 你可以增删，但建议第一版别开太多
AUGMENTATIONS_TO_USE = [
    "dark",         # 变暗
    "bright",       # 变亮
    "contrast_up",  # 提高对比度
    "contrast_down",# 降低对比度
    "saturation_up",# 提高饱和度
    "gamma_dark",   # gamma压暗
    "gamma_bright", # gamma提亮
    "blur",         # 轻微模糊
    "noise",        # 轻微噪声
    "shadow",       # 人造阴影
]

# 是否限制每张图生成的增强数量
# None 表示全都生成
# 例如设成 4，表示每张图随机挑4种增强
MAX_AUG_PER_IMAGE = 5


# ============================================================
# 二、增强参数（推荐值都写在这里）
# ============================================================

# 亮度增强参数
# alpha > 1 会更亮，alpha < 1 会更暗
BRIGHT_ALPHA = 1.25   # 推荐 1.15 ~ 1.35
DARK_ALPHA = 0.72     # 推荐 0.60 ~ 0.85

# 对比度增强参数
CONTRAST_UP_ALPHA = 1.30    # 推荐 1.15 ~ 1.40
CONTRAST_DOWN_ALPHA = 0.78  # 推荐 0.70 ~ 0.90

# 饱和度增强参数（HSV 的 S 通道倍率）
SATURATION_UP_ALPHA = 1.35  # 推荐 1.15 ~ 1.50

# Gamma 增强
# gamma < 1 会变亮，gamma > 1 会变暗
GAMMA_BRIGHT = 0.75   # 推荐 0.70 ~ 0.90
GAMMA_DARK = 1.45     # 推荐 1.20 ~ 1.60

# 模糊
BLUR_KERNEL = 5       # 推荐 3 或 5，别太大

# 高斯噪声
NOISE_STD = 8         # 推荐 5 ~ 12

# 阴影
SHADOW_ALPHA_MIN = 0.45   # 阴影区域最暗程度
SHADOW_ALPHA_MAX = 0.70   # 阴影区域最亮程度


# ============================================================
# 三、基础工具函数
# ============================================================

def set_seed(seed: int = 42):
    """固定随机种子，便于复现。"""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path):
    """确保目录存在。"""
    path.mkdir(parents=True, exist_ok=True)


def list_images(image_dir: Path):
    """收集常见格式图片。"""
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        image_files.extend(image_dir.glob(ext))
    return sorted(image_files)


def read_image(image_path: Path):
    """读取图片。"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    return img


def save_image(image_path: Path, image: np.ndarray):
    """保存图片。"""
    ok = cv2.imwrite(str(image_path), image)
    if not ok:
        raise ValueError(f"保存图片失败: {image_path}")


def copy_label(src_label: Path, dst_label: Path):
    """
    复制标签。
    因为这里做的是非几何增强，目标位置不变，所以标签可直接复用。
    """
    if not src_label.exists():
        print(f"警告: 标签不存在，跳过复制 -> {src_label}")
        return
    shutil.copy2(src_label, dst_label)


def clamp_uint8(img: np.ndarray):
    """裁剪到 0~255 并转为 uint8。"""
    return np.clip(img, 0, 255).astype(np.uint8)


# ============================================================
# 四、具体增强函数（都不改几何位置）
# ============================================================

def adjust_brightness(img: np.ndarray, alpha: float):
    """
    调整亮度
    alpha > 1 更亮
    alpha < 1 更暗
    """
    out = img.astype(np.float32) * alpha
    return clamp_uint8(out)


def adjust_contrast(img: np.ndarray, alpha: float):
    """
    调整对比度
    以图像均值为中心拉伸/压缩
    """
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    out = (img.astype(np.float32) - mean) * alpha + mean
    return clamp_uint8(out)


def adjust_saturation(img: np.ndarray, alpha: float):
    """
    调整饱和度
    在 HSV 空间中操作 S 通道
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= alpha
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def adjust_gamma(img: np.ndarray, gamma: float):
    """
    Gamma 变换
    gamma < 1 提亮
    gamma > 1 压暗
    """
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255 for i in range(256)
    ]).astype("uint8")
    return cv2.LUT(img, table)


def gaussian_blur(img: np.ndarray, ksize: int = 5):
    """
    高斯模糊
    推荐 ksize=3 或 5，太大容易失真
    """
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def add_gaussian_noise(img: np.ndarray, std: float = 8):
    """
    加轻微高斯噪声
    """
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return clamp_uint8(out)


def add_shadow(img: np.ndarray,
               alpha_min: float = 0.45,
               alpha_max: float = 0.70):
    """
    添加一块随机阴影区域
    只影响亮度，不改几何位置
    """
    h, w = img.shape[:2]
    shadow = np.ones((h, w), dtype=np.float32)

    # 随机生成一个四边形阴影区域
    x1 = random.randint(0, w // 2)
    y1 = random.randint(0, h // 2)
    x2 = random.randint(w // 2, w - 1)
    y2 = random.randint(0, h // 2)
    x3 = random.randint(w // 2, w - 1)
    y3 = random.randint(h // 2, h - 1)
    x4 = random.randint(0, w // 2)
    y4 = random.randint(h // 2, h - 1)

    poly = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)

    alpha = random.uniform(alpha_min, alpha_max)
    shadow[mask > 0] = alpha

    out = img.astype(np.float32).copy()
    for c in range(3):
        out[:, :, c] *= shadow

    return clamp_uint8(out)


# ============================================================
# 五、增强调度
# ============================================================

def apply_augmentation(img: np.ndarray, aug_name: str):
    """
    根据增强名称，返回增强后的图像。
    """
    if aug_name == "dark":
        return adjust_brightness(img, DARK_ALPHA)

    elif aug_name == "bright":
        return adjust_brightness(img, BRIGHT_ALPHA)

    elif aug_name == "contrast_up":
        return adjust_contrast(img, CONTRAST_UP_ALPHA)

    elif aug_name == "contrast_down":
        return adjust_contrast(img, CONTRAST_DOWN_ALPHA)

    elif aug_name == "saturation_up":
        return adjust_saturation(img, SATURATION_UP_ALPHA)

    elif aug_name == "gamma_dark":
        return adjust_gamma(img, GAMMA_DARK)

    elif aug_name == "gamma_bright":
        return adjust_gamma(img, GAMMA_BRIGHT)

    elif aug_name == "blur":
        return gaussian_blur(img, BLUR_KERNEL)

    elif aug_name == "noise":
        return add_gaussian_noise(img, NOISE_STD)

    elif aug_name == "shadow":
        return add_shadow(img, SHADOW_ALPHA_MIN, SHADOW_ALPHA_MAX)

    else:
        raise ValueError(f"不支持的增强类型: {aug_name}")


def build_aug_image_name(stem: str, suffix: str, ext: str):
    """
    构造增强图片文件名
    例如：
    0001 + dark + .jpg -> 0001__aug_dark.jpg
    """
    return f"{stem}__aug_{suffix}{ext}"


def build_aug_label_name(stem: str, suffix: str):
    """
    构造增强标签文件名
    例如：
    0001 + dark -> 0001__aug_dark.txt
    """
    return f"{stem}__aug_{suffix}.txt"


# ============================================================
# 六、主流程
# ============================================================

def main():
    set_seed(SEED)

    image_dir = Path(TRAIN_IMAGE_DIR)
    label_dir = Path(TRAIN_LABEL_DIR)

    ensure_dir(image_dir)
    ensure_dir(label_dir)

    image_files = list_images(image_dir)

    print("=" * 60)
    print("离线光照增强开始")
    print("=" * 60)
    print(f"训练集图片目录: {image_dir}")
    print(f"训练集标签目录: {label_dir}")
    print(f"找到原始图片数量: {len(image_files)}")
    print(f"计划使用增强类型: {AUGMENTATIONS_TO_USE}")
    print(f"MAX_AUG_PER_IMAGE = {MAX_AUG_PER_IMAGE}")
    print()

    total_new_images = 0
    total_missing_labels = 0

    for img_path in image_files:
        stem = img_path.stem
        ext = img_path.suffix
        label_path = label_dir / f"{stem}.txt"

        # 如果没有标签，通常不做增强
        if not label_path.exists():
            print(f"警告: 找不到对应标签，跳过 {img_path.name}")
            total_missing_labels += 1
            continue

        img = read_image(img_path)

        # 每张图到底用哪些增强
        aug_list = AUGMENTATIONS_TO_USE.copy()
        if MAX_AUG_PER_IMAGE is not None:
            aug_count = min(MAX_AUG_PER_IMAGE, len(aug_list))
            aug_list = random.sample(aug_list, aug_count)

        for aug_name in aug_list:
            aug_img = apply_augmentation(img, aug_name)

            new_img_name = build_aug_image_name(stem, aug_name, ext)
            new_lbl_name = build_aug_label_name(stem, aug_name)

            new_img_path = image_dir / new_img_name
            new_lbl_path = label_dir / new_lbl_name

            save_image(new_img_path, aug_img)
            copy_label(label_path, new_lbl_path)

            total_new_images += 1
            print(f"生成: {new_img_name}  | 标签: {new_lbl_name}")

    final_image_count = len(list_images(image_dir))
    final_label_count = len(list(label_dir.glob("*.txt")))

    print()
    print("=" * 60)
    print("离线光照增强完成")
    print("=" * 60)
    print(f"原始训练图片数量: {len(image_files)}")
    print(f"新增增强图片数量: {total_new_images}")
    print(f"增强后训练图片总数: {final_image_count}")
    print(f"训练标签总数: {final_label_count}")
    print(f"缺少标签而跳过的图片数: {total_missing_labels}")
    print()
    print("注意:")
    print("1. 本脚本只增强 train，不动 val")
    print("2. 因为只做光照/模糊/噪声类增强，所以标签直接复制")
    print("3. 如果以后做旋转/平移/裁剪，标签不能直接复制，必须同步变换")


if __name__ == "__main__":
    main()