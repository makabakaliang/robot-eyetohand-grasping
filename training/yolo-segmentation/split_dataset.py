import random
import shutil
from pathlib import Path


def split_dataset(
    image_dir='photos',
    label_dir='yolo_labels',
    train_ratio=0.8,
    output_dir='dataset',
    seed=42
):
    """
    将图片和标签划分成训练集与验证集。

    参数说明：
    image_dir   : 原始图片目录
    label_dir   : YOLO 标签目录（与图片同名的 .txt 文件）
    train_ratio : 训练集比例，默认 0.8，表示 80% 训练、20% 验证
    output_dir  : 输出数据集目录
    seed        : 随机种子，用于保证每次划分结果尽量一致
    """

    # ------------------------------------------------------------------
    # 固定随机种子
    #
    # 作用：
    # - 保证每次运行划分结果更稳定
    # - 便于复现实验
    #
    # 推荐：
    # - 42 即可
    #
    # 如果你想重新随机分一次，可以换个 seed。
    # ------------------------------------------------------------------
    random.seed(seed)

    # ------------------------------------------------------------------
    # 定义输出目录结构
    #
    # 最终会生成：
    # dataset/
    #   train/
    #       images/
    #       labels/
    #   val/
    #       images/
    #       labels/
    #
    # 这和你现在 data.yaml 的结构是匹配的。:contentReference[oaicite:6]{index=6}
    # ------------------------------------------------------------------
    train_image_dir = Path(output_dir) / 'train' / 'images'
    val_image_dir = Path(output_dir) / 'val' / 'images'
    train_label_dir = Path(output_dir) / 'train' / 'labels'
    val_label_dir = Path(output_dir) / 'val' / 'labels'

    # ------------------------------------------------------------------
    # 创建目录
    #
    # parents=True  : 如果上层目录不存在，就一起创建
    # exist_ok=True : 如果目录已经存在，不报错
    # ------------------------------------------------------------------
    train_image_dir.mkdir(parents=True, exist_ok=True)
    val_image_dir.mkdir(parents=True, exist_ok=True)
    train_label_dir.mkdir(parents=True, exist_ok=True)
    val_label_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 收集所有图片文件
    #
    # 这里兼容常见图片格式：
    # - jpg
    # - jpeg
    # - png
    #
    # 如果你后面还有 bmp、webp，也可以继续往这里加。
    # ------------------------------------------------------------------
    image_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
        image_files.extend(Path(image_dir).glob(ext))

    # ------------------------------------------------------------------
    # 先排序，再随机打乱
    #
    # 为什么先 sorted？
    # - 保证同一批数据在同样 seed 下尽量稳定
    #
    # 为什么还要 shuffle？
    # - 因为不能按文件名顺序直接切，容易导致分布不均
    # ------------------------------------------------------------------
    image_files = sorted(image_files)

    # 打印找到的图片数量，方便检查路径是否正确
    print(f"找到 {len(image_files)} 个图片文件")

    # ------------------------------------------------------------------
    # 随机打乱图片顺序
    #
    # 注意：
    # 这里是“随机划分”。
    # 如果你后面想做更严格的困难验证集，
    # 可以手工把极端光照图单独放进 val。
    # ------------------------------------------------------------------
    random.shuffle(image_files)

    # ------------------------------------------------------------------
    # 计算训练集数量
    #
    # train_ratio=0.8 表示 80% 训练集
    #
    # 推荐值：
    # - 0.8：最常用
    # - 0.85：数据很少时，可以让训练集更多一些
    # - 0.7：如果你很重视验证稳定性，可多留一点验证集
    # ------------------------------------------------------------------
    train_count = int(len(image_files) * train_ratio)

    # ------------------------------------------------------------------
    # 切分训练集和验证集
    # ------------------------------------------------------------------
    train_files = image_files[:train_count]
    val_files = image_files[train_count:]

    print(f"训练集: {len(train_files)} 个文件")
    print(f"验证集: {len(val_files)} 个文件")

    # ------------------------------------------------------------------
    # 复制训练集图片和对应标签
    #
    # 逻辑：
    # - 图像文件名：xxx.jpg
    # - 标签文件名：xxx.txt
    #
    # 要求：
    # - 标签文件与图片同名（只差后缀）
    # ------------------------------------------------------------------
    for img_file in train_files:
        # stem 表示不带后缀的文件名
        # 例如 photo_001.jpg -> photo_001
        stem = img_file.stem

        # 构造对应标签路径
        label_file = Path(label_dir) / f"{stem}.txt"

        # 复制图片到训练集 images 目录
        shutil.copy2(img_file, train_image_dir / img_file.name)

        # 如果对应标签存在，就复制到训练集 labels 目录
        if label_file.exists():
            shutil.copy2(label_file, train_label_dir / label_file.name)
        else:
            # 没有标签就给警告
            # 这通常意味着：
            # - 图片没标注
            # - 标签文件名不匹配
            # - 路径写错
            print(f"警告: 找不到标签文件 {label_file}")

    # ------------------------------------------------------------------
    # 复制验证集图片和对应标签
    # ------------------------------------------------------------------
    for img_file in val_files:
        stem = img_file.stem
        label_file = Path(label_dir) / f"{stem}.txt"

        # 复制图片到验证集 images 目录
        shutil.copy2(img_file, val_image_dir / img_file.name)

        # 复制标签到验证集 labels 目录
        if label_file.exists():
            shutil.copy2(label_file, val_label_dir / label_file.name)
        else:
            print(f"警告: 找不到标签文件 {label_file}")

    # ------------------------------------------------------------------
    # 输出总结信息
    # ------------------------------------------------------------------
    print("\n数据集划分完成！")
    print(f"训练集: {len(train_files)} 个文件")
    print(f"验证集: {len(val_files)} 个文件")
    print(f"输出目录: {output_dir}")
    print(f"训练集图片: {train_image_dir}")
    print(f"训练集标签: {train_label_dir}")
    print(f"验证集图片: {val_image_dir}")
    print(f"验证集标签: {val_label_dir}")


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 直接运行这个脚本时，会按默认参数执行
    #
    # 默认含义：
    # - 从 photos 目录找图片
    # - 从 yolo_labels 目录找标签
    # - 按 8:2 划分
    # - 输出到 dataset 目录
    #
    # 如果你想改路径，可以直接改这里的参数，或者改函数默认值。
    # ------------------------------------------------------------------
    split_dataset()