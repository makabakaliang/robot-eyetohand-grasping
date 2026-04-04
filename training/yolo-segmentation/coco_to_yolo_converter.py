#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COCO -> YOLO Segmentation 标签转换器
----------------------------------
适用于 Ultralytics YOLO11-seg / YOLOv8-seg

功能：
1. 将 COCO JSON 标注转换为 YOLO 分割标签
2. 输出格式严格为：
   class_id x1 y1 x2 y2 x3 y3 ...
3. 支持一个目标包含多个 polygon：
   - 每个 polygon 单独输出一行
4. 自动生成 classes.txt
5. 自动生成适配当前项目的 data.yaml

适配你的项目流程：
- 原图目录：photos/
- 转换输出：yolo_labels/
- 后续再由 split_dataset.py 划分到 dataset/train 和 dataset/val
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class CocoToYoloSegConverter:
    def __init__(
        self,
        coco_json_path: str,
        output_dir: str = "yolo_labels",
        min_points: int = 3,
        skip_crowd: bool = True,
    ):
        """
        Args:
            coco_json_path: COCO 格式 JSON 文件路径
            output_dir: 输出 YOLO 标签目录
            min_points: polygon 至少需要几个点（>=3）
            skip_crowd: 是否跳过 iscrowd=1 的标注
        """
        self.coco_json_path = Path(coco_json_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.min_points = min_points
        self.skip_crowd = skip_crowd

        if not self.coco_json_path.exists():
            raise FileNotFoundError(f"找不到 COCO JSON 文件: {self.coco_json_path}")

        with open(self.coco_json_path, "r", encoding="utf-8") as f:
            self.coco_data = json.load(f)

        self.images = self.coco_data.get("images", [])
        self.annotations = self.coco_data.get("annotations", [])
        self.categories_raw = self.coco_data.get("categories", [])

        if not self.images:
            raise ValueError("COCO JSON 中没有 images")
        if not self.categories_raw:
            raise ValueError("COCO JSON 中没有 categories")

        # 按 COCO category id 排序，保证映射稳定
        self.categories_sorted = sorted(self.categories_raw, key=lambda c: c["id"])

        # COCO category_id -> category_name
        self.catid_to_name: Dict[int, str] = {
            cat["id"]: cat["name"] for cat in self.categories_sorted
        }

        # COCO category_id -> YOLO class_id (0-based)
        self.catid_to_yoloid: Dict[int, int] = {
            cat["id"]: idx for idx, cat in enumerate(self.categories_sorted)
        }

        # YOLO class names
        self.class_names: List[str] = [cat["name"] for cat in self.categories_sorted]

        # image_id -> image_info
        self.image_id_to_info: Dict[int, dict] = {
            img["id"]: img for img in self.images
        }

        print("=" * 60)
        print("类别映射（COCO -> YOLO）")
        print("=" * 60)
        for cat in self.categories_sorted:
            coco_id = cat["id"]
            yolo_id = self.catid_to_yoloid[coco_id]
            print(f"COCO {coco_id:>3} -> YOLO {yolo_id:>3} : {cat['name']}")
        print("=" * 60)

    @staticmethod
    def normalize_polygon(
        polygon: List[float],
        img_width: int,
        img_height: int
    ) -> Optional[List[float]]:
        """
        将 polygon 归一化到 [0,1]

        Args:
            polygon: [x1, y1, x2, y2, ...]
            img_width: 图像宽
            img_height: 图像高

        Returns:
            归一化后的 polygon，若无效则返回 None
        """
        if not polygon or len(polygon) < 6:
            return None
        if len(polygon) % 2 != 0:
            return None
        if img_width <= 0 or img_height <= 0:
            return None

        out = []
        for i in range(0, len(polygon), 2):
            x = polygon[i]
            y = polygon[i + 1]

            x_norm = x / img_width
            y_norm = y / img_height

            # 裁剪到 [0,1]
            x_norm = min(max(x_norm, 0.0), 1.0)
            y_norm = min(max(y_norm, 0.0), 1.0)

            out.extend([x_norm, y_norm])

        # 至少 3 个点
        if len(out) < 6:
            return None

        return out

    @staticmethod
    def segmentation_to_polygons(segmentation) -> List[List[float]]:
        """
        将 COCO segmentation 转为 polygon 列表

        COCO 常见情况：
        1. segmentation = [[x1,y1,...], [x1,y1,...], ...]
        2. segmentation = [x1,y1,...]   （少见，但兼容）
        3. segmentation 为 RLE（dict） -> 这里不处理，直接跳过

        Returns:
            polygon list
        """
        if segmentation is None:
            return []

        # RLE，不在这个脚本里处理
        if isinstance(segmentation, dict):
            return []

        # 单个 polygon
        if isinstance(segmentation, list) and segmentation and all(
            isinstance(v, (int, float)) for v in segmentation
        ):
            return [segmentation]

        # 多个 polygon
        if isinstance(segmentation, list):
            polys = []
            for item in segmentation:
                if isinstance(item, list) and len(item) >= 6:
                    polys.append(item)
            return polys

        return []

    def convert_single_image(self, image_id: int) -> str:
        """
        转换单张图像的所有标注

        Returns:
            该图像对应的 YOLO segmentation 标签文本
        """
        image_info = self.image_id_to_info.get(image_id)
        if image_info is None:
            return ""

        img_width = int(image_info["width"])
        img_height = int(image_info["height"])

        anns = [ann for ann in self.annotations if ann.get("image_id") == image_id]
        if not anns:
            return ""

        yolo_lines: List[str] = []

        for ann in anns:
            if self.skip_crowd and int(ann.get("iscrowd", 0)) == 1:
                continue

            category_id = ann.get("category_id")
            if category_id not in self.catid_to_yoloid:
                continue

            class_id = self.catid_to_yoloid[category_id]
            segmentation = ann.get("segmentation", None)

            polygons = self.segmentation_to_polygons(segmentation)
            if not polygons:
                # 分割任务下，如果没有 polygon，就跳过
                continue

            for poly in polygons:
                norm_poly = self.normalize_polygon(poly, img_width, img_height)
                if norm_poly is None:
                    continue

                # 至少 3 个点
                num_points = len(norm_poly) // 2
                if num_points < self.min_points:
                    continue

                line = f"{class_id}"
                for coord in norm_poly:
                    line += f" {coord:.6f}"

                yolo_lines.append(line)

        return "\n".join(yolo_lines)

    def convert_all(self):
        """
        转换全部图像标注
        """
        print(f"\n开始转换，共 {len(self.images)} 张图像...")

        converted_count = 0
        empty_count = 0

        for image_info in self.images:
            image_id = image_info["id"]
            file_name = image_info["file_name"]
            stem = Path(file_name).stem

            yolo_text = self.convert_single_image(image_id)
            output_file = self.output_dir / f"{stem}.txt"

            if yolo_text.strip():
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(yolo_text)
                converted_count += 1
                print(f"✓ 转换完成: {output_file.name}")
            else:
                # 对于无有效 segmentation 的图，也生成空文件更稳妥
                output_file.touch(exist_ok=True)
                empty_count += 1
                print(f"- 无有效分割标注: {output_file.name}")

        print("\n转换完成")
        print(f"有效标签文件: {converted_count}")
        print(f"空标签文件: {empty_count}")

        self.generate_classes_file()
        self.generate_data_yaml()

    def generate_classes_file(self):
        """
        生成 classes.txt
        """
        classes_file = self.output_dir / "classes.txt"
        with open(classes_file, "w", encoding="utf-8") as f:
            for name in self.class_names:
                f.write(f"{name}\n")
        print(f"✓ 类别文件已生成: {classes_file}")

    def generate_data_yaml(self):
        """
        生成适配你项目的 data.yaml

        注意：
        你的 train.py 直接读取 data.yaml。:contentReference[oaicite:4]{index=4}
        你的 split_dataset.py 会把图像/标签整理到：
        dataset/train/images
        dataset/train/labels
        dataset/val/images
        dataset/val/labels
        所以这里按这个结构写。:contentReference[oaicite:5]{index=5}
        """
        yaml_path = Path("data.yaml")

        names_str = ", ".join([f"'{n}'" for n in self.class_names])

        content = f"""# 自动生成的数据集配置
# 适用于 Ultralytics YOLO segmentation

path: .
train: dataset/train/images
val: dataset/val/images

nc: {len(self.class_names)}
names: [{names_str}]
"""

        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"✓ data.yaml 已生成: {yaml_path.resolve()}")

    def print_summary(self):
        """
        输出摘要
        """
        print("\n" + "=" * 60)
        print("转换摘要")
        print("=" * 60)
        print(f"输入 JSON: {self.coco_json_path}")
        print(f"输出标签目录: {self.output_dir}")
        print(f"图像数量: {len(self.images)}")
        print(f"标注数量: {len(self.annotations)}")
        print(f"类别数量: {len(self.class_names)}")
        print("类别列表:")
        for idx, name in enumerate(self.class_names):
            print(f"  {idx}: {name}")
        print("=" * 60)


def main():
    print("COCO -> YOLO Segmentation 转换器")
    print("=" * 60)

    # 你项目当前默认路径
    coco_json_path = "labels.json"
    output_dir = "yolo_labels"

    if not os.path.exists(coco_json_path):
        print(f"错误: 找不到文件 {coco_json_path}")
        return

    try:
        converter = CocoToYoloSegConverter(
            coco_json_path=coco_json_path,
            output_dir=output_dir,
            min_points=3,
            skip_crowd=True,
        )
        converter.convert_all()
        converter.print_summary()

        print("\n✅ 转换成功完成")
        print(f"📂 YOLO 分割标签目录: {output_dir}/")
        print("📄 已同时生成 classes.txt 和 data.yaml")

    except Exception as e:
        print(f"❌ 转换过程中出现错误: {e}")
        raise


if __name__ == "__main__":
    main()