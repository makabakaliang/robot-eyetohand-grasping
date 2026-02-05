import importlib.metadata
import platform
import sys


def get_lib_version(lib_name):
    try:
        return importlib.metadata.version(lib_name)
    except importlib.metadata.PackageNotFoundError:
        return "Not Installed"


# 核心代码涉及的关键库
core_libs = [
    "pyrealsense2",  # 相机驱动
    "ultralytics",  # YOLO 分割模型
    "numpy",  # 矩阵运算
    "opencv-python",  # cv2
    "torch",  # YOLO 运行底座
    "torchvision",  # 视觉模型相关
]


def main():
    print("=" * 50)
    print("项目迁移环境依赖报告")
    print("=" * 50)

    # 1. 系统基础信息
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version.split()[0]}")
    print("-" * 30)

    # 2. 核心依赖检测
    print("核心依赖库版本:")
    results = []
    for lib in core_libs:
        # 特殊处理：opencv 的包名可能是 opencv-python 或 opencv-contrib-python
        if lib == "opencv-python":
            version = get_lib_version("opencv-python")
            if version == "Not Installed":
                version = get_lib_version("opencv-contrib-python")
        else:
            version = get_lib_version(lib)

        print(f"  {lib:20}: {version}")
        if version != "Not Installed":
            results.append(f"{lib}=={version}")

    # 3. 生成 requirements.txt 内容预览
    print("-" * 30)
    print("可直接用于 Linux 的安装指令预览:")
    print(f"pip install {' '.join(results)}")
    print("=" * 50)


if __name__ == "__main__":
    main()