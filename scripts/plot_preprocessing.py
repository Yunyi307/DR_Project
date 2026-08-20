from __future__ import annotations
import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

# 将上级目录（code/）加入系统路径，以便正确导入 src 模块
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.preprocessing import crop_to_roi, apply_clahe, ben_graham, circular_mask


def generate_pipeline_figure():
    # 使用测试集图片的相对路径
    image_path = "../data/aptos2019/test_images/0ca261d6e31d.png"

    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"无法读取图片，请检查路径: {image_path}")

    orig_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 逐步提取预处理管道的各个中间状态
    img_a = orig_rgb.copy()  # (a) 原始图
    img_b = crop_to_roi(orig_rgb)  # (b) ROI 裁剪去黑边
    img_c = apply_clahe(img_b)  # (c) + CLAHE 局部对比度增强
    img_d = ben_graham(img_c)  # (d) + Ben Graham 光照归一化
    img_d = circular_mask(img_d)  # 圆形掩膜处理

    # 调整画布高度并增加顶部空间，彻底解决子标题显示不全的问题
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    stages = [
        (img_a, "(a) Original"),
        (img_b, "(b) Cropped ROI"),
        (img_c, "(c) + CLAHE"),
        (img_d, "(d) + Ben Graham")
    ]

    for ax, (img, title) in zip(axes, stages):
        ax.imshow(img)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.axis("off")

    # 优化整体布局内边距，确保保存时不切除边缘文字
    plt.tight_layout(pad=2.0)

    output_filename = "preprocessing_stages.jpg"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"对比图已成功生成并保存至当前目录: {output_filename}")
    plt.show()


if __name__ == "__main__":
    generate_pipeline_figure()