import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


def generate_binary_confusion_matrix():
    # 1. 载入您的 5x5 混淆矩阵数据
    input_path = '../outputs/effnet_b4_baseline/test_metrics.json'

    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
        cm_5x5 = np.array(data['confusion'])
    except FileNotFoundError:
        print(f"❌ 找不到文件: {input_path}。请确保路径正确且模型已完成测试评估。")
        return

    # 2. 划分为 2x2 Referable DR 混淆矩阵 (阈值为 2：0-1 为无需转诊，2-4 为需转诊)[cite: 9]
    # tn: True Non-referable predicted as Non-referable
    # fp: True Non-referable predicted as Referable
    # fn: True Referable predicted as Non-referable
    # tp: True Referable predicted as Referable
    tn = cm_5x5[:2, :2].sum()
    fp = cm_5x5[:2, 2:].sum()
    fn = cm_5x5[2:, :2].sum()
    tp = cm_5x5[2:, 2:].sum()

    cm_2x2 = np.array([[tn, fp],
                       [fn, tp]])

    # 3. 计算行归一化比例 (用于颜色映射)
    cm_norm = cm_2x2.astype('float') / cm_2x2.sum(axis=1, keepdims=True)

    # 4. 绘制 2x2 混淆矩阵
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
    labels = ['Non-referable DR\n(Grade 0-1)', 'Referable DR\n(Grade 2-4)']

    # 使用 Seaborn 绘制热力图
    sns.heatmap(cm_norm, annot=cm_2x2, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar_kws={'label': 'Row-normalised fraction'}, vmin=0, vmax=1)

    # 优化坐标轴和标题样式
    ax.set_xlabel('Predicted Decision', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Clinical Status', fontsize=11, fontweight='bold')
    ax.set_title('Binary Referable-DR Confusion Matrix', fontsize=12, fontweight='bold')

    plt.tight_layout()

    # 5. 保存到您指定的路径，并确保目录存在
    output_dir = '../outputs/effnet_b4_baseline/figures'
    os.makedirs(output_dir, exist_ok=True)

    output_path = f'{output_dir}/binary_referable_confusion_matrix.png'
    plt.savefig(output_path, dpi=300, facecolor='white')
    plt.close(fig)

    print(f"✅ 2x2 Referable DR 混淆矩阵已成功保存至:\n{output_path}")


if __name__ == '__main__':
    generate_binary_confusion_matrix()