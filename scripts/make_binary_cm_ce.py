import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


def generate_binary_confusion_matrix():
    # 1. 载入 effnet_b4_ce 实验的 5x5 混淆矩阵数据
    input_path = '../outputs/effnet_b4_ce/test_metrics.json'

    with open(input_path, 'r') as f:
        data = json.load(f)
    cm_5x5 = np.array(data['confusion'])

    # 2. 划分为 2x2 Referable DR 混淆矩阵 (阈值为 2)
    tn = cm_5x5[:2, :2].sum()
    fp = cm_5x5[:2, 2:].sum()
    fn = cm_5x5[2:, :2].sum()
    tp = cm_5x5[2:, 2:].sum()

    cm_2x2 = np.array([[tn, fp],
                       [fn, tp]])

    # 3. 计算行归一化比例
    cm_norm = cm_2x2.astype('float') / cm_2x2.sum(axis=1, keepdims=True)

    # 4. 绘制 2x2 混淆矩阵
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
    labels = ['Non-referable DR\n(Grade 0-1)', 'Referable DR\n(Grade 2-4)']

    sns.heatmap(cm_norm, annot=cm_2x2, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar_kws={'label': 'Row-normalised fraction'}, vmin=0, vmax=1)

    ax.set_xlabel('Predicted Decision', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Clinical Status', fontsize=11, fontweight='bold')
    ax.set_title('Binary Referable-DR Confusion Matrix (CE Loss)', fontsize=12, fontweight='bold')

    plt.tight_layout()

    # 5. 保存到指定路径
    output_dir = '../outputs/effnet_b4_ce/figures'
    os.makedirs(output_dir, exist_ok=True)

    output_path = f'{output_dir}/binary_referable_cm.png'
    plt.savefig(output_path, dpi=300, facecolor='white')
    plt.close(fig)

    print(f"✅ 2x2 Referable DR 混淆矩阵已保存至:\n{output_path}")


if __name__ == '__main__':
    generate_binary_confusion_matrix()