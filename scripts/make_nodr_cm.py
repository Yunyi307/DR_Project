import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


def generate_nodr_confusion_matrix(exp_name, title_suffix):
    # 1. 读取对应实验的 test_metrics.json
    input_path = f'../outputs/{exp_name}/test_metrics.json'

    if not os.path.exists(input_path):
        print(f"❌ 找不到文件: {input_path}，请确认该实验是否已完成测试！")
        return

    with open(input_path, 'r') as f:
        data = json.load(f)
    cm_5x5 = np.array(data['confusion'])

    # 2. 划分为 No DR (0级) vs Any DR (1-4级) 混淆矩阵[cite: 9, 12]
    # 行/列 0: True/Pred No DR (Grade 0)
    # 行/列 1: True/Pred Any DR (Grade 1-4)
    tn = cm_5x5[0, 0]  # 真实 No DR 预测为 No DR
    fp = cm_5x5[0, 1:].sum()  # 真实 No DR 预测为 Any DR
    fn = cm_5x5[1:, 0].sum()  # 真实 Any DR 预测为 No DR
    tp = cm_5x5[1:, 1:].sum()  # 真实 Any DR 预测为 Any DR

    cm_2x2 = np.array([[tn, fp],
                       [fn, tp]])

    # 3. 计算行归一化比例 (Recall / Specificity)
    cm_norm = cm_2x2.astype('float') / cm_2x2.sum(axis=1, keepdims=True)

    # 4. 绘制热力图
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
    labels = ['No DR\n(Grade 0)', 'Any DR\n(Grade 1-4)']

    sns.heatmap(cm_norm, annot=cm_2x2, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar_kws={'label': 'Row-normalised fraction'}, vmin=0, vmax=1)

    ax.set_xlabel('Predicted Decision', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Clinical Status', fontsize=11, fontweight='bold')
    ax.set_title(f'No DR vs Any DR Confusion Matrix ({title_suffix})', fontsize=12, fontweight='bold')

    plt.tight_layout()

    # 5. 保存图片到对应的 figures 文件夹下
    output_dir = f'../outputs/{exp_name}/figures'
    os.makedirs(output_dir, exist_ok=True)

    output_path = f'{output_dir}/nodr_vs_anydr_cm.png'
    plt.savefig(output_path, dpi=300, facecolor='white')
    plt.close(fig)

    print(f"✅ [{exp_name}] 的 No DR 混淆矩阵已保存至:\n{output_path}")


if __name__ == '__main__':
    # 同时生成 baseline 和 ce 的图表
    generate_nodr_confusion_matrix('effnet_b4_baseline', 'Baseline')
    generate_nodr_confusion_matrix('effnet_b4_ce', 'CE Loss')