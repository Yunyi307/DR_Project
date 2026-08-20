import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def generate_dr_only_confusion_matrix(
    exp_name, title_suffix, json_name='test_metrics.json'
):
  # 拼接实际的 JSON 路径
  input_path = f'../outputs/{exp_name}/{json_name}'

  if not os.path.exists(input_path):
    print(f'❌ 找不到文件: {input_path}')
    return

  with open(input_path, 'r') as f:
    data = json.load(f)

  # 提取原始 5x5 混淆矩阵
  cm_5x5 = np.array(data['confusion'])

  # 剔除第 0 行和第 0 列 (No DR)，仅保留 1-4 级 (Mild, Moderate, Severe, Proliferative)
  cm_4x4 = cm_5x5[1:, 1:]

  # 计算行归一化比例
  row_sums = cm_4x4.sum(axis=1, keepdims=True)
  row_sums[row_sums == 0] = 1
  cm_norm = cm_4x4.astype('float') / row_sums

  # 绘制 4x4 热力图
  fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300, facecolor='white')
  labels = [
      'Mild\n(Grade 1)',
      'Moderate\n(Grade 2)',
      'Severe\n(Grade 3)',
      'Proliferative\n(Grade 4)',
  ]

  sns.heatmap(
      cm_norm,
      annot=cm_4x4,
      fmt='d',
      cmap='Blues',
      xticklabels=labels,
      yticklabels=labels,
      ax=ax,
      cbar_kws={'label': 'Row-normalised fraction'},
      vmin=0,
      vmax=1,
      annot_kws={'size': 11, 'weight': 'bold'},
  )

  ax.set_xlabel('Predicted Grade', fontsize=11, fontweight='bold')
  ax.set_ylabel('True Grade', fontsize=11, fontweight='bold')
  ax.set_title(
      f'DR-Only (Grade 1-4) Confusion Matrix ({title_suffix})',
      fontsize=12,
      fontweight='bold',
      pad=12,
  )

  plt.tight_layout()

  # 保存图片至对应的 figures 目录下
  output_dir = f'../outputs/{exp_name}/figures'
  os.makedirs(output_dir, exist_ok=True)

  output_path = f'{output_dir}/dr_only_4x4_cm.png'
  plt.savefig(output_path, dpi=300, facecolor='white')
  plt.close(fig)

  print(f'✅ [{exp_name}] 的 4x4 剔除0级混淆矩阵已保存至:\n{output_path}')


if __name__ == '__main__':
  # 针对你截图中的文件夹和实际文件名进行调整
  generate_dr_only_confusion_matrix(
      exp_name='effnet_b4_weighted_ce_5fold_5fold',
      title_suffix='5-Fold Ensemble',
      json_name='test_metrics_5fold.json',  # 👈 这里改成了你目录里实际存在的文件名
  )