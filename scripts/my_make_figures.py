import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# 1. 加载预测数据 (使用您修正后的路径)
data = np.load('../outputs/effnet_b4_baseline/predictions.npz')
y_true = data['y_true']
y_prob = data['y_prob']

# 2. 将真实标签进行 One-Hot 编码处理 (5个类别)
n_classes = 5
y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])

# 3. 设置绘图参数 (强制设置纯白背景)
plt.figure(figsize=(10, 8), facecolor='white')

# 采用强对比度的色系：亮红、深蓝、青绿、亮橙、深红
colors = ['#E63946', '#1D3557', '#2A9D8F', '#F4A261', '#9B2226']
class_names = ['Grade 0 (No DR)', 'Grade 1 (Mild)',
               'Grade 2 (Moderate)', 'Grade 3 (Severe)',
               'Grade 4 (Proliferative)']

# 4. 遍历每个类别，计算并绘制 ROC 曲线
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    # lw (line width) 设置为 1.2，使线条更加精细
    plt.plot(fpr, tpr, color=colors[i], lw=1.2,
             label=f'{class_names[i]} (AUC = {roc_auc:.4f})')

# 5. 绘制对角线并设置图表样式 (对角线改为黑色虚线且变细)
plt.plot([0, 1], [0, 1], color='black', linestyle='--', lw=1.2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

# 坐标轴和标题字体加粗，增强阅读对比度
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('Multiclass ROC Curve (One-vs-Rest)', fontsize=14, fontweight='bold')

# 设置图例不透明 (framealpha=1.0)
plt.legend(loc="lower right", fontsize=11, framealpha=1.0)

# 设置高对比度点状网格线
plt.grid(color='gray', linestyle=':', alpha=0.6)
plt.tight_layout()

# 6. 保存高清图片供 PPT 使用 (关闭透明背景，强制白底)
plt.savefig('../outputs/effnet_b4_baseline/figures/multiclass_roc_curve.png',
            dpi=300, facecolor='white', transparent=False)
print("多分类 ROC 曲线已成功保存！")