import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(8, 6))
# 1. 从文件中读取FPR和TPR数据
file_path = "D:/yuxia/test/20230817/iDNA_ABT-test/train/h_k_roc_curve_data.txt"  # 替换为包含FPR和TPR数据的文件路径
with open(file_path, "r") as file:
    lines = file.readlines()
fpr = []
tpr = []
for line in lines:
    parts = line.strip().split(", ")  # 假设数据格式是 "FPR: 0.0, TPR: 0.0"
    fpr_value = float(parts[0].split(": ")[1])
    tpr_value = float(parts[1].split(": ")[1])
    fpr.append(fpr_value)
    tpr.append(tpr_value)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', lw=2, label='h_k (area = %0.2f)' % roc_auc)

# 1. 从文件中读取FPR和TPR数据
file_path = "D:/yuxia/test/20230817/iDNA_ABT-test/train/h_b_roc_curve_data.txt"  # 替换为包含FPR和TPR数据的文件路径
with open(file_path, "r") as file:
    lines = file.readlines()
fpr = []
tpr = []
for line in lines:
    parts = line.strip().split(", ")  # 假设数据格式是 "FPR: 0.0, TPR: 0.0"
    fpr_value = float(parts[0].split(": ")[1])
    tpr_value = float(parts[1].split(": ")[1])
    fpr.append(fpr_value)
    tpr.append(tpr_value)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkred', lw=2, label='h_b (area = %0.2f)' % roc_auc)

# 1. 从文件中读取FPR和TPR数据
file_path = "D:/yuxia/test/20230817/iDNA_ABT-test/train/h_l_roc_curve_data.txt"  # 替换为包含FPR和TPR数据的文件路径
with open(file_path, "r") as file:
    lines = file.readlines()
fpr = []
tpr = []
for line in lines:
    parts = line.strip().split(", ")  # 假设数据格式是 "FPR: 0.0, TPR: 0.0"
    fpr_value = float(parts[0].split(": ")[1])
    tpr_value = float(parts[1].split(": ")[1])
    fpr.append(fpr_value)
    tpr.append(tpr_value)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='purple', lw=2, label='h_l (area = %0.2f)' % roc_auc)

# 1. 从文件中读取FPR和TPR数据
file_path = "D:/yuxia/test/20230817/iDNA_ABT-test/train/m_b_roc_curve_data.txt"  # 替换为包含FPR和TPR数据的文件路径
with open(file_path, "r") as file:
    lines = file.readlines()
fpr = []
tpr = []
for line in lines:
    parts = line.strip().split(", ")  # 假设数据格式是 "FPR: 0.0, TPR: 0.0"
    fpr_value = float(parts[0].split(": ")[1])
    tpr_value = float(parts[1].split(": ")[1])
    fpr.append(fpr_value)
    tpr.append(tpr_value)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='black', lw=2, label='m_b (area = %0.2f)' % roc_auc)

# 1. 从文件中读取FPR和TPR数据
file_path = "D:/yuxia/test/20230817/iDNA_ABT-test/train/m_h_roc_curve_data.txt"  # 替换为包含FPR和TPR数据的文件路径
with open(file_path, "r") as file:
    lines = file.readlines()
fpr = []
tpr = []
for line in lines:
    parts = line.strip().split(", ")  # 假设数据格式是 "FPR: 0.0, TPR: 0.0"
    fpr_value = float(parts[0].split(": ")[1])
    tpr_value = float(parts[1].split(": ")[1])
    fpr.append(fpr_value)
    tpr.append(tpr_value)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='cyan', lw=2, label='m_h (area = %0.2f)' % roc_auc)

# 1. 从文件中读取FPR和TPR数据
file_path = "D:/yuxia/test/20230817/iDNA_ABT-test/train/m_k_roc_curve_data.txt"  # 替换为包含FPR和TPR数据的文件路径
with open(file_path, "r") as file:
    lines = file.readlines()
fpr = []
tpr = []
for line in lines:
    parts = line.strip().split(", ")  # 假设数据格式是 "FPR: 0.0, TPR: 0.0"
    fpr_value = float(parts[0].split(": ")[1])
    tpr_value = float(parts[1].split(": ")[1])
    fpr.append(fpr_value)
    tpr.append(tpr_value)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='magenta', lw=2, label='m_k (area = %0.2f)' % roc_auc)

# 1. 从文件中读取FPR和TPR数据
file_path = "D:/yuxia/test/20230817/iDNA_ABT-test/train/m_l_roc_curve_data.txt"  # 替换为包含FPR和TPR数据的文件路径
with open(file_path, "r") as file:
    lines = file.readlines()
fpr = []
tpr = []
for line in lines:
    parts = line.strip().split(", ")  # 假设数据格式是 "FPR: 0.0, TPR: 0.0"
    fpr_value = float(parts[0].split(": ")[1])
    tpr_value = float(parts[1].split(": ")[1])
    fpr.append(fpr_value)
    tpr.append(tpr_value)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='yellow', lw=2, label='m_l (area = %0.2f)' % roc_auc)

# 1. 从文件中读取FPR和TPR数据
file_path = "D:/yuxia/test/20230817/iDNA_ABT-test/train/m_t_roc_curve_data.txt"  # 替换为包含FPR和TPR数据的文件路径
with open(file_path, "r") as file:
    lines = file.readlines()
fpr = []
tpr = []
for line in lines:
    parts = line.strip().split(", ")  # 假设数据格式是 "FPR: 0.0, TPR: 0.0"
    fpr_value = float(parts[0].split(": ")[1])
    tpr_value = float(parts[1].split(": ")[1])
    fpr.append(fpr_value)
    tpr.append(tpr_value)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='red', lw=2, label='m_t (area = %0.2f)' % roc_auc)

# 1. 从文件中读取FPR和TPR数据
file_path = "D:/yuxia/test/20230817/iDNA_ABT-test/train/r_b_roc_curve_data.txt"  # 替换为包含FPR和TPR数据的文件路径
with open(file_path, "r") as file:
    lines = file.readlines()
fpr = []
tpr = []
for line in lines:
    parts = line.strip().split(", ")  # 假设数据格式是 "FPR: 0.0, TPR: 0.0"
    fpr_value = float(parts[0].split(": ")[1])
    tpr_value = float(parts[1].split(": ")[1])
    fpr.append(fpr_value)
    tpr.append(tpr_value)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='green', lw=2, label='r_b (area = %0.2f)' % roc_auc)

# 1. 从文件中读取FPR和TPR数据
file_path = "D:/yuxia/test/20230817/iDNA_ABT-test/train/r_k_roc_curve_data.txt"  # 替换为包含FPR和TPR数据的文件路径
with open(file_path, "r") as file:
    lines = file.readlines()
fpr = []
tpr = []
for line in lines:
    parts = line.strip().split(", ")  # 假设数据格式是 "FPR: 0.0, TPR: 0.0"
    fpr_value = float(parts[0].split(": ")[1])
    tpr_value = float(parts[1].split(": ")[1])
    fpr.append(fpr_value)
    tpr.append(tpr_value)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkblue', lw=2, label='r_k (area = %0.2f)' % roc_auc)

# 1. 从文件中读取FPR和TPR数据
file_path = "D:/yuxia/test/20230817/iDNA_ABT-test/train/r_l_roc_curve_data.txt"  # 替换为包含FPR和TPR数据的文件路径
with open(file_path, "r") as file:
    lines = file.readlines()
fpr = []
tpr = []
for line in lines:
    parts = line.strip().split(", ")  # 假设数据格式是 "FPR: 0.0, TPR: 0.0"
    fpr_value = float(parts[0].split(": ")[1])
    tpr_value = float(parts[1].split(": ")[1])
    fpr.append(fpr_value)
    tpr.append(tpr_value)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkgreen', lw=2, label='h_b (area = %0.2f)' % roc_auc)


plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.savefig('RNA_ROC_curve.png', dpi=300, bbox_inches='tight')
plt.show()
