#!/usr/bin/env python3
"""
简单改进：类别平衡 + 完整数据集
分步实施，确保每一步都成功
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

print("="*80)
print("简单改进：类别平衡 + 完整数据集")
print("="*80)

# 1. 导入工具
print("\n1. 导入工具...")
import sys
sys.path.append(str(Path(__file__).parent.parent))
from muufl_utils import MUUFLDataset

# 2. 加载完整数据集
print("\n2. 加载完整数据集...")
loader = MUUFLDataset(data_dir="./data")
spectra, labels = loader.prepare_dataset(
    normalize=True,
    remap_labels=True
)

print(f"完整数据集:")
print(f"  光谱数据: {spectra.shape}")
print(f"  标签: {labels.shape}")

# 显示类别分布
unique, counts = np.unique(labels, return_counts=True)
print(f"\n类别分布:")
for cls, count in zip(unique, counts):
    class_name = loader.CLASS_NAMES.get(cls+1, f"class_{cls}")
    percentage = count / len(labels) * 100
    print(f"  {class_name:15s}: {count:6d} ({percentage:5.1f}%)")

# 3. 计算简单的类别权重
print("\n3. 计算类别权重...")
# 方法1：反比于类别频率
class_weights = 1.0 / counts
class_weights = class_weights / class_weights.sum() * len(unique)

print(f"类别权重:")
for cls, weight in zip(unique, class_weights):
    class_name = loader.CLASS_NAMES.get(cls+1, f"class_{cls}")
    print(f"  {class_name:15s}: 权重={weight:.3f}")

# 转换为PyTorch张量
weight_tensor = torch.zeros(max(unique) + 1, dtype=torch.float32)
for cls, weight in zip(unique, class_weights):
    weight_tensor[cls] = weight

# 4. 使用小样本测试平衡效果（先验证）
print("\n4. 使用小样本测试平衡效果...")
# 取前5000个样本进行快速测试
test_size = 5000
test_spectra = spectra[:test_size]
test_labels = labels[:test_size]

print(f"测试样本: {test_spectra.shape}")

# 5. 简单的ESN编码（使用之前的配置）
print("\n5. ESN编码...")

# 设置随机种子
rng = np.random.RandomState(42)
torch.manual_seed(42)
np.random.seed(42)

# ESN参数
in_dim = test_spectra.shape[1]
rff_dim = 256
res_dim = 200
sr = 0.9
leak = 0.8

print(f"ESN配置:")
print(f"  输入维度: {in_dim}")
print(f"  RFF维度: {rff_dim}")
print(f"  储备池: {res_dim}")

# 构建RFF
W_rff = rng.randn(in_dim, rff_dim).astype(np.float32)
b_rff = rng.uniform(0, 2 * np.pi, size=rff_dim).astype(np.float32)
scale = np.float32(np.sqrt(2.0 / rff_dim))

# 构建ESN
W_in = rng.randn(res_dim, rff_dim).astype(np.float32)
W = rng.randn(res_dim, res_dim).astype(np.float32)
eigmax = float(np.abs(np.linalg.eigvals(W)).max())
W *= np.float32(sr / (eigmax + 1e-12))

# 编码函数
def encode_batch(batch_spectra):
    batch_size = len(batch_spectra)
    phi = scale * np.cos(batch_spectra @ W_rff + b_rff)
    
    states = np.zeros((batch_size, res_dim), dtype=np.float32)
    for i in range(batch_size):
        h = np.tanh(W_in @ phi[i] + W @ states[i])
        states[i] = (1.0 - leak) * states[i] + leak * h
    
    return states

# 编码测试样本
print(f"编码 {len(test_spectra)} 个样本...")
batch_size = 500
feats = []

for i in range(0, len(test_spectra), batch_size):
    batch = test_spectra[i:i+batch_size]
    batch_feats = encode_batch(batch)
    feats.append(batch_feats)
    
    if (i // batch_size) % 5 == 0:
        print(f"  进度: {min(i+batch_size, len(test_spectra))}/{len(test_spectra)}")

X = np.vstack(feats)
y = test_labels
print(f"编码完成: {X.shape}")

# 6. 数据划分
print("\n6. 数据划分...")
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"训练集: {X_tr.shape}")
print(f"验证集: {X_val.shape}")

# 7. 构建MLP模型（带Dropout和BatchNorm）
print("\n7. 构建MLP模型...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class ImprovedMLP(nn.Module):
    def __init__(self, in_dim, hidden1=128, hidden2=64, out_dim=11):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.fc3 = nn.Linear(hidden2, out_dim)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)

model = ImprovedMLP(in_dim=X.shape[1], hidden1=128, hidden2=64, out_dim=11).to(device)
print(f"模型结构: {X.shape[1]} -> 128 -> 64 -> 11")

# 转换为张量
X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_tr_tensor = torch.tensor(y_tr, dtype=torch.long).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

# 8. 训练（使用类别权重）
print("\n8. 训练MLP（使用类别权重）...")

# 使用带权重的交叉熵损失
criterion = nn.CrossEntropyLoss(weight=weight_tensor.to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

epochs = 25
batch_size = 64

for epoch in range(epochs):
    model.train()
    
    indices = torch.randperm(len(X_tr_tensor))
    X_shuffled = X_tr_tensor[indices]
    y_shuffled = y_tr_tensor[indices]
    
    epoch_loss = 0
    correct = 0
    total = 0
    
    for i in range(0, len(X_shuffled), batch_size):
        batch_X = X_shuffled[i:i+batch_size]
        batch_y = y_shuffled[i:i+batch_size]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * len(batch_X)
        _, predicted = outputs.max(1)
        correct += predicted.eq(batch_y).sum().item()
        total += len(batch_X)
    
    train_acc = 100. * correct / total
    train_loss = epoch_loss / total
    
    # 验证
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor).item()
        _, val_predicted = val_outputs.max(1)
        val_correct = val_predicted.eq(y_val_tensor).sum().item()
        val_acc = 100. * val_correct / len(y_val_tensor)
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:3d}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

# 9. 最终评估
print("\n9. 最终评估...")
model.eval()
with torch.no_grad():
    outputs = model(X_val_tensor)
    _, predicted = outputs.max(1)
    
    y_true = y_val_tensor.cpu().numpy()
    y_pred = predicted.cpu().numpy()
    
    from sklearn.metrics import accuracy_score, classification_report
    
    overall_acc = accuracy_score(y_true, y_pred) * 100
    print(f"验证集准确率: {overall_acc:.2f}%")
    
    # 每个类别的准确率
    print("\n每个类别的准确率:")
    class_names = loader.CLASS_NAMES
    class_accuracies = {}
    
    for i in range(11):
        mask = y_true == i
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == y_true[mask]).mean() * 100
            class_name = class_names.get(i+1, f"class_{i}")
            class_accuracies[class_name] = class_acc
            print(f"  {class_name:15s}: {class_acc:6.2f}% ({mask.sum()} samples)")

# 10. 与之前结果对比
print("\n10. 与之前结果对比...")
previous_acc = 84.30  # 之前的ESN结果
majority_acc = 44.3

print(f"\n性能对比:")
print(f"  方法           | 验证准确率 | 相对提升")
print(f"  --------------|------------|----------")
print(f"  多数类(trees)  |   44.3%    |   基准")
print(f"  ESN+MLP(之前) |   84.3%    |   +40.0%")
print(f"  ESN+MLP(改进) |   {overall_acc:.1f}%    |   +{overall_acc-majority_acc:.1f}%")

if overall_acc > previous_acc:
    improvement = overall_acc - previous_acc
    print(f"\n✅ 改进成功! 准确率提升: +{improvement:.1f}%")
else:
    print(f"\n⚠️  准确率变化: {overall_acc-previous_acc:+.1f}%")

# 11. 少数类别特别分析
print("\n11. 少数类别分析:")
minority_classes = ['yellowcurb', 'cloth', 'sidewalk']
minority_improvement = False

for cls in minority_classes:
    if cls in class_accuracies:
        acc = class_accuracies[cls]
        # 之前的准确率（从之前的结果）
        prev_acc = {'yellowcurb': 0.0, 'cloth': 50.0, 'sidewalk': 34.6}.get(cls, 0.0)
        
        if acc > prev_acc:
            improvement = acc - prev_acc
            print(f"  {cls:15s}: {acc:6.2f}% (之前: {prev_acc:.1f}%, 提升: +{improvement:.1f}%) ✅")
            minority_improvement = True
        else:
            print(f"  {cls:15s}: {acc:6.2f}% (之前: {prev_acc:.1f}%, 变化: {acc-prev_acc:+.1f}%)")

if minority_improvement:
    print(f"\n✅ 少数类别有明显改善!")
else:
    print(f"\n⚠️  少数类别需要进一步优化")

# 12. 保存结果
print("\n12. 保存结果...")
cache_dir = Path("./cache/improved")
cache_dir.mkdir(exist_ok=True)

torch.save(model.state_dict(), cache_dir / "improved_mlp_model.pth")
np.save(cache_dir / "improved_features.npy", X)
np.save(cache_dir / "improved_labels.npy", y)

print(f"模型和特征已保存到 {cache_dir}/")

print("\n" + "="*80)
print("简单改进完成!")
print(f"最终验证准确率: {overall_acc:.2f}%")
print("="*80)

# 下一步建议
print("\n下一步建议:")
print("1. 在完整数据集上运行此改进方法")
print("2. 尝试不同的类别权重计算方法")
print("3. 实现数据增强（光谱扰动）")
print("4. 优化ESN超参数")