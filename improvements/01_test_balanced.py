#!/usr/bin/env python3
"""
测试版本：使用10,000个样本测试类别平衡效果
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

print("="*80)
print("测试：类别平衡策略")
print("="*80)

# 导入工具
import sys
sys.path.append(str(Path(__file__).parent.parent))
from muufl_utils import MUUFLDataset
from utils.balanced_loss import compute_class_weights, BalancedCrossEntropyLoss

# 1. 加载数据
print("\n1. 加载数据...")
loader = MUUFLDataset(data_dir="./data")
spectra, labels = loader.prepare_dataset(
    normalize=True,
    remap_labels=True
)

# 使用10,000个样本进行测试
print(f"原始数据集: {spectra.shape}")
n_samples = 10000
indices = np.random.choice(len(spectra), n_samples, replace=False)
spectra = spectra[indices]
labels = labels[indices]
print(f"测试样本: {spectra.shape}")

# 2. 计算类别权重
print("\n2. 计算类别权重...")
class_weights = compute_class_weights(labels, method='balanced')

# 3. ESN编码
print("\n3. ESN编码...")

rng = np.random.RandomState(42)
torch.manual_seed(42)
np.random.seed(42)

in_dim = spectra.shape[1]
rff_dim = 256
res_dim = 200
sr = 0.9
leak = 0.8

print(f"ESN配置: RFF={rff_dim}, 储备池={res_dim}")

# 构建RFF和ESN
W_rff = rng.randn(in_dim, rff_dim).astype(np.float32)
b_rff = rng.uniform(0, 2 * np.pi, size=rff_dim).astype(np.float32)
scale = np.float32(np.sqrt(2.0 / rff_dim))

W_in = rng.randn(res_dim, rff_dim).astype(np.float32)
W = rng.randn(res_dim, res_dim).astype(np.float32)
eigmax = float(np.abs(np.linalg.eigvals(W)).max())
W *= np.float32(sr / (eigmax + 1e-12))

def encode_batch(batch_spectra):
    batch_size = len(batch_spectra)
    phi = scale * np.cos(batch_spectra @ W_rff + b_rff)
    
    states = np.zeros((batch_size, res_dim), dtype=np.float32)
    for i in range(batch_size):
        h = np.tanh(W_in @ phi[i] + W @ states[i])
        states[i] = (1.0 - leak) * states[i] + leak * h
    
    return states

# 编码
print(f"编码 {len(spectra)} 个样本...")
batch_size = 500
feats = []

for i in range(0, len(spectra), batch_size):
    batch = spectra[i:i+batch_size]
    batch_feats = encode_batch(batch)
    feats.append(batch_feats)
    
    if (i // batch_size) % 5 == 0:
        print(f"  进度: {min(i+batch_size, len(spectra))}/{len(spectra)}")

X = np.vstack(feats)
y = labels
print(f"编码完成: {X.shape}")

# 4. 数据划分
print("\n4. 数据划分...")
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"训练集: {X_tr.shape}")
print(f"验证集: {X_val.shape}")

# 5. 模型定义
print("\n5. 构建模型...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=11):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

model = MLP(in_dim=X.shape[1], hidden=128, out_dim=11).to(device)

# 转换为张量
X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_tr_tensor = torch.tensor(y_tr, dtype=torch.long).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

# 6. 对比实验：平衡损失 vs 普通损失
print("\n6. 对比实验：平衡损失 vs 普通损失")

# 实验1：普通交叉熵损失
print("\n--- 实验1: 普通交叉熵损失 ---")
model1 = MLP(in_dim=X.shape[1], hidden=128, out_dim=11).to(device)
criterion1 = nn.CrossEntropyLoss()
optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)

epochs = 20
batch_size = 64

for epoch in range(epochs):
    model1.train()
    indices = torch.randperm(len(X_tr_tensor))
    X_shuffled = X_tr_tensor[indices]
    y_shuffled = y_tr_tensor[indices]
    
    epoch_loss = 0
    correct = 0
    total = 0
    
    for i in range(0, len(X_shuffled), batch_size):
        batch_X = X_shuffled[i:i+batch_size]
        batch_y = y_shuffled[i:i+batch_size]
        
        optimizer1.zero_grad()
        outputs = model1(batch_X)
        loss = criterion1(outputs, batch_y)
        loss.backward()
        optimizer1.step()
        
        epoch_loss += loss.item() * len(batch_X)
        _, predicted = outputs.max(1)
        correct += predicted.eq(batch_y).sum().item()
        total += len(batch_X)
    
    train_acc = 100. * correct / total
    
    # 验证
    model1.eval()
    with torch.no_grad():
        val_outputs = model1(X_val_tensor)
        _, val_predicted = val_outputs.max(1)
        val_correct = val_predicted.eq(y_val_tensor).sum().item()
        val_acc = 100. * val_correct / len(y_val_tensor)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1:3d}/{epochs}] | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

# 最终评估
model1.eval()
with torch.no_grad():
    outputs = model1(X_val_tensor)
    _, predicted = outputs.max(1)
    y_pred1 = predicted.cpu().numpy()

from sklearn.metrics import accuracy_score
acc1 = accuracy_score(y_val, y_pred1) * 100
print(f"普通损失最终准确率: {acc1:.2f}%")

# 实验2：平衡交叉熵损失
print("\n--- 实验2: 平衡交叉熵损失 ---")
model2 = MLP(in_dim=X.shape[1], hidden=128, out_dim=11).to(device)
criterion2 = BalancedCrossEntropyLoss(class_weights=class_weights.to(device))
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)

for epoch in range(epochs):
    model2.train()
    indices = torch.randperm(len(X_tr_tensor))
    X_shuffled = X_tr_tensor[indices]
    y_shuffled = y_tr_tensor[indices]
    
    epoch_loss = 0
    correct = 0
    total = 0
    
    for i in range(0, len(X_shuffled), batch_size):
        batch_X = X_shuffled[i:i+batch_size]
        batch_y = y_shuffled[i:i+batch_size]
        
        optimizer2.zero_grad()
        outputs = model2(batch_X)
        loss = criterion2(outputs, batch_y)
        loss.backward()
        optimizer2.step()
        
        epoch_loss += loss.item() * len(batch_X)
        _, predicted = outputs.max(1)
        correct += predicted.eq(batch_y).sum().item()
        total += len(batch_X)
    
    train_acc = 100. * correct / total
    
    # 验证
    model2.eval()
    with torch.no_grad():
        val_outputs = model2(X_val_tensor)
        _, val_predicted = val_outputs.max(1)
        val_correct = val_predicted.eq(y_val_tensor).sum().item()
        val_acc = 100. * val_correct / len(y_val_tensor)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1:3d}/{epochs}] | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

model2.eval()
with torch.no_grad():
    outputs = model2(X_val_tensor)
    _, predicted = outputs.max(1)
    y_pred2 = predicted.cpu().numpy()

acc2 = accuracy_score(y_val, y_pred2) * 100
print(f"平衡损失最终准确率: {acc2:.2f}%")

# 7. 类别级别对比
print("\n7. 类别级别对比分析:")

class_names = loader.CLASS_NAMES
print(f"\n{'类别':<15} {'普通损失':<10} {'平衡损失':<10} {'提升':<10}")
print("-" * 45)

for i in range(11):
    mask = y_val == i
    if mask.sum() > 0:
        acc_normal = (y_pred1[mask] == y_val[mask]).mean() * 100
        acc_balanced = (y_pred2[mask] == y_val[mask]).mean() * 100
        improvement = acc_balanced - acc_normal
        class_name = class_names.get(i+1, f"class_{i}")
        
        marker = "↑" if improvement > 0 else "↓" if improvement < 0 else "="
        print(f"{class_name:<15} {acc_normal:>8.1f}% {acc_balanced:>9.1f}% {improvement:>8.1f}% {marker}")

# 8. 少数类别详细分析
print("\n8. 少数类别详细分析:")
minority_classes = ['yellowcurb', 'cloth', 'sidewalk']
for cls_name in minority_classes:
    cls_id = None
    for k, v in class_names.items():
        if v == cls_name:
            cls_id = k-1
            break
    
    if cls_id is not None:
        mask = y_val == cls_id
        if mask.sum() > 0:
            acc_normal = (y_pred1[mask] == y_val[mask]).mean() * 100
            acc_balanced = (y_pred2[mask] == y_val[mask]).mean() * 100
            improvement = acc_balanced - acc_normal
            print(f"  {cls_name:<15}: 普通={acc_normal:5.1f}%, 平衡={acc_balanced:5.1f}%, 提升={improvement:5.1f}%")

# 9. 总结
print("\n" + "="*80)
print("测试总结:")
print(f"总体准确率 - 普通损失: {acc1:.2f}%")
print(f"总体准确率 - 平衡损失: {acc2:.2f}%")
print(f"总体提升: {acc2-acc1:.2f}%")

if acc2 > acc1:
    print(f"✅ 类别平衡策略有效! 提升 {acc2-acc1:.2f}%")
else:
    print(f"⚠️  类别平衡策略未显示明显优势")

print("="*80)