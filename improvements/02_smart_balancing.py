#!/usr/bin/env python3
"""
改进2: 智能平衡策略
结合焦点损失和自适应权重
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

print("="*80)
print("改进2: 智能平衡策略")
print("="*80)

# 导入工具
import sys
sys.path.append(str(Path(__file__).parent.parent))
from muufl_utils import MUUFLDataset
from utils.balanced_loss import FocalLoss, compute_class_weights

# 1. 加载数据
print("\n1. 加载数据...")
loader = MUUFLDataset(data_dir="./data")
spectra, labels = loader.prepare_dataset(
    normalize=True,
    remap_labels=True
)

# 使用10,000个样本
n_samples = 10000
indices = np.random.choice(len(spectra), n_samples, replace=False)
spectra = spectra[indices]
labels = labels[indices]
print(f"测试样本: {spectra.shape}")

# 2. ESN编码
print("\n2. ESN编码...")

rng = np.random.RandomState(42)
torch.manual_seed(42)
np.random.seed(42)

in_dim = spectra.shape[1]
rff_dim = 256
res_dim = 200
sr = 0.9
leak = 0.8

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

X = np.vstack(feats)
y = labels
print(f"编码完成: {X.shape}")

# 3. 数据划分
print("\n3. 数据划分...")
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"训练集: {X_tr.shape}")
print(f"验证集: {X_val.shape}")

# 4. 智能平衡策略
print("\n4. 智能平衡策略...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 计算类别权重（多种方法）
weights_balanced = compute_class_weights(y_tr, method='balanced')
weights_sqrt = compute_class_weights(y_tr, method='sqrt')
weights_inverse = compute_class_weights(y_tr, method='inverse')

print(f"\n不同权重计算方法:")
print(f"  平衡权重: 最大/最小 = {weights_balanced.max()/weights_balanced.min():.1f}x")
print(f"  平方根权重: 最大/最小 = {weights_sqrt.max()/weights_sqrt.min():.1f}x")
print(f"  反比权重: 最大/最小 = {weights_inverse.max()/weights_inverse.min():.1f}x")

# 5. 模型定义
class EnhancedMLP(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=11):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)

# 转换为张量
X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_tr_tensor = torch.tensor(y_tr, dtype=torch.long).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

# 6. 多种损失函数对比
print("\n6. 多种损失函数对比实验")

experiments = [
    ("标准交叉熵", nn.CrossEntropyLoss(), None),
    ("平衡交叉熵", nn.CrossEntropyLoss(), weights_balanced.to(device)),
    ("平方根权重", nn.CrossEntropyLoss(), weights_sqrt.to(device)),
    ("Focal Loss (γ=2)", FocalLoss(gamma=2.0), None),
    ("Focal Loss + 权重", FocalLoss(gamma=2.0, alpha=weights_balanced.to(device)), None),
]

results = {}
epochs = 20
batch_size = 64

for exp_name, criterion, weights in experiments:
    print(f"\n--- {exp_name} ---")
    
    model = EnhancedMLP(in_dim=X.shape[1], hidden=128, out_dim=11).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 如果有权重，设置损失函数的权重
    if weights is not None and hasattr(criterion, 'weight'):
        criterion.weight = weights
    
    best_val_acc = 0
    class_accuracies = {}
    
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
            
            if weights is not None and not hasattr(criterion, 'weight'):
                # 手动应用权重
                loss = F.cross_entropy(outputs, batch_y, weight=weights)
            else:
                loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_X)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += len(batch_X)
        
        train_acc = 100. * correct / total
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            _, val_predicted = val_outputs.max(1)
            val_correct = val_predicted.eq(y_val_tensor).sum().item()
            val_acc = 100. * val_correct / len(y_val_tensor)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # 保存最佳模型的预测结果
                y_pred = val_predicted.cpu().numpy()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                  f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
    
    # 计算类别准确率
    y_true = y_val
    class_names = loader.CLASS_NAMES
    
    minority_acc = {}
    for i in range(11):
        mask = y_true == i
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == y_true[mask]).mean() * 100
            class_name = class_names.get(i+1, f"class_{i}")
            if class_name in ['yellowcurb', 'cloth', 'sidewalk']:
                minority_acc[class_name] = class_acc
    
    results[exp_name] = {
        'overall_acc': best_val_acc,
        'minority_acc': minority_acc,
        'y_pred': y_pred
    }
    
    print(f"最终验证准确率: {best_val_acc:.2f}%")
    print(f"少数类别准确率: {minority_acc}")

# 7. 结果分析
print("\n" + "="*80)
print("结果分析")
print("="*80)

print(f"\n{'损失函数':<25} {'总体准确率':<12} {'yellowcurb':<12} {'cloth':<10} {'sidewalk':<10}")
print("-" * 70)

for exp_name, result in results.items():
    overall = result['overall_acc']
    minority = result['minority_acc']
    
    yellowcurb = minority.get('yellowcurb', 0)
    cloth = minority.get('cloth', 0)
    sidewalk = minority.get('sidewalk', 0)
    
    print(f"{exp_name:<25} {overall:>10.2f}% {yellowcurb:>10.1f}% {cloth:>9.1f}% {sidewalk:>9.1f}%")

# 8. 最佳策略选择
print("\n8. 最佳策略推荐:")

# 计算综合得分（总体准确率 + 少数类别提升）
best_overall = max(r['overall_acc'] for r in results.values())
best_minority_avg = max(
    np.mean([r['minority_acc'].get('yellowcurb', 0), 
             r['minority_acc'].get('cloth', 0),
             r['minority_acc'].get('sidewalk', 0)])
    for r in results.values()
)

scores = {}
for exp_name, result in results.items():
    overall = result['overall_acc']
    minority = result['minority_acc']
    
    minority_avg = np.mean([
        minority.get('yellowcurb', 0),
        minority.get('cloth', 0),
        minority.get('sidewalk', 0)
    ])
    
    # 综合得分：70%总体准确率 + 30%少数类别平均准确率
    score = 0.7 * (overall / best_overall) + 0.3 * (minority_avg / best_minority_avg)
    scores[exp_name] = score

# 排序
sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

print(f"\n综合得分排名:")
for i, (exp_name, score) in enumerate(sorted_scores, 1):
    result = results[exp_name]
    print(f"{i:2d}. {exp_name:<25} 得分: {score:.3f} | "
          f"总体: {result['overall_acc']:.2f}% | "
          f"少数类平均: {np.mean(list(result['minority_acc'].values())):.1f}%")

# 9. 推荐配置
best_method = sorted_scores[0][0]
best_result = results[best_method]

print(f"\n9. 推荐配置: {best_method}")
print(f"   总体准确率: {best_result['overall_acc']:.2f}%")
print(f"   少数类别表现:")
for cls_name, acc in best_result['minority_acc'].items():
    print(f"     {cls_name}: {acc:.1f}%")

# 10. 与baseline对比
baseline_acc = 84.60  # 之前的普通损失结果
improvement = best_result['overall_acc'] - baseline_acc

print(f"\n10. 与baseline对比:")
print(f"   Baseline (普通交叉熵): {baseline_acc:.2f}%")
print(f"   最佳策略 ({best_method}): {best_result['overall_acc']:.2f}%")
print(f"   提升: {improvement:+.2f}%")

if improvement > 0:
    print(f"✅ 智能平衡策略有效! 总体准确率提升 {improvement:.2f}%")
else:
    print(f"⚠️  需要进一步优化平衡策略")

print("\n" + "="*80)
print("改进2完成!")
print("="*80)