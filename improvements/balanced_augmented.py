#!/usr/bin/env python3
"""
改进2: 类别平衡 + 数据增强
结合两种改进策略
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

print("="*80)
print("改进2: 类别平衡 + 数据增强")
print("="*80)

# 1. 导入工具
print("\n1. 导入工具...")
import sys
sys.path.append(str(Path(__file__).parent.parent))
from muufl_utils import MUUFLDataset
from utils.augmentation import SpectralAugmentation, augment_minority_classes

# 2. 加载数据
print("\n2. 加载数据...")
loader = MUUFLDataset(data_dir="./data")
spectra, labels = loader.prepare_dataset(
    normalize=True,
    remap_labels=True
)

# 使用小样本进行快速迭代
sample_size = 5000
spectra = spectra[:sample_size]
labels = labels[:sample_size]

print(f"使用 {sample_size} 个样本进行快速迭代")
print(f"光谱数据: {spectra.shape}")
print(f"标签: {labels.shape}")

# 3. 数据增强（对少数类别）
print("\n3. 数据增强...")
X_augmented, y_augmented = augment_minority_classes(spectra, labels, target_multiplier=3)

print(f"\n增强后数据:")
print(f"  光谱数据: {X_augmented.shape}")
print(f"  标签: {y_augmented.shape}")

# 4. 计算类别权重（基于增强后的数据）
print("\n4. 计算类别权重...")
unique, counts = np.unique(y_augmented, return_counts=True)

# 计算平衡权重
class_weights = 1.0 / counts
class_weights = class_weights / class_weights.sum() * len(unique)

print(f"类别权重（基于增强后数据）:")
for cls, weight in zip(unique, class_weights):
    class_name = loader.CLASS_NAMES.get(cls+1, f"class_{cls}")
    count = counts[list(unique).index(cls)]
    print(f"  {class_name:15s}: 权重={weight:.3f}, 样本数={count}")

weight_tensor = torch.zeros(max(unique) + 1, dtype=torch.float32)
for cls, weight in zip(unique, class_weights):
    weight_tensor[cls] = weight

# 5. ESN编码
print("\n5. ESN编码...")

# 设置随机种子
rng = np.random.RandomState(42)
torch.manual_seed(42)
np.random.seed(42)

# ESN参数（稍作优化）
in_dim = X_augmented.shape[1]
rff_dim = 512  # 增加RFF维度
res_dim = 300  # 增加储备池
sr = 0.85
leak = 0.75

print(f"ESN配置（优化）:")
print(f"  输入维度: {in_dim}")
print(f"  RFF维度: {rff_dim}")
print(f"  储备池: {res_dim}")
print(f"  谱半径: {sr}")
print(f"  泄漏率: {leak}")

# 构建RFF
W_rff = rng.randn(in_dim, rff_dim).astype(np.float32)
b_rff = rng.uniform(0, 2 * np.pi, size=rff_dim).astype(np.float32)
scale = np.float32(np.sqrt(2.0 / rff_dim))

# 构建ESN
W_in = rng.randn(res_dim, rff_dim).astype(np.float32)
W = rng.randn(res_dim, res_dim).astype(np.float32)
eigmax = float(np.abs(np.linalg.eigvals(W)).max())
W *= np.float32(sr / (eigmax + 1e-12))

# 编码函数（优化版）
def encode_batch_optimized(batch_spectra):
    batch_size = len(batch_spectra)
    phi = scale * np.cos(batch_spectra @ W_rff + b_rff)
    
    states = np.zeros((batch_size, res_dim), dtype=np.float32)
    for i in range(batch_size):
        h = np.tanh(W_in @ phi[i] + W @ states[i])
        states[i] = (1.0 - leak) * states[i] + leak * h
    
    return states

# 编码增强后的数据
print(f"编码 {len(X_augmented)} 个样本...")
batch_size = 500
feats = []

for i in range(0, len(X_augmented), batch_size):
    batch = X_augmented[i:i+batch_size]
    batch_feats = encode_batch_optimized(batch)
    feats.append(batch_feats)
    
    if (i // batch_size) % 5 == 0:
        print(f"  进度: {min(i+batch_size, len(X_augmented))}/{len(X_augmented)}")

X = np.vstack(feats)
y = y_augmented
print(f"编码完成: {X.shape}")

# 6. 数据划分
print("\n6. 数据划分...")
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"训练集: {X_tr.shape}")
print(f"验证集: {X_val.shape}")

# 7. 构建改进的MLP模型
print("\n7. 构建改进的MLP模型...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class AdvancedMLP(nn.Module):
    def __init__(self, in_dim, hidden1=256, hidden2=128, hidden3=64, out_dim=11):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.bn3 = nn.BatchNorm1d(hidden3)
        self.fc4 = nn.Linear(hidden3, out_dim)
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.3)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout2(x)
        return self.fc4(x)

model = AdvancedMLP(in_dim=X.shape[1], hidden1=256, hidden2=128, hidden3=64, out_dim=11).to(device)
print(f"模型结构: {X.shape[1]} -> 256 -> 128 -> 64 -> 11")
print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")

# 转换为张量
X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_tr_tensor = torch.tensor(y_tr, dtype=torch.long).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

# 8. 训练（使用类别权重 + 数据增强）
print("\n8. 训练MLP...")

# 使用带权重的交叉熵损失
criterion = nn.CrossEntropyLoss(weight=weight_tensor.to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

epochs = 30
batch_size = 128

best_val_acc = 0
patience_counter = 0
patience = 10

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    
    scheduler.step(val_acc)
    
    # 早停检查
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        # 保存最佳模型
        torch.save(model.state_dict(), "./cache/best_model.pth")
    else:
        patience_counter += 1
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:3d}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    if patience_counter >= patience:
        print(f"\n早停触发，最佳验证准确率: {best_val_acc:.2f}%")
        break

# 9. 最终评估（使用最佳模型）
print("\n9. 最终评估...")
model.load_state_dict(torch.load("./cache/best_model.pth"))
model.eval()

with torch.no_grad():
    outputs = model(X_val_tensor)
    _, predicted = outputs.max(1)
    
    y_true = y_val_tensor.cpu().numpy()
    y_pred = predicted.cpu().numpy()
    
    from sklearn.metrics import accuracy_score, classification_report
    
    overall_acc = accuracy_score(y_true, y_pred) * 100
    print(f"验证集准确率: {overall_acc:.2f}%")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    
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
previous_acc = 84.30  # 原始ESN结果
balanced_acc = 83.40  # 仅类别平衡的结果
majority_acc = 44.3

print(f"\n性能对比:")
print(f"  方法                | 验证准确率 | 相对提升")
print(f"  -------------------|------------|----------")
print(f"  多数类(trees)       |   44.3%    |   基准")
print(f"  ESN+MLP(原始)      |   84.3%    |   +40.0%")
print(f"  ESN+MLP(仅平衡)    |   83.4%    |   +39.1%")
print(f"  ESN+MLP(平衡+增强) |   {overall_acc:.1f}%    |   +{overall_acc-majority_acc:.1f}%")

if overall_acc > previous_acc:
    improvement = overall_acc - previous_acc
    print(f"\n✅ 改进成功! 准确率提升: +{improvement:.1f}%")
elif overall_acc > balanced_acc:
    improvement = overall_acc - balanced_acc
    print(f"\n✅ 数据增强有效! 相比仅平衡提升: +{improvement:.1f}%")
else:
    print(f"\n⚠️  需要进一步优化")

# 11. 少数类别特别分析
print("\n11. 少数类别分析:")
minority_classes = ['yellowcurb', 'cloth', 'sidewalk']
minority_results = {}

for cls in minority_classes:
    if cls in class_accuracies:
        acc = class_accuracies[cls]
        # 之前的准确率
        prev_acc = {'yellowcurb': 0.0, 'cloth': 50.0, 'sidewalk': 57.7}.get(cls, 0.0)
        minority_results[cls] = (acc, prev_acc)

print(f"\n少数类别对比:")
for cls, (acc, prev_acc) in minority_results.items():
    if acc > prev_acc:
        improvement = acc - prev_acc
        print(f"  {cls:15s}: {acc:6.2f}% (之前: {prev_acc:.1f}%, 提升: +{improvement:.1f}%) ✅")
    else:
        print(f"  {cls:15s}: {acc:6.2f}% (之前: {prev_acc:.1f}%, 变化: {acc-prev_acc:+.1f}%)")

# 12. 保存结果
print("\n12. 保存结果...")
cache_dir = Path("./cache/balanced_augmented")
cache_dir.mkdir(exist_ok=True)

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_val_acc': best_val_acc,
    'final_val_acc': overall_acc,
    'class_accuracies': class_accuracies
}, cache_dir / "balanced_augmented_model.pth")

np.save(cache_dir / "augmented_features.npy", X)
np.save(cache_dir / "augmented_labels.npy", y)

print(f"模型和特征已保存到 {cache_dir}/")

print("\n" + "="*80)
print("改进2完成!")
print(f"最终验证准确率: {overall_acc:.2f}%")
print(f"最佳验证准确率: {best_val_acc:.2f}%")
print("="*80)

# 下一步建议
print("\n下一步建议:")
print("1. 在完整数据集上应用此方法")
print("2. 尝试不同的增强策略组合")
print("3. 优化ESN超参数（网格搜索）")
print("4. 尝试集成学习")