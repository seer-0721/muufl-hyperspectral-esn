#!/usr/bin/env python3
"""
改进1: 完整数据集 + 类别平衡
在53,687个样本上运行ESN+MLP，应用类别平衡策略
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

print("="*80)
print("改进1: 完整数据集 + 类别平衡")
print("="*80)

# 导入工具
import sys
sys.path.append(str(Path(__file__).parent.parent))
from muufl_utils import MUUFLDataset
from utils.balanced_loss import compute_class_weights, BalancedCrossEntropyLoss, FocalLoss

# 1. 加载完整数据集
print("\n1. 加载完整数据集...")
loader = MUUFLDataset(data_dir="./data")
spectra, labels = loader.prepare_dataset(
    normalize=True,
    remap_labels=True
)

print(f"完整数据集:")
print(f"  光谱数据: {spectra.shape}")
print(f"  标签: {labels.shape}")

# 2. 计算类别权重
print("\n2. 计算类别权重...")
class_weights = compute_class_weights(labels, method='balanced')

# 3. ESN编码（优化版）
print("\n3. ESN编码（优化版）...")

# 设置随机种子
rng = np.random.RandomState(42)
torch.manual_seed(42)
np.random.seed(42)

# ESN参数（使用稍大的配置）
in_dim = spectra.shape[1]
rff_dim = 512  # 增加RFF维度
res_dim = 400  # 增加储备池大小
sr = 0.9
leak = 0.8

print(f"ESN配置（优化）:")
print(f"  输入维度: {in_dim}")
print(f"  RFF维度: {rff_dim}")
print(f"  储备池: {res_dim}")

# 检查缓存
cache_dir = Path("./cache/full_balanced")
cache_dir.mkdir(exist_ok=True)
features_cache = cache_dir / "esn_features_full.npy"

if features_cache.exists():
    print(f"从缓存加载ESN特征...")
    X = np.load(features_cache)
    print(f"  特征: {X.shape}")
else:
    # 构建RFF
    print("构建RFF...")
    W_rff = rng.randn(in_dim, rff_dim).astype(np.float32)
    b_rff = rng.uniform(0, 2 * np.pi, size=rff_dim).astype(np.float32)
    scale = np.float32(np.sqrt(2.0 / rff_dim))
    
    # 构建ESN
    print("构建ESN...")
    W_in = rng.randn(res_dim, rff_dim).astype(np.float32)
    W = rng.randn(res_dim, res_dim).astype(np.float32)
    eigmax = float(np.abs(np.linalg.eigvals(W)).max())
    W *= np.float32(sr / (eigmax + 1e-12))
    
    # 优化的批量编码函数
    def encode_batch_optimized(batch_spectra):
        """优化的批量编码"""
        batch_size = len(batch_spectra)
        phi = scale * np.cos(batch_spectra @ W_rff + b_rff)
        
        # 使用矩阵运算加速
        states = np.zeros((batch_size, res_dim), dtype=np.float32)
        for i in range(batch_size):
            # 仍然需要循环，但可以进一步优化
            h = np.tanh(W_in @ phi[i] + W @ states[i])
            states[i] = (1.0 - leak) * states[i] + leak * h
        
        return states
    
    # 分批编码完整数据集
    print(f"编码 {len(spectra)} 个样本...")
    batch_size = 2000  # 更大的批处理
    feats = []
    
    import time
    start_time = time.time()
    
    for i in range(0, len(spectra), batch_size):
        batch = spectra[i:i+batch_size]
        batch_feats = encode_batch_optimized(batch)
        feats.append(batch_feats)
        
        if (i // batch_size) % 5 == 0:
            elapsed = time.time() - start_time
            samples_per_sec = min(i+batch_size, len(spectra)) / elapsed
            remaining = (len(spectra) - i) / samples_per_sec if samples_per_sec > 0 else 0
            print(f"  进度: {min(i+batch_size, len(spectra))}/{len(spectra)} "
                  f"({min(i+batch_size, len(spectra))/len(spectra)*100:.1f}%) "
                  f"ETA: {remaining/60:.1f}分钟")
    
    X = np.vstack(feats)
    print(f"编码完成: {X.shape}, 耗时: {time.time()-start_time:.1f}秒")
    
    # 保存到缓存
    np.save(features_cache, X)
    print(f"特征已保存到缓存")

# 4. 数据划分（使用分层抽样）
print("\n4. 数据划分...")
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(
    X, labels, test_size=0.2, stratify=labels, random_state=42
)

print(f"训练集: {X_tr.shape} ({X_tr.shape[0]/len(X)*100:.1f}%)")
print(f"验证集: {X_val.shape} ({X_val.shape[0]/len(X)*100:.1f}%)")

# 5. 增强的MLP模型
print("\n5. 构建增强的MLP模型...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class EnhancedMLP(nn.Module):
    """增强的MLP模型"""
    def __init__(self, in_dim, hidden1=256, hidden2=128, out_dim=11, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.fc3 = nn.Linear(hidden2, out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)

import torch.nn.functional as F

model = EnhancedMLP(in_dim=X.shape[1], hidden1=256, hidden2=128, out_dim=11).to(device)
print(f"模型结构: {X.shape[1]} -> 256 -> 128 -> 11")
print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")

# 转换为张量
X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_tr_tensor = torch.tensor(y_tr, dtype=torch.long).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

# 6. 训练（使用平衡损失）
print("\n6. 训练MLP（使用平衡损失）...")

# 使用平衡交叉熵损失
criterion = BalancedCrossEntropyLoss(class_weights=class_weights.to(device))
# 或者使用Focal Loss
# criterion = FocalLoss(gamma=2.0, alpha=class_weights.to(device))

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

epochs = 30
batch_size = 512  # 更大的批处理

# 训练记录
train_losses = []
train_accs = []
val_losses = []
val_accs = []

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
    
    scheduler.step()
    
    # 记录
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:3d}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

# 7. 最终评估
print("\n7. 最终评估...")
model.eval()
with torch.no_grad():
    outputs = model(X_val_tensor)
    _, predicted = outputs.max(1)
    
    y_true = y_val_tensor.cpu().numpy()
    y_pred = predicted.cpu().numpy()
    
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
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
    
    # 分类报告
    print("\n分类报告:")
    target_names = [class_names.get(i+1, f"class_{i}") for i in range(11)]
    print(classification_report(y_true, y_pred, target_names=target_names, digits=3))
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    print(f"混淆矩阵形状: {cm.shape}")

# 8. 保存结果
print("\n8. 保存结果...")

# 保存模型
model_path = cache_dir / "full_balanced_mlp_model.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses,
    'train_accs': train_accs,
    'val_accs': val_accs,
    'final_val_acc': overall_acc,
    'class_accuracies': class_accuracies
}, model_path)
print(f"模型已保存到 {model_path}")

# 保存训练历史
history = {
    'train_loss': train_losses,
    'train_acc': train_accs,
    'val_loss': val_losses,
    'val_acc': val_accs,
    'final_val_acc': overall_acc,
    'class_accuracies': class_accuracies
}
np.save(cache_dir / "training_history.npy", history)
print(f"训练历史已保存")

# 保存预测结果
results = {
    'y_true': y_true,
    'y_pred': y_pred,
    'X_val': X_val,
    'y_val': y_val
}
np.save(cache_dir / "prediction_results.npy", results)
print(f"预测结果已保存")

# 9. 与baseline对比
print("\n9. 与baseline对比...")

# 加载之前的baseline结果
baseline_acc = 84.30  # 之前的ESN结果
cnn_acc = 80.80  # 1D-CNN结果
majority_acc = 44.3

print(f"\n性能对比:")
print(f"  方法           | 验证准确率 | 相对提升")
print(f"  --------------|------------|----------")
print(f"  多数类(trees)  |   44.3%    |   基准")
print(f"  1D-CNN        |   80.8%    |   +36.5%")
print(f"  ESN+MLP(之前) |   84.3%    |   +40.0%")
print(f"  ESN+MLP(改进) |   {overall_acc:.1f}%    |   +{overall_acc-majority_acc:.1f}%")

if overall_acc > baseline_acc:
    improvement = overall_acc - baseline_acc
    print(f"\n✅ 改进成功! 准确率提升: +{improvement:.1f}%")
else:
    print(f"\n⚠️  需要进一步优化")

# 10. 少数类别分析
print("\n10. 少数类别分析:")
minority_classes = ['yellowcurb', 'cloth', 'sidewalk']
for cls in minority_classes:
    if cls in class_accuracies:
        acc = class_accuracies[cls]
        print(f"  {cls:15s}: {acc:6.2f}%")

print("\n" + "="*80)
print("改进1完成!")
print(f"最终验证准确率: {overall_acc:.2f}%")
print("="*80)