#!/usr/bin/env python3
"""
实验1: 完整数据集上的ESN+MLP实验
使用MUUFL Gulfport所有有效像素（53,687个样本）
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# 导入MUUFL工具
import sys
sys.path.append(str(Path(__file__).parent.parent))
from muufl_utils import MUUFLDataset

print("="*70)
print("实验1: 完整数据集ESN+MLP")
print("="*70)

# 配置
DATA_DIR = Path("../data")
CACHE_DIR = Path("../cache/full_experiment")
CACHE_DIR.mkdir(exist_ok=True)

# 1. 加载完整数据集
print("\n1. 加载完整数据集...")
loader = MUUFLDataset(data_dir=DATA_DIR)
spectra, labels = loader.prepare_dataset(
    normalize=True,
    remap_labels=True
)

print(f"完整数据集:")
print(f"  光谱数据: {spectra.shape}")
print(f"  标签: {labels.shape}")
print(f"  类别分布:")
loader.get_class_distribution(labels)

# 2. ESN编码（完整数据集）
print("\n2. ESN编码（完整数据集）...")

# 设置随机种子
rng = np.random.RandomState(42)
torch.manual_seed(42)
np.random.seed(42)

# ESN参数（使用原论文参数）
in_dim = spectra.shape[1]  # 64
rff_dim = 1280  # 原论文参数
res_dim = 600   # 原论文参数
sr = 0.7526922197568455  # 原论文参数
leak = 0.8316475567419054  # 原论文参数

print(f"ESN配置（原论文参数）:")
print(f"  输入维度: {in_dim}")
print(f"  RFF维度: {rff_dim}")
print(f"  储备池: {res_dim}")
print(f"  谱半径: {sr:.4f}")
print(f"  泄漏率: {leak:.4f}")

# 检查缓存
features_cache = CACHE_DIR / "full_esn_features.npy"
labels_cache = CACHE_DIR / "full_labels.npy"

if features_cache.exists() and labels_cache.exists():
    print(f"从缓存加载ESN特征...")
    X = np.load(features_cache)
    y = np.load(labels_cache)
    print(f"  特征: {X.shape}, 标签: {y.shape}")
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
    
    # 编码函数（优化版）
    def encode_batch(batch_spectra):
        """批量编码光谱数据"""
        batch_size = len(batch_spectra)
        
        # RFF变换
        phi = scale * np.cos(batch_spectra @ W_rff + b_rff)  # (batch, rff_dim)
        
        # ESN前向传播（单时间步）
        states = np.zeros((batch_size, res_dim), dtype=np.float32)
        for i in range(batch_size):
            h = np.tanh(W_in @ phi[i] + W @ states[i])
            states[i] = (1.0 - leak) * states[i] + leak * h
        
        return states
    
    # 批量编码
    print(f"编码 {len(spectra)} 个样本...")
    batch_size = 1000
    feats = []
    
    for i in range(0, len(spectra), batch_size):
        batch = spectra[i:i+batch_size]
        batch_feats = encode_batch(batch)
        feats.append(batch_feats)
        
        if (i // batch_size) % 10 == 0:
            print(f"  进度: {min(i+batch_size, len(spectra))}/{len(spectra)}")
    
    X = np.vstack(feats)
    y = labels
    
    print(f"编码完成: {X.shape}")
    
    # 保存到缓存
    np.save(features_cache, X)
    np.save(labels_cache, y)
    print(f"特征已保存到缓存")

# 3. 数据划分
print("\n3. 数据划分...")
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"训练集: {X_tr.shape} ({X_tr.shape[0]/len(X)*100:.1f}%)")
print(f"验证集: {X_val.shape} ({X_val.shape[0]/len(X)*100:.1f}%)")

# 4. MLP模型
print("\n4. 构建MLP模型...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=512, out_dim=11):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

model = MLP(in_dim=X.shape[1], hidden=512, out_dim=11).to(device)
print(f"模型结构: {X.shape[1]} -> 512 -> 11")
print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")

# 转换为PyTorch张量
X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_tr_tensor = torch.tensor(y_tr, dtype=torch.long).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

# 5. 训练
print("\n5. 训练MLP...")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

epochs = 30
batch_size = 256

# 训练记录
train_losses = []
train_accs = []
val_losses = []
val_accs = []

for epoch in range(epochs):
    model.train()
    
    # 随机打乱
    indices = torch.randperm(len(X_tr_tensor))
    X_shuffled = X_tr_tensor[indices]
    y_shuffled = y_tr_tensor[indices]
    
    epoch_loss = 0
    correct = 0
    total = 0
    
    # 小批量训练
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

# 6. 最终评估
print("\n6. 最终评估...")
model.eval()
with torch.no_grad():
    outputs = model(X_val_tensor)
    _, predicted = outputs.max(1)
    
    y_true = y_val_tensor.cpu().numpy()
    y_pred = predicted.cpu().numpy()
    
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    
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

# 7. 保存结果
print("\n7. 保存结果...")

# 保存模型
model_path = CACHE_DIR / "full_muufl_mlp_model.pth"
torch.save(model.state_dict(), model_path)
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
np.save(CACHE_DIR / "training_history.npy", history)
print(f"训练历史已保存")

# 保存预测结果
results = {
    'y_true': y_true,
    'y_pred': y_pred,
    'X_val': X_val,
    'y_val': y_val
}
np.save(CACHE_DIR / "prediction_results.npy", results)
print(f"预测结果已保存")

print("\n" + "="*70)
print("实验1完成!")
print(f"最终验证准确率: {overall_acc:.2f}%")
print("="*70)

# 与baseline对比
print("\n与baseline对比:")
print(f"  随机猜测 (11类): {100/11:.1f}%")
print(f"  多数类 (trees): {23246/53687*100:.1f}%")
print(f"  ESN+MLP: {overall_acc:.1f}%")

if overall_acc > 23246/53687*100:
    print(f"✅ ESN+MLP优于多数类baseline!")
else:
    print(f"⚠️  ESN+MLP未优于多数类baseline")