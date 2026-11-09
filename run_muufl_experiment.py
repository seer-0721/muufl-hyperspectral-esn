#!/usr/bin/env python3
"""
运行MUUFL ESN实验的简化版本
使用小样本快速验证流程
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# 导入MUUFL工具
from muufl_utils import MUUFLDataset

print("="*60)
print("MUUFL Gulfport ESN实验 - 简化版本")
print("="*60)

# 创建目录
cache_dir = Path("./cache")
cache_dir.mkdir(exist_ok=True)

# 1. 加载数据（使用小样本）
print("\n1. 加载数据...")
loader = MUUFLDataset(data_dir="./data")
spectra, labels = loader.prepare_dataset(
    normalize=True,
    remap_labels=True
)

# 使用小样本加速测试
SAMPLE_SIZE = 5000  # 使用5000个样本进行快速测试
if len(spectra) > SAMPLE_SIZE:
    indices = np.random.choice(len(spectra), SAMPLE_SIZE, replace=False)
    spectra = spectra[indices]
    labels = labels[indices]

print(f"使用 {len(spectra)} 个样本进行测试")
print(f"光谱数据: {spectra.shape}")
print(f"标签: {labels.shape}")

# 2. ESN编码
print("\n2. ESN编码...")

# 设置随机种子
rng = np.random.RandomState(42)
torch.manual_seed(42)
np.random.seed(42)

# ESN参数（简化）
in_dim = spectra.shape[1]  # 64
rff_dim = 256  # 简化
res_dim = 200  # 简化
sr = 0.9
leak = 0.8

print(f"ESN配置:")
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

# 编码函数
def encode_single(seq):
    """编码单个光谱序列"""
    phi = scale * np.cos(seq @ W_rff + b_rff)  # 单时间步
    
    # ESN前向传播
    state = np.zeros(res_dim, dtype=np.float32)
    h = np.tanh(W_in @ phi + W @ state)
    state = (1.0 - leak) * state + leak * h
    
    return state

# 批量编码
print("编码光谱数据...")
batch_size = 500
feats = []

for i in range(0, len(spectra), batch_size):
    batch = spectra[i:i+batch_size]
    batch_feats = np.array([encode_single(sample) for sample in batch])
    feats.append(batch_feats)
    
    if i % 1000 == 0:
        print(f"  已编码 {min(i+batch_size, len(spectra))}/{len(spectra)} 个样本")

X = np.vstack(feats)
y = labels

print(f"编码后特征: {X.shape}")

# 保存特征
np.save(cache_dir / "muufl_esn_features_small.npy", X)
np.save(cache_dir / "muufl_labels_small.npy", y)
print(f"特征已保存到 {cache_dir}/")

# 3. 数据划分
print("\n3. 数据划分...")
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"训练集: {X_tr.shape}")
print(f"验证集: {X_val.shape}")

# 4. MLP模型
print("\n4. 构建MLP模型...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class SimpleMLP(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=11):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

model = SimpleMLP(in_dim=X.shape[1], hidden=128, out_dim=11).to(device)
print(f"模型结构: {X.shape[1]} -> 128 -> 11")

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

epochs = 20
batch_size = 64

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
    
    if (epoch + 1) % 5 == 0:
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
    
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    overall_acc = accuracy_score(y_true, y_pred) * 100
    print(f"验证集准确率: {overall_acc:.2f}%")
    
    # 每个类别的准确率
    print("\n每个类别的准确率:")
    class_names = loader.CLASS_NAMES
    for i in range(11):
        mask = y_true == i
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == y_true[mask]).mean() * 100
            class_name = class_names.get(i+1, f"class_{i}")
            print(f"  {class_name:15s}: {class_acc:6.2f}% ({mask.sum()} samples)")

# 7. 保存结果
print("\n7. 保存结果...")
torch.save(model.state_dict(), cache_dir / "muufl_mlp_model_small.pth")
print(f"模型已保存到 {cache_dir}/muufl_mlp_model_small.pth")

print("\n" + "="*60)
print("实验完成!")
print(f"最终验证准确率: {overall_acc:.2f}%")
print("="*60)