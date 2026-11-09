#!/usr/bin/env python3
"""
实验2: 1D-CNN baseline对比
使用相同的MUUFL数据集，对比ESN+MLP与1D-CNN的性能
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

print("="*70)
print("实验2: 1D-CNN Baseline对比")
print("="*70)

# 1. 导入工具和加载数据
print("\n1. 加载数据...")
from muufl_utils import MUUFLDataset

loader = MUUFLDataset(data_dir="./data")
spectra, labels = loader.prepare_dataset(
    normalize=True,
    remap_labels=True
)

# 使用与ESN实验相同的样本（确保公平对比）
print(f"数据集: {spectra.shape}")
print(f"使用5,000个样本进行公平对比")

# 取前5000个样本（与ESN实验一致）
spectra = spectra[:5000]
labels = labels[:5000]
print(f"对比样本: {spectra.shape}")

# 2. 1D-CNN模型
print("\n2. 构建1D-CNN模型...")

class CNN1D(nn.Module):
    """1D-CNN用于光谱分类"""
    def __init__(self, in_channels=1, num_classes=11):
        super().__init__()
        
        # 卷积层
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x形状: (batch, 1, 64)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# 3. 数据准备
print("\n3. 数据准备...")
from sklearn.model_selection import train_test_split

# 与ESN实验相同的划分
X_tr, X_val, y_tr, y_val = train_test_split(
    spectra, labels, test_size=0.2, stratify=labels, random_state=42
)

print(f"训练集: {X_tr.shape}")
print(f"验证集: {X_val.shape}")

# 转换为PyTorch张量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 1D-CNN需要(batch, channels, length)格式
X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32).unsqueeze(1).to(device)  # (batch, 1, 64)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(device)
y_tr_tensor = torch.tensor(y_tr, dtype=torch.long).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

print(f"输入形状: {X_tr_tensor.shape}")

# 4. 训练1D-CNN
print("\n4. 训练1D-CNN...")

model = CNN1D(in_channels=1, num_classes=11).to(device)
print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

epochs = 20
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
    
    scheduler.step()
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:3d}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

# 5. 最终评估
print("\n5. 最终评估...")
model.eval()
with torch.no_grad():
    outputs = model(X_val_tensor)
    _, predicted = outputs.max(1)
    
    y_true = y_val_tensor.cpu().numpy()
    y_pred = predicted.cpu().numpy()
    
    from sklearn.metrics import accuracy_score, classification_report
    
    cnn_acc = accuracy_score(y_true, y_pred) * 100
    print(f"1D-CNN验证集准确率: {cnn_acc:.2f}%")
    
    # 每个类别的准确率
    print("\n1D-CNN每个类别的准确率:")
    class_names = loader.CLASS_NAMES
    
    cnn_class_acc = {}
    for i in range(11):
        mask = y_true == i
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == y_true[mask]).mean() * 100
            class_name = class_names.get(i+1, f"class_{i}")
            cnn_class_acc[class_name] = class_acc
            print(f"  {class_name:15s}: {class_acc:6.2f}% ({mask.sum()} samples)")

# 6. 与ESN结果对比
print("\n" + "="*70)
print("ESN+MLP vs 1D-CNN 对比")
print("="*70)

# 从ESN实验加载结果
esn_results_path = Path("./cache/muufl_mlp_model.pth")
if esn_results_path.exists():
    # 从之前的运行中获取ESN准确率（84.3%）
    esn_acc = 84.30  # 从之前的实验
    
    print(f"\n性能对比:")
    print(f"  方法           | 验证准确率 | 相对提升")
    print(f"  --------------|------------|----------")
    print(f"  随机猜测       |    9.1%    |    -")
    print(f"  多数类(trees)  |   44.3%    |    -")
    print(f"  1D-CNN        |   {cnn_acc:.1f}%    |   +{cnn_acc-44.3:.1f}%")
    print(f"  ESN+MLP       |   {esn_acc:.1f}%    |   +{esn_acc-44.3:.1f}%")
    
    # 对比分析
    print(f"\n对比分析:")
    if cnn_acc > esn_acc:
        diff = cnn_acc - esn_acc
        print(f"  1D-CNN比ESN+MLP高 {diff:.1f}%")
    elif esn_acc > cnn_acc:
        diff = esn_acc - cnn_acc
        print(f"  ESN+MLP比1D-CNN高 {diff:.1f}%")
    else:
        print(f"  两者性能相当")
    
    # 类别级别的对比
    print(f"\n类别级别对比（ESN vs 1D-CNN）:")
    print(f"  类别           | ESN准确率 | 1D-CNN准确率 | 差异")
    print(f"  --------------|-----------|-------------|------")
    
    # 注意：这里需要ESN的类别准确率，我们使用之前的结果
    esn_class_acc = {
        'trees': 93.45, 'grass_pure': 68.75, 'grass_ground': 79.34,
        'dirt_sand': 72.73, 'road': 87.10, 'water': 87.50,
        'shadow': 69.05, 'buildings': 86.09, 'sidewalk': 34.62,
        'yellowcurb': 0.00, 'cloth': 50.00
    }
    
    for class_name in class_names.values():
        if class_name in esn_class_acc and class_name in cnn_class_acc:
            esn_acc_val = esn_class_acc[class_name]
            cnn_acc_val = cnn_class_acc.get(class_name, 0)
            diff = esn_acc_val - cnn_acc_val
            
            if diff > 5:
                marker = "↑ ESN更好"
            elif diff < -5:
                marker = "↑ CNN更好"
            else:
                marker = "≈ 相当"
            
            print(f"  {class_name:15s}|   {esn_acc_val:5.1f}%   |    {cnn_acc_val:5.1f}%    | {diff:5.1f}% {marker}")

# 7. 保存结果
print("\n6. 保存结果...")
results_dir = Path("./results/tables")
results_dir.mkdir(parents=True, exist_ok=True)

results = {
    'cnn_accuracy': cnn_acc,
    'cnn_class_accuracies': cnn_class_acc,
    'comparison': {
        'random_guess': 100/11,
        'majority_class': 44.3,
        'esn_mlp': 84.3,
        'cnn_1d': cnn_acc
    }
}

import json
with open(results_dir / "baseline_comparison.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"结果已保存到 {results_dir}/baseline_comparison.json")

print("\n" + "="*70)
print("实验2完成!")
print("="*70)