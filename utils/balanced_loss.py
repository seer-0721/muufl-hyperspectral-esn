#!/usr/bin/env python3
"""
类别平衡损失函数工具
用于处理极度不平衡的MUUFL数据集
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_class_weights(labels, method='balanced'):
    """
    计算类别权重
    
    Args:
        labels: 标签数组
        method: 权重计算方法
            - 'balanced': 反比于类别频率
            - 'sqrt': 反比于类别频率的平方根
            - 'inverse': 反比于类别频率
            - 'uniform': 均匀权重
    
    Returns:
        类别权重张量
    """
    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    unique, counts = np.unique(labels_np, return_counts=True)
    n_classes = len(unique)
    
    if method == 'balanced':
        # 反比于类别频率
        weights = 1.0 / counts
        weights = weights / weights.sum() * n_classes
    
    elif method == 'sqrt':
        # 反比于类别频率的平方根
        weights = 1.0 / np.sqrt(counts)
        weights = weights / weights.sum() * n_classes
    
    elif method == 'inverse':
        # 简单的反比
        weights = 1.0 / counts
        weights = weights / weights.min()
    
    elif method == 'uniform':
        # 均匀权重
        weights = np.ones(n_classes)
    
    else:
        raise ValueError(f"未知的权重计算方法: {method}")
    
    # 创建权重映射
    weight_map = torch.zeros(max(unique) + 1, dtype=torch.float32)
    for cls, weight in zip(unique, weights):
        weight_map[cls] = weight
    
    print(f"类别权重 ({method}):")
    for cls, weight in zip(unique, weights):
        print(f"  类别 {cls}: 权重={weight:.3f}, 样本数={counts[list(unique).index(cls)]}")
    
    return weight_map

class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification
    Paper: Focal Loss for Dense Object Detection
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (list, np.ndarray, torch.Tensor)):
                if len(self.alpha) == inputs.size(0):  # per-sample weights
                    alpha_t = torch.tensor(self.alpha, device=inputs.device)
                else:  # per-class weights
                    alpha_t = torch.tensor(self.alpha, device=inputs.device)[targets]
            else:
                alpha_t = self.alpha
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class BalancedCrossEntropyLoss(nn.Module):
    """
    带类别权重的交叉熵损失
    """
    def __init__(self, class_weights=None, reduction='mean'):
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        if self.class_weights is not None:
            # class_weights应该是长度为num_classes的张量
            weight = self.class_weights.to(inputs.device)
        else:
            weight = None
        
        return F.cross_entropy(inputs, targets, weight=weight, reduction=self.reduction)

def apply_oversampling(X, y, target_samples=1000):
    """
    对少数类别进行过采样
    
    Args:
        X: 特征矩阵
        y: 标签
        target_samples: 目标样本数（每个类别）
    
    Returns:
        过采样后的X, y
    """
    from sklearn.utils import resample
    
    unique, counts = np.unique(y, return_counts=True)
    X_resampled = []
    y_resampled = []
    
    print("过采样统计:")
    for cls, count in zip(unique, counts):
        X_cls = X[y == cls]
        y_cls = y[y == cls]
        
        if count < target_samples:
            # 过采样
            X_cls_resampled = resample(X_cls, n_samples=target_samples, random_state=42)
            y_cls_resampled = np.full(target_samples, cls)
            print(f"  类别 {cls}: {count} -> {target_samples} (过采样)")
        else:
            # 保持原样或欠采样
            X_cls_resampled = X_cls
            y_cls_resampled = y_cls
            print(f"  类别 {cls}: {count} (保持)")
        
        X_resampled.append(X_cls_resampled)
        y_resampled.append(y_cls_resampled)
    
    X_balanced = np.vstack(X_resampled)
    y_balanced = np.concatenate(y_resampled)
    
    # 打乱顺序
    indices = np.random.permutation(len(X_balanced))
    X_balanced = X_balanced[indices]
    y_balanced = y_balanced[indices]
    
    print(f"平衡后: {X_balanced.shape}, 类别分布: {np.bincount(y_balanced.astype(int))}")
    
    return X_balanced, y_balanced

def test_balanced_loss():
    """测试平衡损失函数"""
    print("测试平衡损失函数...")
    
    # 模拟不平衡数据
    n_samples = 1000
    n_classes = 11
    y = np.random.choice([0, 1, 2], size=n_samples, p=[0.7, 0.2, 0.1])
    
    print(f"原始分布: {np.bincount(y)}")
    
    # 计算类别权重
    weights = compute_class_weights(y, method='balanced')
    print(f"类别权重: {weights}")
    
    # 测试过采样
    X = np.random.randn(n_samples, 64)
    X_balanced, y_balanced = apply_oversampling(X, y, target_samples=500)
    
    # 测试损失函数
    criterion1 = BalancedCrossEntropyLoss(class_weights=weights)
    criterion2 = FocalLoss(gamma=2.0)
    
    print("✅ 平衡损失函数测试完成")
    
    return True

if __name__ == "__main__":
    test_balanced_loss()