#!/usr/bin/env python3
"""
光谱数据增强工具
用于增加高光谱数据的多样性
"""

import numpy as np
import torch

class SpectralAugmentation:
    """光谱数据增强"""
    
    def __init__(self, augment_prob=0.5):
        self.augment_prob = augment_prob
        
    def random_band_dropout(self, spectrum, drop_rate=0.1):
        """随机波段丢弃"""
        if np.random.random() > self.augment_prob:
            return spectrum
        
        n_bands = len(spectrum)
        n_drop = int(n_bands * drop_rate)
        drop_indices = np.random.choice(n_bands, n_drop, replace=False)
        
        augmented = spectrum.copy()
        augmented[drop_indices] = 0.0  # 或者用均值填充
        return augmented
    
    def spectral_noise(self, spectrum, noise_level=0.05):
        """添加随机噪声"""
        if np.random.random() > self.augment_prob:
            return spectrum
        
        noise = np.random.normal(0, noise_level * np.std(spectrum), len(spectrum))
        return spectrum + noise
    
    def band_scaling(self, spectrum, scale_range=(0.8, 1.2)):
        """波段缩放"""
        if np.random.random() > self.augment_prob:
            return spectrum
        
        scale = np.random.uniform(*scale_range)
        return spectrum * scale
    
    def band_shifting(self, spectrum, shift_range=(-0.1, 0.1)):
        """波段平移"""
        if np.random.random() > self.augment_prob:
            return spectrum
        
        shift = np.random.uniform(*shift_range)
        return spectrum + shift
    
    def mixup(self, spectrum1, spectrum2, label1, label2, alpha=0.2):
        """Mixup数据增强"""
        lam = np.random.beta(alpha, alpha)
        mixed_spectrum = lam * spectrum1 + (1 - lam) * spectrum2
        mixed_label = lam * label1 + (1 - lam) * label2
        return mixed_spectrum, mixed_label
    
    def random_augment(self, spectrum):
        """随机选择一种增强方法"""
        methods = [
            self.spectral_noise,
            self.band_scaling,
            self.band_shifting,
            self.random_band_dropout
        ]
        
        if np.random.random() > self.augment_prob:
            return spectrum
        
        method = np.random.choice(methods)
        
        if method == self.spectral_noise:
            return method(spectrum, noise_level=np.random.uniform(0.02, 0.08))
        elif method == self.band_scaling:
            return method(spectrum, scale_range=(0.7, 1.3))
        elif method == self.band_shifting:
            return method(spectrum, shift_range=(-0.15, 0.15))
        elif method == self.random_band_dropout:
            return method(spectrum, drop_rate=np.random.uniform(0.05, 0.2))
        
        return spectrum

def augment_minority_classes(X, y, target_multiplier=3):
    """
    对少数类别进行增强
    
    Args:
        X: 特征矩阵
        y: 标签
        target_multiplier: 目标增强倍数
    
    Returns:
        增强后的X, y
    """
    from sklearn.utils import resample
    
    unique, counts = np.unique(y, return_counts=True)
    X_augmented = []
    y_augmented = []
    
    aug = SpectralAugmentation(augment_prob=0.7)
    
    print("少数类别增强:")
    for cls, count in zip(unique, counts):
        X_cls = X[y == cls]
        y_cls = y[y == cls]
        
        # 如果是少数类别，进行增强
        if count < 1000:  # 少于1000个样本的类别
            target_count = min(count * target_multiplier, 5000)
            n_augment = target_count - count
            
            if n_augment > 0:
                # 从原始样本中随机选择进行增强
                indices = np.random.choice(len(X_cls), n_augment, replace=True)
                X_aug = []
                
                for idx in indices:
                    spectrum = X_cls[idx]
                    augmented = aug.random_augment(spectrum)
                    X_aug.append(augmented)
                
                X_aug = np.array(X_aug)
                y_aug = np.full(n_augment, cls)
                
                X_cls = np.vstack([X_cls, X_aug])
                y_cls = np.concatenate([y_cls, y_aug])
                
                print(f"  类别 {cls}: {count} -> {len(X_cls)} (增强 {n_augment}个)")
            else:
                print(f"  类别 {cls}: {count} (保持)")
        else:
            print(f"  类别 {cls}: {count} (多数类，不增强)")
        
        X_augmented.append(X_cls)
        y_augmented.append(y_cls)
    
    X_balanced = np.vstack(X_augmented)
    y_balanced = np.concatenate(y_augmented)
    
    # 打乱顺序
    indices = np.random.permutation(len(X_balanced))
    X_balanced = X_balanced[indices]
    y_balanced = y_balanced[indices]
    
    print(f"增强后: {X_balanced.shape}, 类别分布: {np.bincount(y_balanced.astype(int))}")
    
    return X_balanced, y_balanced

def test_augmentation():
    """测试数据增强"""
    print("测试数据增强...")
    
    # 创建测试数据
    n_samples = 100
    n_bands = 64
    X = np.random.randn(n_samples, n_bands)
    y = np.random.choice([0, 1, 2], size=n_samples, p=[0.7, 0.2, 0.1])
    
    print(f"原始数据: {X.shape}")
    print(f"原始分布: {np.bincount(y)}")
    
    # 测试增强
    aug = SpectralAugmentation(augment_prob=1.0)
    
    # 测试各种增强方法
    test_spectrum = X[0]
    
    print("\n增强示例:")
    print(f"原始光谱形状: {test_spectrum.shape}")
    print(f"原始均值: {test_spectrum.mean():.3f}, 标准差: {test_spectrum.std():.3f}")
    
    # 噪声增强
    noisy = aug.spectral_noise(test_spectrum, noise_level=0.05)
    print(f"噪声增强 - 均值: {noisy.mean():.3f}, 标准差: {noisy.std():.3f}")
    
    # 缩放增强
    scaled = aug.band_scaling(test_spectrum, scale_range=(0.8, 1.2))
    print(f"缩放增强 - 均值: {scaled.mean():.3f}, 标准差: {scaled.std():.3f}")
    
    # 平移增强
    shifted = aug.band_shifting(test_spectrum, shift_range=(-0.1, 0.1))
    print(f"平移增强 - 均值: {shifted.mean():.3f}, 标准差: {shifted.std():.3f}")
    
    # 随机增强
    random_aug = aug.random_augment(test_spectrum)
    print(f"随机增强 - 均值: {random_aug.mean():.3f}, 标准差: {random_aug.std():.3f}")
    
    # 测试少数类别增强
    X_aug, y_aug = augment_minority_classes(X, y, target_multiplier=2)
    print(f"\n增强后数据: {X_aug.shape}")
    print(f"增强后分布: {np.bincount(y_aug)}")
    
    print("✅ 数据增强测试完成")
    
    return True

if __name__ == "__main__":
    test_augmentation()