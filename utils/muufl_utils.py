"""
MUUFL数据集工具函数
数据加载、预处理和基本分析
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings("ignore")


class MUUFLDataset:
    """MUUFL Gulfport高光谱数据集加载器"""
    
    # 类别名称映射
    CLASS_NAMES = {
        1: "trees",
        2: "grass_pure", 
        3: "grass_ground",
        4: "dirt_sand",
        5: "road",
        6: "water",
        7: "shadow",
        8: "buildings",
        9: "sidewalk",
        10: "yellowcurb",
        11: "cloth"
    }
    
    # 类别颜色（用于可视化）
    CLASS_COLORS = {
        1: "#228B22",  # 森林绿
        2: "#7CFC00",  # 草地绿
        3: "#32CD32",  # 酸橙绿
        4: "#D2691E",  # 巧克力色
        5: "#696969",  # 暗灰
        6: "#1E90FF",  # 道奇蓝
        7: "#2F4F4F",  # 深岩灰
        8: "#8B0000",  # 暗红
        9: "#FFD700",  # 金色
        10: "#FFFF00", # 黄色
        11: "#FF69B4"  # 热粉红
    }
    
    def __init__(self, data_dir: str = "./data"):
        """
        初始化数据集加载器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.hsi_data = None
        self.labels = None
        self.spectra = None
        self.processed_labels = None
        
    def load_mat_file(self, filename: str = "muufl_gulfport_campus_1_hsi_220_label.mat") -> Tuple[np.ndarray, np.ndarray]:
        """
        加载MAT文件
        
        Args:
            filename: MAT文件名
            
        Returns:
            hsi_data: HSI数据立方体 (height, width, bands)
            labels: 标签矩阵 (height, width)
        """
        mat_path = self.data_dir / filename
        
        if not mat_path.exists():
            raise FileNotFoundError(f"MAT文件不存在: {mat_path}")
        
        print(f"加载MAT文件: {mat_path}")
        mat_data = sio.loadmat(mat_path)
        
        # 提取HSI数据和标签
        hsi_data = mat_data['hsi']  # 形状: (325, 220, 64)
        labels = mat_data['labels']  # 形状: (325, 220)
        
        print(f"HSI数据形状: {hsi_data.shape}")
        print(f"标签形状: {labels.shape}")
        
        self.hsi_data = hsi_data
        self.labels = labels
        
        return hsi_data, labels
    
    def prepare_dataset(self, 
                       normalize: bool = True,
                       remap_labels: bool = True,
                       remove_unlabeled: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备数据集：提取有效像素并预处理
        
        Args:
            normalize: 是否标准化光谱数据
            remap_labels: 是否重新映射标签（从1-11到0-10）
            remove_unlabeled: 是否移除未标记像素（标签=0）
            
        Returns:
            spectra: 光谱数据 (n_samples, n_bands)
            labels: 标签 (n_samples,)
        """
        if self.hsi_data is None or self.labels is None:
            self.load_mat_file()
        
        hsi_data = self.hsi_data
        labels = self.labels
        
        # 提取有效像素
        if remove_unlabeled:
            # 移除未标记像素（标签=0）
            valid_mask = labels > 0
        else:
            valid_mask = np.ones_like(labels, dtype=bool)
        
        n_valid = np.sum(valid_mask)
        n_total = labels.size
        print(f"有效像素: {n_valid}/{n_total} ({n_valid/n_total*100:.1f}%)")
        
        # 展平数据
        height, width, bands = hsi_data.shape
        spectra = hsi_data.reshape(-1, bands)[valid_mask.ravel()]
        labels_flat = labels.ravel()[valid_mask.ravel()]
        
        # 重新映射标签（从1-11到0-10）
        if remap_labels:
            unique_labels = np.unique(labels_flat)
            label_mapping = {old: new for new, old in enumerate(sorted(unique_labels))}
            labels_flat = np.vectorize(label_mapping.get)(labels_flat)
            print(f"重新映射后的标签值: {np.unique(labels_flat)}")
        
        # 标准化
        if normalize:
            spectra_mean = np.mean(spectra, axis=0)
            spectra_std = np.std(spectra, axis=0) + 1e-8
            spectra = (spectra - spectra_mean) / spectra_std
            print(f"标准化: 均值={spectra_mean.mean():.3f}, 标准差={spectra_std.mean():.3f}")
        
        self.spectra = spectra
        self.processed_labels = labels_flat
        
        # 打印数据集统计信息
        self._print_dataset_stats(spectra, labels_flat)
        
        return spectra, labels_flat
    
    def _print_dataset_stats(self, spectra: np.ndarray, labels: np.ndarray):
        """打印数据集统计信息"""
        print("\n数据集统计:")
        print(f"  光谱数据: {spectra.shape}")
        print(f"  标签: {labels.shape}")
        print(f"  类别数: {len(np.unique(labels))}")
        
        # 每个类别的样本数
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        
        print("  类别分布:")
        for cls, count in zip(unique, counts):
            cls_name = self.CLASS_NAMES.get(cls + 1, f"class_{cls}")
            percentage = count / total * 100
            print(f"    {cls_name:15s}: {count:6d} ({percentage:5.1f}%)")
    
    def get_class_distribution(self) -> Dict[str, int]:
        """获取类别分布"""
        if self.processed_labels is None:
            raise ValueError("请先调用prepare_dataset()")
        
        unique, counts = np.unique(self.processed_labels, return_counts=True)
        distribution = {}
        
        for cls, count in zip(unique, counts):
            cls_name = self.CLASS_NAMES.get(cls + 1, f"class_{cls}")
            distribution[cls_name] = count
        
        return distribution
    
    def select_bands_by_variance(self, n_bands: int = 32) -> np.ndarray:
        """
        基于方差选择最重要的波段
        
        Args:
            n_bands: 要选择的波段数量
            
        Returns:
            selected_indices: 选择的波段索引
        """
        if self.spectra is None:
            raise ValueError("请先调用prepare_dataset()")
        
        # 计算每个波段的方差
        band_variances = np.var(self.spectra, axis=0)
        
        # 选择方差最大的波段
        selected_indices = np.argsort(band_variances)[-n_bands:]
        selected_indices = np.sort(selected_indices)  # 保持原始顺序
        
        print(f"波段选择: 选择{n_bands}个波段 (原始{self.spectra.shape[1]}个)")
        print(f"  选择的波段索引: {selected_indices}")
        print(f"  波段方差范围: {band_variances.min():.3f} - {band_variances.max():.3f}")
        
        return selected_indices
    
    def extract_spectral_derivative(self) -> np.ndarray:
        """
        提取光谱一阶导数
        
        Returns:
            derivative: 光谱导数 (n_samples, n_bands)
        """
        if self.spectra is None:
            raise ValueError("请先调用prepare_dataset()")
        
        derivative = np.gradient(self.spectra, axis=1)
        return derivative
    
    def extract_statistical_features(self) -> np.ndarray:
        """
        提取统计特征
        
        Returns:
            stats_features: 统计特征 (n_samples, 4)
                           [均值, 标准差, 最小值, 最大值]
        """
        if self.spectra is None:
            raise ValueError("请先调用prepare_dataset()")
        
        mean = np.mean(self.spectra, axis=1, keepdims=True)
        std = np.std(self.spectra, axis=1, keepdims=True)
        min_val = np.min(self.spectra, axis=1, keepdims=True)
        max_val = np.max(self.spectra, axis=1, keepdims=True)
        
        stats_features = np.concatenate([mean, std, min_val, max_val], axis=1)
        return stats_features


def test_muufl_loader():
    """测试MUUFL数据加载器"""
    print("测试MUUFL数据加载器...")
    
    try:
        # 创建数据加载器
        loader = MUUFLDataset(data_dir="./data")
        
        # 加载数据
        spectra, labels = loader.prepare_dataset(
            normalize=True,
            remap_labels=True,
            remove_unlabeled=True
        )
        
        print(f"\n测试通过!")
        print(f"光谱数据形状: {spectra.shape}")
        print(f"标签形状: {labels.shape}")
        print(f"类别数: {len(np.unique(labels))}")
        
        # 测试波段选择
        selected_bands = loader.select_bands_by_variance(n_bands=32)
        print(f"选择的波段: {len(selected_bands)}个")
        
        # 测试导数特征
        derivative = loader.extract_spectral_derivative()
        print(f"导数特征形状: {derivative.shape}")
        
        # 测试统计特征
        stats = loader.extract_statistical_features()
        print(f"统计特征形状: {stats.shape}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        return False


if __name__ == "__main__":
    test_muufl_loader()