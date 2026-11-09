#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MUUFL Gulfport 数据集工具函数
用于加载和预处理MUUFL高光谱数据
"""

import numpy as np
import scipy.io
from pathlib import Path
from typing import Tuple, Dict, Optional
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class MUUFLDataset:
    """MUUFL Gulfport数据集加载器"""
    
    # 材料类别映射（根据README）
    MATERIAL_CLASSES = {
        -1: "un-labeled",
        1: "trees",
        2: "grass_pure",
        3: "grass_groundsurface", 
        4: "dirt_and_sand",
        5: "road_materials",
        6: "water",
        7: "shadow_building",
        8: "buildings",
        9: "sidewalk",
        10: "yellowcurb",
        11: "cloth_panels"
    }
    
    # 简化类别名称（用于显示）
    CLASS_NAMES = {
        -1: "unlabeled",
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
    
    def __init__(self, data_dir: str = "./data"):
        """
        初始化MUUFL数据集
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.hsi_data = None
        self.labels = None
        self.wavelengths = None
        self.valid_mask = None
        
    def load_mat_file(self, mat_path: Optional[str] = None) -> Dict:
        """
        加载.mat文件
        
        Args:
            mat_path: .mat文件路径，如果为None则使用默认路径
            
        Returns:
            包含数据的字典
        """
        if mat_path is None:
            mat_path = self.data_dir / "muufl_gulfport_campus_1_hsi_220_label.mat"
        
        mat_path = Path(mat_path)
        if not mat_path.exists():
            raise FileNotFoundError(f"MAT文件不存在: {mat_path}")
        
        print(f"加载MAT文件: {mat_path}")
        data = scipy.io.loadmat(str(mat_path))
        
        return data
    
    def extract_data(self, data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        从加载的数据中提取HSI数据、标签和波长
        
        Args:
            data: scipy.io.loadmat返回的数据
            
        Returns:
            (hsi_cube, labels, wavelengths)
        """
        # 提取hsi结构
        hsi_struct = data['hsi'][0, 0]
        
        # 提取HSI数据立方体 - 直接是numpy数组
        hsi_data = hsi_struct['Data']
        print(f"HSI数据形状: {hsi_data.shape}")
        
        # 提取场景标签 - sceneLabels是结构化数组
        scene_labels = hsi_struct['sceneLabels']
        
        # sceneLabels是(1,1)的结构化数组，需要提取第一个元素
        if scene_labels.shape == (1, 1) and scene_labels.dtype.names:
            scene_labels_struct = scene_labels[0, 0]
            
            # 提取labels字段
            if 'labels' in scene_labels.dtype.names:
                labels_field = scene_labels_struct['labels']
                
                # labels字段是object数组，需要提取实际数据
                if labels_field.dtype == object and labels_field.size > 0:
                    labels = labels_field[0]
                else:
                    labels = labels_field
            else:
                raise ValueError("无法找到labels字段")
        else:
            raise ValueError("sceneLabels结构不符合预期")
        
        print(f"标签形状: {labels.shape}")
        
        # 提取波长信息 - info是(1,1)的object数组
        info = hsi_struct['info']
        if info.dtype == object and info.size > 0:
            info_struct = info[0]
            
            if info_struct.dtype.names and 'wavelength' in info_struct.dtype.names:
                wavelength_field = info_struct['wavelength']
                
                if wavelength_field.dtype == object and wavelength_field.size > 0:
                    wavelengths = wavelength_field[0]
                else:
                    wavelengths = wavelength_field
            else:
                wavelengths = None
        else:
            wavelengths = None
        
        if wavelengths is not None:
            print(f"波段数: {len(wavelengths)}")
            # 安全地打印波长范围
            try:
                wl_start = float(wavelengths[0][0]) if wavelengths[0].size > 0 else float(wavelengths[0])
                wl_end = float(wavelengths[-1][0]) if wavelengths[-1].size > 0 else float(wavelengths[-1])
                print(f"波长范围: {wl_start:.1f} - {wl_end:.1f} nm")
            except:
                print(f"波长数据: {wavelengths.shape}")
        
        # 检查数据对齐
        if hsi_data.shape[0] != labels.shape[0] or hsi_data.shape[1] != labels.shape[1]:
            print(f"⚠️  警告: HSI数据({hsi_data.shape[:2]})和标签({labels.shape})形状不匹配")
            # 尝试转置
            if hsi_data.shape[0] == labels.shape[1] and hsi_data.shape[1] == labels.shape[0]:
                print("  正在转置标签以匹配HSI数据...")
                labels = labels.T
        
        return hsi_data, labels, wavelengths
    
    def create_valid_mask(self, labels: np.ndarray) -> np.ndarray:
        """
        创建有效像素掩码（过滤未标注像素）
        
        Args:
            labels: 标签矩阵
            
        Returns:
            有效像素掩码（True表示有效）
        """
        valid_mask = labels != -1
        valid_count = np.sum(valid_mask)
        total_count = labels.size
        
        print(f"有效像素: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
        
        return valid_mask
    
    def extract_pixel_spectra(self, hsi_data: np.ndarray, labels: np.ndarray, 
                             valid_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取像素级光谱数据和对应标签
        
        Args:
            hsi_data: HSI数据立方体 (H, W, B)
            labels: 标签矩阵 (H, W)
            valid_mask: 有效像素掩码，如果为None则自动创建
            
        Returns:
            (spectra, pixel_labels) - 光谱数据和像素标签
        """
        H, W, B = hsi_data.shape
        
        if valid_mask is None:
            valid_mask = self.create_valid_mask(labels)
        
        # 获取有效像素的坐标
        valid_coords = np.where(valid_mask)
        num_valid = len(valid_coords[0])
        
        print(f"提取 {num_valid} 个有效像素的光谱...")
        
        # 预分配数组
        spectra = np.zeros((num_valid, B), dtype=hsi_data.dtype)
        pixel_labels = np.zeros(num_valid, dtype=labels.dtype)
        
        # 提取每个有效像素的光谱
        for i in range(num_valid):
            h, w = valid_coords[0][i], valid_coords[1][i]
            spectra[i] = hsi_data[h, w, :]
            pixel_labels[i] = labels[h, w]
        
        print(f"光谱数据形状: {spectra.shape}")
        print(f"标签形状: {pixel_labels.shape}")
        
        return spectra, pixel_labels
    
    def normalize_spectra(self, spectra: np.ndarray, method: str = 'standard') -> np.ndarray:
        """
        光谱数据归一化
        
        Args:
            spectra: 光谱数据 (N, B)
            method: 归一化方法
                - 'standard': 标准化 (均值0，方差1)
                - 'minmax': 最小最大归一化到[0,1]
                - 'band': 波段归一化（每个波段独立）
                
        Returns:
            归一化后的光谱数据
        """
        if method == 'standard':
            # 标准化
            mean = np.mean(spectra, axis=0)
            std = np.std(spectra, axis=0)
            std[std == 0] = 1  # 避免除零
            normalized = (spectra - mean) / std
            print(f"标准化: 均值={mean.mean():.3f}, 标准差={std.mean():.3f}")
            
        elif method == 'minmax':
            # 最小最大归一化
            min_val = np.min(spectra, axis=0)
            max_val = np.max(spectra, axis=0)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1  # 避免除零
            normalized = (spectra - min_val) / range_val
            print(f"最小最大归一化: 范围=[{min_val.min():.3f}, {max_val.max():.3f}]")
            
        elif method == 'band':
            # 波段归一化（每个波段独立）
            normalized = np.zeros_like(spectra)
            for b in range(spectra.shape[1]):
                band_data = spectra[:, b]
                band_mean = np.mean(band_data)
                band_std = np.std(band_data)
                if band_std == 0:
                    band_std = 1
                normalized[:, b] = (band_data - band_mean) / band_std
            print(f"波段归一化完成")
            
        else:
            raise ValueError(f"未知的归一化方法: {method}")
        
        return normalized
    
    def remap_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        重新映射标签：-1→-1, 1-11→0-10
        
        Args:
            labels: 原始标签
            
        Returns:
            重新映射后的标签
        """
        # 创建映射：-1保持不变，1-11映射到0-10
        remapped = labels.copy()
        
        # 将1-11映射到0-10
        mask = labels > 0
        remapped[mask] = labels[mask] - 1
        
        # 统计
        unique_vals = np.unique(remapped)
        print(f"重新映射后的标签值: {unique_vals}")
        
        return remapped
    
    def get_class_distribution(self, labels: np.ndarray) -> Dict:
        """
        获取类别分布
        
        Args:
            labels: 标签数组
            
        Returns:
            包含类别分布的字典
        """
        unique_vals, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        
        distribution = {}
        for val, count in zip(unique_vals, counts):
            if val == -1:
                class_name = "unlabeled"
            else:
                class_name = self.CLASS_NAMES.get(val + 1, f"class_{val}")  # 注意：val是0-10
            
            percentage = count / total * 100
            distribution[val] = {
                'name': class_name,
                'count': count,
                'percentage': percentage
            }
            
            print(f"  {class_name:15s}: {count:6d} ({percentage:5.1f}%)")
        
        return distribution
    
    def prepare_dataset(self, mat_path: Optional[str] = None, 
                       normalize: bool = True,
                       remap_labels: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备数据集（一站式函数）
        
        Args:
            mat_path: .mat文件路径
            normalize: 是否归一化光谱数据
            remap_labels: 是否重新映射标签
            
        Returns:
            (spectra, labels) - 准备好的光谱数据和标签
        """
        # 1. 加载数据
        data = self.load_mat_file(mat_path)
        
        # 2. 提取数据
        hsi_data, labels, wavelengths = self.extract_data(data)
        self.hsi_data = hsi_data
        self.labels = labels
        self.wavelengths = wavelengths
        
        # 3. 创建有效掩码
        valid_mask = self.create_valid_mask(labels)
        self.valid_mask = valid_mask
        
        # 4. 提取像素光谱
        spectra, pixel_labels = self.extract_pixel_spectra(hsi_data, labels, valid_mask)
        
        # 5. 重新映射标签（如果需要）
        if remap_labels:
            pixel_labels = self.remap_labels(pixel_labels)
        
        # 6. 归一化光谱（如果需要）
        if normalize:
            spectra = self.normalize_spectra(spectra, method='standard')
        
        # 7. 打印统计信息
        print("\n数据集统计:")
        print(f"  光谱数据: {spectra.shape}")
        print(f"  标签: {pixel_labels.shape}")
        print(f"  类别数: {len(np.unique(pixel_labels[pixel_labels >= 0]))}")
        
        self.get_class_distribution(pixel_labels)
        
        return spectra, pixel_labels

def test_muufl_loader():
    """测试MUUFL数据加载器"""
    print("测试MUUFL数据加载器...")
    
    # 创建数据加载器
    loader = MUUFLDataset(data_dir="./data")
    
    try:
        # 准备数据集
        spectra, labels = loader.prepare_dataset(
            mat_path="./data/muufl_gulfport_campus_1_hsi_220_label.mat",
            normalize=True,
            remap_labels=True
        )
        
        print(f"\n✅ 数据加载成功!")
        print(f"光谱数据: {spectra.shape}")
        print(f"标签: {labels.shape}")
        print(f"数据类型: {spectra.dtype}")
        print(f"标签范围: [{labels.min()}, {labels.max()}]")
        
        # 保存样本数据用于测试
        np.save("./cache/test_spectra.npy", spectra[:1000])
        np.save("./cache/test_labels.npy", labels[:1000])
        print(f"测试数据已保存到 ./cache/")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_muufl_loader()