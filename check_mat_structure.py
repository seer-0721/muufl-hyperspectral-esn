#!/usr/bin/env python3
"""
检查MUUFL Gulfport .mat文件结构
"""

import scipy.io
import h5py
import numpy as np
import sys

def check_mat_version(filepath):
    """检查.mat文件版本"""
    try:
        # 尝试用scipy.io.loadmat读取（v7.2及以下）
        data = scipy.io.loadmat(filepath)
        print("✅ 使用scipy.io.loadmat成功读取（v7.2或更早版本）")
        return data, 'v7'
    except Exception as e1:
        print(f"⚠️ scipy.io.loadmat失败: {e1}")
        try:
            # 尝试用h5py读取（v7.3格式）
            data = h5py.File(filepath, 'r')
            print("✅ 使用h5py成功读取（v7.3格式）")
            return data, 'v7.3'
        except Exception as e2:
            print(f"❌ h5py也失败: {e2}")
            return None, None

def inspect_scipy_data(data):
    """检查scipy加载的数据"""
    print("\n=== 字段列表 ===")
    for key in data.keys():
        if not key.startswith('__'):
            value = data[key]
            if isinstance(value, np.ndarray):
                print(f"  {key}: {value.shape} {value.dtype}")
            else:
                print(f"  {key}: {type(value)}")
    
    # 检查可能的数据立方体
    print("\n=== 可能的数据立方体 ===")
    for key in data.keys():
        if not key.startswith('__'):
            value = data[key]
            if isinstance(value, np.ndarray) and value.ndim >= 3:
                print(f"  {key}: {value.shape} - 可能是HSI数据立方体")
    
    # 检查可能的标签
    print("\n=== 可能的标签/地面真值 ===")
    for key in data.keys():
        if not key.startswith('__'):
            value = data[key]
            if isinstance(value, np.ndarray) and value.ndim == 2:
                print(f"  {key}: {value.shape} - 可能是标签")

def inspect_h5py_data(data):
    """检查h5py加载的数据"""
    print("\n=== HDF5结构 ===")
    
    def print_structure(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"  📁 {name}/")
        elif isinstance(obj, h5py.Dataset):
            print(f"  📄 {name}: {obj.shape} {obj.dtype}")
    
    data.visititems(print_structure)
    
    # 尝试查找常见字段
    print("\n=== 查找常见字段 ===")
    common_names = ['hsi', 'cube', 'data', 'image', 'lidar', 'gt', 'label', 'truth', 'wavelength']
    
    def find_common(name, obj):
        if isinstance(obj, h5py.Dataset):
            for common in common_names:
                if common in name.lower():
                    print(f"  🔍 找到 '{common}' 在: {name} - {obj.shape} {obj.dtype}")
    
    data.visititems(find_common)

def main():
    filepath = "/home/inubashiri/02_FANG/low_light_segment_exp/data/muufl_gulfport_campus_w_lidar_1.mat"
    
    print(f"检查文件: {filepath}")
    print(f"文件大小: {np.round(os.path.getsize(filepath) / 1024 / 1024, 2)} MB")
    
    data, version = check_mat_version(filepath)
    
    if data is None:
        print("❌ 无法读取.mat文件")
        return
    
    if version == 'v7':
        inspect_scipy_data(data)
        
        # 尝试提取一些样本数据
        print("\n=== 样本数据检查 ===")
        for key in data.keys():
            if not key.startswith('__'):
                value = data[key]
                if isinstance(value, np.ndarray) and value.ndim >= 3:
                    print(f"\n{key} 的详细信息:")
                    print(f"  维度: {value.shape}")
                    print(f"  数据类型: {value.dtype}")
                    print(f"  值范围: [{value.min():.3f}, {value.max():.3f}]")
                    print(f"  均值: {value.mean():.3f}, 标准差: {value.std():.3f}")
                    
                    # 如果是3D数据，可能是HSI立方体
                    if value.ndim == 3:
                        print(f"  可能解释: 高度={value.shape[0]}, 宽度={value.shape[1]}, 波段={value.shape[2]}")
                        # 检查波段数
                        if value.shape[2] <= 100:
                            print(f"  波段数: {value.shape[2]} (可能是多光谱/高光谱)")
                    
    elif version == 'v7.3':
        inspect_h5py_data(data)
        
        # 尝试读取一些数据
        print("\n=== 数据样本 ===")
        for name in data:
            if isinstance(data[name], h5py.Dataset):
                dataset = data[name]
                if dataset.ndim >= 2 and dataset.size < 100:  # 小数据集
                    try:
                        sample = dataset[:]
                        print(f"{name}: {sample.shape}")
                        if dataset.ndim == 1 and len(sample) < 20:
                            print(f"  值: {sample}")
                    except:
                        pass

if __name__ == "__main__":
    import os
    main()