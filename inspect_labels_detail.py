#!/usr/bin/env python3
"""
详细检查MUUFL场景标签
"""

import scipy.io
import numpy as np

def extract_nested_data(data):
    """提取嵌套的MATLAB数据"""
    if isinstance(data, np.ndarray):
        if data.dtype == object and data.size > 0:
            return extract_nested_data(data[0])
        elif data.dtype.names:  # 结构化数组
            result = {}
            for field in data.dtype.names:
                result[field] = extract_nested_data(data[field])
            return result
        else:
            return data
    else:
        return data

def main():
    label_path = "/home/inubashiri/02_FANG/low_light_segment_exp/data/muufl_gulfport_campus_1_hsi_220_label.mat"
    
    print("加载场景标签文件...")
    data = scipy.io.loadmat(label_path)
    
    # 提取hsi结构
    hsi_struct = data['hsi'][0, 0]
    
    # 提取sceneLabels
    scene_labels = extract_nested_data(hsi_struct['sceneLabels'])
    
    print("\n=== 场景标签详细信息 ===")
    
    # 1. 材料类型
    if 'Materials_Type' in scene_labels:
        materials = scene_labels['Materials_Type']
        if isinstance(materials, np.ndarray) and materials.dtype == object:
            materials_list = materials[0]
            print(f"材料类型 ({len(materials_list)}类):")
            for i, material in enumerate(materials_list):
                print(f"  {i+1}. {material[0]}")
    
    # 2. 标签数据
    if 'labels' in scene_labels:
        labels_data = scene_labels['labels']
        if isinstance(labels_data, np.ndarray):
            labels = labels_data[0] if labels_data.dtype == object else labels_data
            
            print(f"\n标签矩阵形状: {labels.shape}")
            print(f"标签数据类型: {labels.dtype}")
            
            # 展平并统计
            labels_flat = labels.flatten()
            unique_vals, counts = np.unique(labels_flat, return_counts=True)
            
            print(f"\n标签统计:")
            total_pixels = labels.size
            for val, count in zip(unique_vals, counts):
                percentage = count / total_pixels * 100
                print(f"  标签 {val}: {count} 像素 ({percentage:.1f}%)")
            
            # 检查有效标签（非-1）
            valid_mask = labels_flat != -1
            valid_pixels = np.sum(valid_mask)
            print(f"\n有效像素（非-1标签）: {valid_pixels}/{total_pixels} ({valid_pixels/total_pixels*100:.1f}%)")
    
    # 3. HSI数据
    if 'Data' in hsi_struct.dtype.names:
        data_field = hsi_struct['Data']
        if data_field.dtype == object and data_field.size > 0:
            hsi_cube = data_field[0]
            print(f"\n=== HSI数据立方体 ===")
            print(f"  形状: {hsi_cube.shape} (高度×宽度×波段)")
            print(f"  数据类型: {hsi_cube.dtype}")
            print(f"  值范围: {hsi_cube.min():.3f} 到 {hsi_cube.max():.3f}")
            print(f"  均值: {hsi_cube.mean():.3f}, 标准差: {hsi_cube.std():.3f}")
    
    # 4. 波长信息
    if 'info' in hsi_struct.dtype.names:
        info = extract_nested_data(hsi_struct['info'])
        if 'wavelength' in info:
            wavelengths = info['wavelength']
            if isinstance(wavelengths, np.ndarray):
                print(f"\n=== 光谱信息 ===")
                print(f"  波段数: {len(wavelengths)}")
                print(f"  波长范围: {float(wavelengths[0]):.1f} - {float(wavelengths[-1]):.1f} nm")
                print(f"  波长间隔: ~{float(wavelengths[1] - wavelengths[0]):.1f} nm")
                
                # 检查是否在目标范围内
                target_min, target_max = 290, 1050
                coverage = np.sum((wavelengths >= target_min) & (wavelengths <= target_max)) / len(wavelengths) * 100
                print(f"  覆盖290-1050nm: {coverage:.1f}%")
    
    # 5. 数据匹配检查
    print(f"\n=== 数据匹配检查 ===")
    if 'Data' in hsi_struct.dtype.names and 'labels' in scene_labels:
        data_field = hsi_struct['Data']
        if data_field.dtype == object and data_field.size > 0:
            hsi_cube = data_field[0]
            labels_data = scene_labels['labels']
            labels = labels_data[0] if labels_data.dtype == object else labels_data
            
            print(f"HSI数据形状: {hsi_cube.shape}")
            print(f"标签形状: {labels.shape}")
            
            if hsi_cube.shape[:2] == labels.shape:
                print("✅ HSI数据和标签形状匹配")
            else:
                print("❌ HSI数据和标签形状不匹配")
                
                # 尝试调整
                if hsi_cube.shape[0] == labels.shape[1] and hsi_cube.shape[1] == labels.shape[0]:
                    print("  可能需要转置标签矩阵")

if __name__ == "__main__":
    main()