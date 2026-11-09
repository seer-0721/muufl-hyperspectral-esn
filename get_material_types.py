#!/usr/bin/env python3
"""
获取材料类型信息
"""

import scipy.io
import numpy as np

def main():
    label_path = "/home/inubashiri/02_FANG/low_light_segment_exp/data/muufl_gulfport_campus_1_hsi_220_label.mat"
    
    print("加载文件...")
    data = scipy.io.loadmat(label_path)
    hsi_struct = data['hsi'][0, 0]
    
    # 提取sceneLabels
    scene_labels = hsi_struct['sceneLabels']
    
    # 获取材料类型
    materials_type = scene_labels['Materials_Type'][0]
    
    print("\n=== MUUFL Gulfport 材料类型 ===")
    print("标签映射:")
    
    # 根据README，标签-1是'un-labeled'，然后1-11对应材料
    label_mapping = {
        -1: "un-labeled",
    }
    
    # 提取材料名称
    if materials_type.dtype == object and materials_type.size > 0:
        materials_list = materials_type[0]
        for i, material_arr in enumerate(materials_list):
            if material_arr.size > 0:
                material_name = material_arr[0]
                label_mapping[i+1] = material_name  # 标签从1开始
    
    # 打印映射
    for label_id in sorted(label_mapping.keys()):
        print(f"  标签 {label_id:2d}: {label_mapping[label_id]}")
    
    # 获取标签数据
    labels_data = scene_labels['labels'][0]
    
    print(f"\n=== 数据统计 ===")
    print(f"HSI数据形状: {hsi_struct['Data'][0].shape}")
    print(f"标签矩阵形状: {labels_data.shape}")
    
    # 统计每个类别的像素数
    unique_vals, counts = np.unique(labels_data, return_counts=True)
    total_pixels = labels_data.size
    
    print(f"\n像素分布:")
    for val, count in zip(unique_vals, counts):
        material_name = label_mapping.get(val, f"未知({val})")
        percentage = count / total_pixels * 100
        print(f"  {material_name:20s}: {count:6d} 像素 ({percentage:5.1f}%)")
    
    # 有效像素统计
    valid_pixels = np.sum(labels_data != -1)
    print(f"\n有效像素（已标注）: {valid_pixels}/{total_pixels} ({valid_pixels/total_pixels*100:.1f}%)")
    
    # 检查数据对齐
    hsi_data = hsi_struct['Data'][0]
    print(f"\n=== 数据对齐检查 ===")
    print(f"HSI数据: {hsi_data.shape[0]}×{hsi_data.shape[1]}×{hsi_data.shape[2]}")
    print(f"标签数据: {labels_data.shape[0]}×{labels_data.shape[1]}")
    
    if hsi_data.shape[0] == labels_data.shape[0] and hsi_data.shape[1] == labels_data.shape[1]:
        print("✅ 数据对齐正确")
    else:
        print("⚠️  数据形状不匹配，可能需要调整")
        
    # 波长信息
    info = hsi_struct['info'][0]
    if 'wavelength' in info.dtype.names:
        wavelengths = info['wavelength'][0]
        print(f"\n=== 光谱信息 ===")
        print(f"波段数: {len(wavelengths)}")
        print(f"波长范围: {wavelengths[0][0]:.1f} - {wavelengths[-1][0]:.1f} nm")
        
        # 检查与目标范围的匹配
        target_min, target_max = 290, 1050
        in_range = np.sum((wavelengths >= target_min) & (wavelengths <= target_max))
        print(f"在目标范围{target_min}-{target_max}nm内: {in_range}/{len(wavelengths)} 波段")

if __name__ == "__main__":
    main()