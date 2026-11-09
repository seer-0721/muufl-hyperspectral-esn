#!/usr/bin/env python3
"""
检查MUUFL场景标签文件结构
"""

import scipy.io
import numpy as np

def main():
    label_path = "/home/inubashiri/02_FANG/low_light_segment_exp/data/muufl_gulfport_campus_1_hsi_220_label.mat"
    
    print("加载场景标签文件...")
    data = scipy.io.loadmat(label_path)
    
    print("\n=== 顶层字段 ===")
    for key in data.keys():
        if not key.startswith('__'):
            print(f"{key}: {type(data[key])}")
    
    # 检查hsi结构
    hsi_data = data['hsi']
    print(f"\n=== hsi字段结构 ===")
    
    if hsi_data.shape == (1, 1) and hsi_data.dtype.names:
        hsi_struct = hsi_data[0, 0]
        
        print("可用的字段:")
        for field in hsi_data.dtype.names:
            print(f"  - {field}")
        
        # 特别检查sceneLabels
        if 'sceneLabels' in hsi_data.dtype.names:
            scene_labels = hsi_struct['sceneLabels']
            print(f"\n=== sceneLabels结构 ===")
            
            if scene_labels.dtype.names:
                print("sceneLabels子字段:")
                for field in scene_labels.dtype.names:
                    field_data = scene_labels[field]
                    print(f"  - {field}: {type(field_data)}")
                    
                    if isinstance(field_data, np.ndarray):
                        if field_data.dtype == object:
                            if field_data.size > 0:
                                actual_data = field_data[0]
                                if isinstance(actual_data, np.ndarray):
                                    print(f"    实际数据: {actual_data.shape} {actual_data.dtype}")
                                    
                                    # 检查labels字段
                                    if field == 'labels':
                                        print(f"    ⭐ 标签数据: {actual_data.shape}")
                                        unique_vals = np.unique(actual_data)
                                        print(f"    唯一标签值: {unique_vals}")
                                        print(f"    标签值范围: [{actual_data.min()}, {actual_data.max()}]")
                                        
                                    elif field == 'Materials_Type':
                                        print(f"    ⭐ 材料类型: {actual_data}")
                                        
                                elif isinstance(actual_data, str):
                                    print(f"    字符串: {actual_data}")
                        else:
                            print(f"    直接数据: {field_data.shape} {field_data.dtype}")
                            if field_data.size < 20:
                                print(f"      值: {field_data}")
        
        # 检查Data字段
        if 'Data' in hsi_data.dtype.names:
            data_field = hsi_struct['Data']
            if data_field.dtype == object and data_field.size > 0:
                hsi_cube = data_field[0]
                print(f"\n=== HSI数据立方体 ===")
                print(f"  形状: {hsi_cube.shape}")
                print(f"  数据类型: {hsi_cube.dtype}")
                print(f"  值范围: [{hsi_cube.min():.3f}, {hsi_cube.max():.3f}]")
        
        # 检查info字段中的波长信息
        if 'info' in hsi_data.dtype.names:
            info_field = hsi_struct['info']
            if info_field.dtype.names and 'wavelength' in info_field.dtype.names:
                wavelength = info_field['wavelength'][0]
                if wavelength.dtype == object and wavelength.size > 0:
                    wavelengths = wavelength[0]
                    print(f"\n=== 波长信息 ===")
                    print(f"  波段数: {len(wavelengths)}")
                    print(f"  波长范围: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")
                    print(f"  前10个波长: {wavelengths[:10].flatten()}")
        
        # 检查groundTruth（目标检测）
        if 'groundTruth' in hsi_data.dtype.names:
            gt_field = hsi_struct['groundTruth']
            print(f"\n=== 目标检测标签 ===")
            if gt_field.dtype.names:
                print("目标检测字段:")
                for field in gt_field.dtype.names:
                    print(f"  - {field}")

if __name__ == "__main__":
    main()