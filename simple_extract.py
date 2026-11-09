#!/usr/bin/env python3
"""
简单提取MUUFL数据
"""

import scipy.io
import numpy as np

def simple_extract():
    mat_path = "/home/inubashiri/02_FANG/low_light_segment_exp/data/muufl_gulfport_campus_1_hsi_220_label.mat"
    
    print("加载数据...")
    data = scipy.io.loadmat(mat_path)
    
    # 提取hsi结构
    hsi_struct = data['hsi'][0, 0]
    
    print("\n=== 字段列表 ===")
    for field in hsi_struct.dtype.names:
        print(f"  {field}: {type(hsi_struct[field])}")
    
    print("\n=== Data字段 ===")
    data_field = hsi_struct['Data']
    print(f"  类型: {type(data_field)}")
    print(f"  形状: {data_field.shape}")
    print(f"  dtype: {data_field.dtype}")
    
    print("\n=== sceneLabels字段 ===")
    scene_labels = hsi_struct['sceneLabels']
    print(f"  类型: {type(scene_labels)}")
    print(f"  形状: {scene_labels.shape}")
    print(f"  dtype: {scene_labels.dtype}")
    
    # 提取labels
    if scene_labels.dtype == object:
        print("  sceneLabels是object数组")
        if scene_labels.size > 0:
            labels_struct = scene_labels[0]
            print(f"  labels_struct类型: {type(labels_struct)}")
            print(f"  labels_struct形状: {labels_struct.shape}")
            
            if hasattr(labels_struct, 'dtype') and labels_struct.dtype.names:
                print("  labels_struct是结构化数组")
                for field in labels_struct.dtype.names:
                    print(f"    {field}: {type(labels_struct[field])}")
                
                if 'labels' in labels_struct.dtype.names:
                    labels_data = labels_struct['labels']
                    print(f"    labels字段类型: {type(labels_data)}")
                    print(f"    labels字段形状: {labels_data.shape}")
                    
                    if labels_data.dtype == object and labels_data.size > 0:
                        actual_labels = labels_data[0]
                        print(f"    实际标签形状: {actual_labels.shape}")
                        print(f"    实际标签dtype: {actual_labels.dtype}")
                        
                        # 检查内容
                        print(f"    标签唯一值: {np.unique(actual_labels)}")
    
    print("\n=== info字段 ===")
    info_field = hsi_struct['info']
    print(f"  类型: {type(info_field)}")
    print(f"  形状: {info_field.shape}")
    
    if info_field.dtype == object:
        print("  info是object数组")
        if info_field.size > 0:
            info_struct = info_field[0]
            print(f"  info_struct类型: {type(info_struct)}")
            
            if hasattr(info_struct, 'dtype') and info_struct.dtype.names:
                print("  info_struct是结构化数组")
                for field in info_struct.dtype.names:
                    print(f"    {field}: {type(info_struct[field])}")
                
                if 'wavelength' in info_struct.dtype.names:
                    wavelengths = info_struct['wavelength']
                    print(f"    wavelength字段类型: {type(wavelengths)}")
                    
                    if wavelengths.dtype == object and wavelengths.size > 0:
                        actual_wavelengths = wavelengths[0]
                        print(f"    实际波长形状: {actual_wavelengths.shape}")
                        print(f"    波长值: {actual_wavelengths[:10].flatten()}")

if __name__ == "__main__":
    simple_extract()