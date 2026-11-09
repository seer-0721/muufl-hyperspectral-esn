#!/usr/bin/env python3
"""
调试数据形状问题
"""

import scipy.io
import numpy as np

def debug_shape():
    mat_path = "/home/inubashiri/02_FANG/low_light_segment_exp/data/muufl_gulfport_campus_1_hsi_220_label.mat"
    
    print("加载数据...")
    data = scipy.io.loadmat(mat_path)
    
    # 检查顶层结构
    print("\n顶层字段:")
    for key in data.keys():
        if not key.startswith('__'):
            print(f"  {key}: {type(data[key])}")
    
    # 检查hsi结构
    hsi_data = data['hsi']
    print(f"\nhsi形状: {hsi_data.shape}")
    print(f"hsi类型: {hsi_data.dtype}")
    
    if hsi_data.dtype.names:
        print("\nhsi字段:")
        for field in hsi_data.dtype.names:
            print(f"  {field}")
    
    # 提取第一个元素
    hsi_struct = hsi_data[0, 0]
    print(f"\nhsi[0,0]类型: {type(hsi_struct)}")
    
    # 检查Data字段
    if 'Data' in hsi_data.dtype.names:
        data_field = hsi_struct['Data']
        print(f"\nData字段类型: {type(data_field)}")
        print(f"Data字段形状: {data_field.shape}")
        print(f"Data字段dtype: {data_field.dtype}")
        
        # 尝试提取实际数据
        if data_field.dtype == object:
            print("Data是object数组，尝试提取...")
            if data_field.size > 0:
                actual_data = data_field[0]
                print(f"实际数据类型: {type(actual_data)}")
                if isinstance(actual_data, np.ndarray):
                    print(f"实际数据形状: {actual_data.shape}")
                    print(f"实际数据dtype: {actual_data.dtype}")
                    
                    # 如果是2D，可能是(高度×宽度, 波段)？
                    if actual_data.ndim == 2:
                        print(f"\n可能是2D数组: {actual_data.shape}")
                        print("尝试解释为 (像素数, 波段数)")
                        print(f"总像素数: {actual_data.shape[0]}")
                        print(f"波段数: {actual_data.shape[1]}")
                        
                        # 检查是否能reshape为3D
                        total_pixels = actual_data.shape[0]
                        # 尝试找到可能的维度
                        for h in [325, 220]:
                            if total_pixels % h == 0:
                                w = total_pixels // h
                                print(f"  可能reshape为: ({h}, {w}, {actual_data.shape[1]})")
    
    # 检查标签
    if 'sceneLabels' in hsi_data.dtype.names:
        scene_labels = hsi_struct['sceneLabels']
        print(f"\nsceneLabels类型: {type(scene_labels)}")
        
        if scene_labels.dtype.names and 'labels' in scene_labels.dtype.names:
            labels_field = scene_labels['labels']
            print(f"labels字段形状: {labels_field.shape}")
            
            if labels_field.dtype == object and labels_field.size > 0:
                actual_labels = labels_field[0]
                print(f"实际标签形状: {actual_labels.shape}")
                print(f"实际标签dtype: {actual_labels.dtype}")

if __name__ == "__main__":
    debug_shape()