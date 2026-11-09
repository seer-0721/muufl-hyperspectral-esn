#!/usr/bin/env python3
"""
深度检查MUUFL .mat文件结构
"""

import scipy.io
import numpy as np

def deep_inspect(data, prefix=""):
    """递归检查嵌套结构"""
    if isinstance(data, np.ndarray):
        if data.dtype.names:  # 结构化数组
            print(f"{prefix}结构化数组，字段: {data.dtype.names}")
            for field in data.dtype.names:
                field_data = data[field]
                if isinstance(field_data, np.ndarray) and field_data.size > 0:
                    if field_data.dtype == object:
                        print(f"{prefix}  {field}: object数组 shape={field_data.shape}")
                        # 检查第一个元素
                        if field_data.size > 0:
                            elem = field_data.flat[0]
                            if isinstance(elem, np.ndarray):
                                print(f"{prefix}   第一个元素: {elem.shape} {elem.dtype}")
                            else:
                                print(f"{prefix}   第一个元素: {type(elem)}")
                    else:
                        print(f"{prefix}  {field}: {field_data.shape} {field_data.dtype}")
        else:
            if data.dtype == object:
                print(f"{prefix}object数组 shape={data.shape}")
                # 检查内容
                for i in range(min(3, data.size)):
                    elem = data.flat[i]
                    if isinstance(elem, np.ndarray):
                        print(f"{prefix}  元素[{i}]: {elem.shape} {elem.dtype}")
                        if elem.size < 10:
                            print(f"{prefix}    值: {elem}")
                    elif elem is not None:
                        print(f"{prefix}  元素[{i}]: {type(elem)}")
            else:
                print(f"{prefix}{data.shape} {data.dtype}")
                if data.size < 20:
                    print(f"{prefix}  值: {data}")
    elif isinstance(data, dict):
        for key, value in data.items():
            if not key.startswith('__'):
                print(f"{prefix}{key}: {type(value)}")
                if isinstance(value, (np.ndarray, dict)):
                    deep_inspect(value, prefix + "  ")

def main():
    filepath = "/home/inubashiri/02_FANG/low_light_segment_exp/data/muufl_gulfport_campus_w_lidar_1.mat"
    
    print("加载.mat文件...")
    data = scipy.io.loadmat(filepath)
    
    print("\n=== 顶层字段 ===")
    for key in data.keys():
        if not key.startswith('__'):
            print(f"{key}: {type(data[key])}")
    
    print("\n=== 深入检查'hsi'字段 ===")
    hsi_data = data['hsi']
    deep_inspect(hsi_data, "  ")
    
    # 尝试提取实际数据
    print("\n=== 尝试提取数据 ===")
    
    # hsi是一个1x1的结构化数组
    if hsi_data.shape == (1, 1) and hsi_data.dtype.names:
        print("hsi是1x1结构化数组")
        
        # 获取第一个（也是唯一一个）元素
        hsi_struct = hsi_data[0, 0]
        
        # 检查所有字段
        for field in hsi_data.dtype.names:
            field_data = hsi_struct[field]
            print(f"\n字段: {field}")
            
            if isinstance(field_data, np.ndarray):
                if field_data.dtype == object:
                    # object数组，需要进一步提取
                    if field_data.size > 0:
                        actual_data = field_data[0]
                        if isinstance(actual_data, np.ndarray):
                            print(f"  实际数据: {actual_data.shape} {actual_data.dtype}")
                            
                            # 特别关注关键字段
                            if field == 'Data':
                                print(f"  ⭐ HSI数据立方体: {actual_data.shape}")
                                print(f"    可能解释: 高度={actual_data.shape[0]}, 宽度={actual_data.shape[1]}, 波段={actual_data.shape[2]}")
                                print(f"    值范围: [{actual_data.min():.3f}, {actual_data.max():.3f}]")
                                
                            elif field == 'groundTruth':
                                print(f"  ⭐ 地面真值: {actual_data.shape}")
                                print(f"    唯一值: {np.unique(actual_data)}")
                                
                            elif field == 'Lidar':
                                print(f"  ⭐ LiDAR数据: {actual_data.shape}")
                                
                            elif field == 'RGB':
                                print(f"  RGB图像: {actual_data.shape}")
                                
                else:
                    print(f"  直接数据: {field_data.shape} {field_data.dtype}")
                    if field_data.size < 20:
                        print(f"    值: {field_data}")
    
    print("\n=== 检查其他字段 ===")
    for key in data.keys():
        if not key.startswith('__') and key != 'hsi':
            value = data[key]
            print(f"{key}: {type(value)}")
            if isinstance(value, np.ndarray):
                print(f"  shape: {value.shape}, dtype: {value.dtype}")
                if value.size < 10:
                    print(f"  值: {value}")

if __name__ == "__main__":
    main()