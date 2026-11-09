#!/usr/bin/env python3
"""
最终检查MUUFL数据集结构
"""

import scipy.io
import numpy as np

def extract_object_data(obj):
    """提取object数组中的实际数据"""
    if isinstance(obj, np.ndarray):
        if obj.dtype == object and obj.size > 0:
            return extract_object_data(obj[0])
        else:
            return obj
    return obj

def main():
    label_path = "/home/inubashiri/02_FANG/low_light_segment_exp/data/muufl_gulfport_campus_1_hsi_220_label.mat"
    
    print("加载MUUFL场景标签文件...")
    data = scipy.io.loadmat(label_path)
    hsi_struct = data['hsi'][0, 0]
    
    print("\n" + "="*60)
    print("MUUFL GULFPORT 数据集结构确认")
    print("="*60)
    
    # 1. HSI数据立方体
    print("\n1. HSI数据立方体:")
    hsi_data = extract_object_data(hsi_struct['Data'])
    print(f"   形状: {hsi_data.shape}")
    print(f"   解释: 高度={hsi_data.shape[0]}, 宽度={hsi_data.shape[1]}, 波段={hsi_data.shape[2]}")
    print(f"   数据类型: {hsi_data.dtype}")
    print(f"   值范围: [{hsi_data.min():.4f}, {hsi_data.max():.4f}]")
    
    # 2. 标签数据
    print("\n2. 场景标签:")
    scene_labels = hsi_struct['sceneLabels'][0]
    
    # 提取标签矩阵
    labels_data = extract_object_data(scene_labels['labels'])
    print(f"   标签矩阵形状: {labels_data.shape}")
    print(f"   标签数据类型: {labels_data.dtype}")
    
    # 提取材料类型
    materials_type = extract_object_data(scene_labels['Materials_Type'])
    print(f"   材料类型数量: {len(materials_type)}")
    
    # 创建标签映射
    label_mapping = {-1: "un-labeled"}
    for i, material in enumerate(materials_type):
        if isinstance(material, np.ndarray) and material.size > 0:
            material_name = material[0]
        else:
            material_name = str(material)
        label_mapping[i+1] = material_name  # 标签从1开始
    
    print(f"\n   标签映射:")
    for label_id in sorted(label_mapping.keys()):
        print(f"     {label_id:3d}: {label_mapping[label_id]}")
    
    # 3. 数据统计
    print("\n3. 数据统计:")
    unique_vals, counts = np.unique(labels_data, return_counts=True)
    total_pixels = labels_data.size
    
    valid_pixels = 0
    for val, count in zip(unique_vals, counts):
        if val != -1:
            valid_pixels += count
        material_name = label_mapping.get(int(val), f"未知({val})")
        percentage = count / total_pixels * 100
        print(f"     {material_name:25s}: {count:6d} 像素 ({percentage:5.1f}%)")
    
    print(f"\n   有效标注像素: {valid_pixels}/{total_pixels} ({valid_pixels/total_pixels*100:.1f}%)")
    
    # 4. 数据对齐检查
    print("\n4. 数据对齐检查:")
    print(f"   HSI数据: {hsi_data.shape[0]}×{hsi_data.shape[1]}")
    print(f"   标签数据: {labels_data.shape[0]}×{labels_data.shape[1]}")
    
    if hsi_data.shape[0] == labels_data.shape[0] and hsi_data.shape[1] == labels_data.shape[1]:
        print("   ✅ 数据对齐正确")
    else:
        print("   ⚠️  注意：数据形状不匹配！")
        print(f"   HSI: {hsi_data.shape}, 标签: {labels_data.shape}")
        
        # 尝试转置
        if hsi_data.shape[0] == labels_data.shape[1] and hsi_data.shape[1] == labels_data.shape[0]:
            print("   💡 提示：标签可能需要转置")
    
    # 5. 光谱信息
    print("\n5. 光谱信息:")
    info = hsi_struct['info'][0]
    if 'wavelength' in info.dtype.names:
        wavelengths = extract_object_data(info['wavelength'])
        print(f"   波段数: {len(wavelengths)}")
        print(f"   波长范围: {float(wavelengths[0]):.1f} - {float(wavelengths[-1]):.1f} nm")
        print(f"   平均波长间隔: {float(np.mean(np.diff(wavelengths))):.1f} nm")
        
        # 检查目标范围覆盖
        target_min, target_max = 290, 1050
        coverage = np.sum((wavelengths >= target_min) & (wavelengths <= target_max)) / len(wavelengths) * 100
        print(f"   覆盖{target_min}-{target_max}nm: {coverage:.1f}%")
        
        # 显示前10个波长
        print(f"   前10个波长: {wavelengths[:10].flatten()}")
    
    # 6. 预处理建议
    print("\n" + "="*60)
    print("预处理建议:")
    print("="*60)
    print("1. 数据读取:")
    print("   - 使用 scipy.io.loadmat 读取 .mat 文件")
    print("   - 提取 hsi['Data'][0,0][0] 获取HSI数据立方体")
    print("   - 提取 hsi['sceneLabels'][0]['labels'][0] 获取标签")
    
    print("\n2. 数据对齐:")
    print("   - 检查HSI数据和标签的形状是否匹配")
    print("   - 如果需要，对标签进行转置")
    
    print("\n3. 无效像素处理:")
    print("   - 标签值为 -1 的像素是未标注的")
    print("   - 建议过滤掉这些像素或单独处理")
    
    print("\n4. 类别处理:")
    print("   - 共有11个材料类别 + 1个未标注类别")
    print("   - 考虑将标签从 1-11 映射到 0-10（机器学习标准）")
    
    print("\n5. 光谱预处理:")
    print("   - 波长范围: 405.8-1005.3nm（覆盖目标范围的71%）")
    print("   - 考虑波段归一化或标准化")
    
    print("\n6. 输入组织（ESN用）:")
    print("   - 每个像素: 64维光谱向量")
    print("   - 序列长度: 1（单时间步）")
    print("   - 样本数: ~53,687个有效像素")

if __name__ == "__main__":
    main()