# 数据集下载链接与说明

## 优先级1：最推荐数据集

### 1. MUUFL Gulfport
**波段范围**: 375-1050 nm  
**波段数**: 64  
**场景**: 城市遥感  
**任务**: 像素级分类  

**下载信息**:
- **来源**: IEEE GRSS Data Fusion Contest 2013
- **官方链接**: https://www.grss-ieee.org/community/technical-committees/data-fusion/2013-ieee-grss-data-fusion-contest/
- **备用链接**: 
  - https://github.com/GatorSense/MUUFLGulfport
  - https://www.kaggle.com/datasets/muufl-gulfport
- **下载方式**: 需要注册，可能需等待审批
- **数据格式**: .mat (MATLAB格式)

**预处理提示**:
- 包含2个子区域：Gulfport1, Gulfport2
- 每个区域约64×64像素
- 包含地面真值标签

### 2. ICVL Hyperspectral
**波段范围**: 400-1000 nm  
**波段数**: 519  
**场景**: 通用物体  
**任务**: 重建/分类  

**下载信息**:
- **来源**: The Interdisciplinary Center Herzliya
- **官方链接**: http://icvl.cs.bgu.ac.il/hyperspectral/
- **直接下载**: http://icvl.cs.bgu.ac.il/hyperspectral/images/ICVL.h5
- **数据大小**: ~1.5GB
- **数据格式**: .h5 (HDF5格式)

**预处理提示**:
- 包含201张高光谱图像
- 每张图像1392×1300像素
- 无分类标签，需自行标注或用于无监督学习

---

## 优先级2：备选数据集

### 3. HSI-SC (Hyperspectral Image Scene Classification)
**波段范围**: 380-1050 nm  
**波段数**: 103  
**场景**: 场景分类  
**任务**: 整图分类  

**下载信息**:
- **来源**: Hyperspectral Remote Sensing Scenes
- **官方链接**: https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
- **包含数据集**:
  - Indian Pines
  - Pavia University  
  - Salinas
  - Botswana
  - KSC
  - 等
- **下载方式**: 直接下载，无需注册

**预处理提示**:
- 每个数据集包含.mat文件和地面真值
- 需要统一预处理脚本

### 4. Food-101 Hyperspectral
**波段范围**: 400-1000 nm  
**波段数**: 31  
**场景**: 食品分类  
**任务**: 整图分类  

**下载信息**:
- **来源**: Food-101扩展版本（需搜索论文）
- **相关论文**: "Hyperspectral Image Classification with Deep Learning: A Review"
- **可能来源**: 
  - GitHub仓库搜索"food hyperspectral"
  - 论文作者个人页面
- **数据格式**: 不确定，可能是.h5或.npy

**预处理提示**:
- 如果找到，检查是否有预分割的训练/测试集
- 注意食品类别的平衡性

### 5. HSI-Material (Material Classification)
**波段范围**: 400-2500 nm  
**波段数**: 224  
**场景**: 材料识别  
**任务**: 材料分类  

**下载信息**:
- **来源**: 材料分类研究数据集
- **搜索关键词**: "hyperspectral material classification dataset"
- **可能来源**:
  - IEEE DataPort
  - Mendeley Data
  - 作者GitHub
- **数据格式**: 可能是.mat或.tif

---

## 优先级3：经典基准数据集

### 6. Indian Pines
**波段范围**: 400-2500 nm  
**波段数**: 224  
**场景**: 农业遥感  
**任务**: 像素级分类  

**下载链接**:
- https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat
- https://www.ehu.eus/ccwintco/uploads/6/68/Indian_pines_gt.mat

### 7. Pavia University
**波段范围**: 430-860 nm  
**波段数**: 103  
**场景**: 城市遥感  
**任务**: 像素级分类  

**下载链接**:
- https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat
- https://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat

### 8. Salinas
**波段范围**: 400-2500 nm  
**波段数**: 224  
**场景**: 农业遥感  
**任务**: 像素级分类  

**下载链接**:
- https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat
- https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat

---

## 快速开始建议

### 立即可以开始的选项
1. **最快开始**: ICVL (直接下载，无需注册)
   - 优点：立即可用，波段密集
   - 缺点：无分类标签，需自行处理

2. **最接近目标**: 注册MUUFL的同时，先用Pavia University
   - 优点：有标签，城市场景，波段范围430-860nm
   - 缺点：缺失290-430nm和860-1050nm

3. **经典基准**: Indian Pines + Pavia University
   - 优点：广泛使用，结果可对比
   - 缺点：植物相关场景

### 下载脚本示例
```bash
# 下载ICVL
wget http://icvl.cs.bgu.ac.il/hyperspectral/images/ICVL.h5

# 下载Pavia University
wget https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat
wget https://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat

# 下载Indian Pines
wget https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat
wget https://www.ehu.eus/ccwintco/uploads/6/68/Indian_pines_gt.mat
```

### 预处理检查清单
下载后检查：
1. 文件完整性（文件大小匹配）
2. 数据格式（.mat, .h5, .tif等）
3. 波段数量（与描述一致）
4. 图像尺寸
5. 标签格式（如果有）

---

## 注意事项

### 版权与使用
1. **学术使用**: 大多数数据集允许学术研究使用
2. **商业使用**: 可能需要额外授权
3. **引用要求**: 使用数据集时需引用相关论文

### 数据质量
1. **噪声**: 检查边缘波段是否有严重噪声
2. **缺失值**: 检查是否有NaN或异常值
3. **标签一致性**: 检查标签与数据的对应关系

### 存储建议
1. **原始数据**: 保留原始下载文件
2. **预处理数据**: 保存预处理后的.npy文件
3. **中间结果**: 保存特征提取结果
4. **备份**: 定期备份重要数据

---

## 更新记录
- 2026-03-17: 创建文档，基于调研结果整理
- 后续更新：根据实际下载情况补充