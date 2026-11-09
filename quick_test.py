#!/usr/bin/env python3
"""
快速测试MUUFL数据加载和基本功能
"""

import numpy as np
import torch

print("快速测试MUUFL ESN迁移...")

# 测试数据加载
print("\n1. 测试数据加载...")
try:
    from muufl_utils import MUUFLDataset
    loader = MUUFLDataset(data_dir="./data")
    spectra, labels = loader.prepare_dataset(
        normalize=True,
        remap_labels=True
    )
    print(f"✅ 数据加载成功!")
    print(f"   光谱数据: {spectra.shape}")
    print(f"   标签: {labels.shape}")
    print(f"   数据类型: {spectra.dtype}")
    print(f"   标签范围: [{labels.min()}, {labels.max()}]")
    
    # 保存小样本用于测试
    np.save("./cache/test_spectra_1000.npy", spectra[:1000])
    np.save("./cache/test_labels_1000.npy", labels[:1000])
    print(f"   测试数据已保存到 ./cache/")
    
except Exception as e:
    print(f"❌ 数据加载失败: {e}")
    import traceback
    traceback.print_exc()

# 测试PyTorch
print("\n2. 测试PyTorch环境...")
print(f"   PyTorch版本: {torch.__version__}")
print(f"   CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU设备: {torch.cuda.get_device_name(0)}")

# 测试ESN编码基础
print("\n3. 测试ESN编码基础...")
try:
    # 使用小样本
    test_spectra = spectra[:100]
    in_dim = test_spectra.shape[1]
    
    # 简化ESN参数
    rng = np.random.RandomState(42)
    rff_dim = 128
    res_dim = 100
    
    # RFF
    W_rff = rng.randn(in_dim, rff_dim).astype(np.float32)
    b_rff = rng.uniform(0, 2 * np.pi, size=rff_dim).astype(np.float32)
    scale = np.float32(np.sqrt(2.0 / rff_dim))
    
    # ESN
    W_in = rng.randn(res_dim, rff_dim).astype(np.float32)
    W = rng.randn(res_dim, res_dim).astype(np.float32)
    sr = 0.9
    eigmax = float(np.abs(np.linalg.eigvals(W)).max())
    W *= np.float32(sr / (eigmax + 1e-12))
    leak = 0.8
    
    # 编码单个样本
    def encode_single(seq):
        phi = scale * np.cos(seq @ W_rff + b_rff)
        state = np.zeros(res_dim, dtype=np.float32)
        h = np.tanh(W_in @ phi + W @ state)
        state = (1.0 - leak) * state + leak * h
        return state
    
    # 测试编码
    test_sample = test_spectra[0:1]  # (1, 64)
    encoded = encode_single(test_sample[0])  # 输入单个向量
    print(f"✅ ESN编码成功!")
    print(f"   输入形状: {test_sample.shape}")
    print(f"   输出形状: {encoded.shape}")
    print(f"   输出范围: [{encoded.min():.3f}, {encoded.max():.3f}]")
    
except Exception as e:
    print(f"❌ ESN编码测试失败: {e}")

# 测试MLP
print("\n4. 测试简单MLP...")
try:
    # 创建简单MLP
    class SimpleMLP(torch.nn.Module):
        def __init__(self, in_dim=100, hidden=64, out_dim=11):
            super().__init__()
            self.fc1 = torch.nn.Linear(in_dim, hidden)
            self.fc2 = torch.nn.Linear(hidden, out_dim)
        
        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))
    
    # 使用模拟的ESN特征
    test_feats = torch.randn(10, 100)  # 10个样本，100维特征
    test_labels = torch.randint(0, 11, (10,))
    
    model = SimpleMLP(in_dim=100, hidden=32, out_dim=11)
    
    with torch.no_grad():
        outputs = model(test_feats)
        predictions = outputs.argmax(dim=1)
        accuracy = (predictions == test_labels).float().mean()
        
    print(f"✅ MLP测试成功!")
    print(f"   模型输出形状: {outputs.shape}")
    print(f"   模拟准确率: {accuracy.item()*100:.1f}%")
    
except Exception as e:
    print(f"❌ MLP测试失败: {e}")

print("\n" + "="*50)
print("快速测试完成!")
print("下一步: 运行完整的 ESN_MLP_MUUFL.py")
print("="*50)