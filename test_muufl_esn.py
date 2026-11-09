#!/usr/bin/env python3
"""
测试MUUFL ESN迁移的简化版本
"""

import numpy as np
import torch
from muufl_utils import MUUFLDataset
from sklearn.model_selection import train_test_split

def test_minimal():
    print("测试MUUFL ESN迁移的最小版本...")
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    loader = MUUFLDataset(data_dir="./data")
    spectra, labels = loader.prepare_dataset(
        normalize=True,
        remap_labels=True
    )
    
    print(f"光谱数据: {spectra.shape}")
    print(f"标签: {labels.shape}")
    print(f"类别数: {len(np.unique(labels))}")
    
    # 2. 测试ESN编码
    print("\n2. 测试ESN编码...")
    
    # 简化参数
    rng = np.random.RandomState(42)
    in_dim = spectra.shape[1]
    rff_dim = 128
    res_dim = 100
    
    # 构建RFF
    W_rff = rng.randn(in_dim, rff_dim).astype(np.float32)
    b_rff = rng.uniform(0, 2 * np.pi, size=rff_dim).astype(np.float32)
    scale = np.float32(np.sqrt(2.0 / rff_dim))
    
    # 构建ESN
    W_in = rng.randn(res_dim, rff_dim).astype(np.float32)
    W = rng.randn(res_dim, res_dim).astype(np.float32)
    sr = 0.9
    eigmax = float(np.abs(np.linalg.eigvals(W)).max())
    W *= np.float32(sr / (eigmax + 1e-12))
    leak = 0.8
    
    # 编码函数
    def encode_single(seq, W_rff, b_rff, scale, W_in, W, leak):
        """编码单个光谱序列"""
        phi = scale * np.cos(seq @ W_rff + b_rff)  # (1, rff_dim)
        
        # ESN前向传播（单时间步）
        res_dim = W.shape[0]
        state = np.zeros(res_dim, dtype=np.float32)
        h = np.tanh(W_in @ phi[0] + W @ state)
        state = (1.0 - leak) * state + leak * h
        
        return state
    
    # 测试编码少量样本
    print("编码前10个样本...")
    test_samples = spectra[:10]
    feats = []
    
    for i in range(len(test_samples)):
        seq = test_samples[i:i+1]  # (1, in_dim)
        feat = encode_single(seq, W_rff, b_rff, scale, W_in, W, leak)
        feats.append(feat)
    
    feats_array = np.vstack(feats)
    print(f"编码后特征: {feats_array.shape}")
    
    # 3. 测试数据划分
    print("\n3. 测试数据划分...")
    
    # 使用小样本测试
    X_small = spectra[:1000]
    y_small = labels[:1000]
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_small, y_small,
        test_size=0.2,
        stratify=y_small,
        random_state=42
    )
    
    print(f"训练集: {X_tr.shape}, 验证集: {X_val.shape}")
    
    # 4. 测试PyTorch环境
    print("\n4. 测试PyTorch环境...")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("使用CPU")
    
    # 5. 测试简单MLP
    print("\n5. 测试简单MLP...")
    
    class SimpleMLP(torch.nn.Module):
        def __init__(self, in_dim, hidden=64, out_dim=11):
            super().__init__()
            self.fc1 = torch.nn.Linear(in_dim, hidden)
            self.fc2 = torch.nn.Linear(hidden, out_dim)
        
        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))
    
    # 使用编码后的特征
    test_feats = torch.tensor(feats_array, dtype=torch.float32)
    test_labels = torch.tensor(labels[:10], dtype=torch.long)
    
    model = SimpleMLP(in_dim=feats_array.shape[1], hidden=32, out_dim=11)
    
    # 前向传播测试
    with torch.no_grad():
        outputs = model(test_feats)
        print(f"模型输出形状: {outputs.shape}")
        print(f"预测类别: {outputs.argmax(dim=1)}")
        print(f"真实类别: {test_labels}")
    
    print("\n✅ 所有测试通过!")
    print("下一步: 运行完整的ESN_MLP_MUUFL.py")
    
    return True

if __name__ == "__main__":
    try:
        test_minimal()
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()