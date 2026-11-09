"""
ESN编码器工具
实现回声状态网络的编码功能
"""

import numpy as np
from typing import Optional, Callable
import warnings
warnings.filterwarnings("ignore")


class ESNEncoder:
    """回声状态网络编码器"""
    
    def __init__(self, 
                 input_dim: int,
                 rff_dim: int = 256,
                 reservoir_dim: int = 200,
                 spectral_radius: float = 0.9,
                 leaking_rate: float = 0.8,
                 seed: int = 42):
        """
        初始化ESN编码器
        
        Args:
            input_dim: 输入维度
            rff_dim: 随机傅里叶特征维度
            reservoir_dim: 储备池维度
            spectral_radius: 谱半径
            leaking_rate: 泄漏率
            seed: 随机种子
        """
        self.input_dim = input_dim
        self.rff_dim = rff_dim
        self.reservoir_dim = reservoir_dim
        self.spectral_radius = spectral_radius
        self.leaking_rate = leaking_rate
        self.seed = seed
        
        # 初始化随机数生成器
        self.rng = np.random.RandomState(seed)
        
        # 构建ESN参数
        self._build_esn()
        
        print(f"ESN编码器初始化:")
        print(f"  输入维度: {input_dim}")
        print(f"  RFF维度: {rff_dim}")
        print(f"  储备池维度: {reservoir_dim}")
        print(f"  谱半径: {spectral_radius:.3f}")
        print(f"  泄漏率: {leaking_rate:.3f}")
        print(f"  随机种子: {seed}")
    
    def _build_esn(self):
        """构建ESN参数"""
        # 1. 随机傅里叶特征 (RFF)
        self.W_rff = self.rng.randn(self.input_dim, self.rff_dim).astype(np.float32)
        self.b_rff = self.rng.uniform(0, 2 * np.pi, size=self.rff_dim).astype(np.float32)
        self.scale = np.float32(np.sqrt(2.0 / self.rff_dim))
        
        # 2. 输入权重
        self.W_in = self.rng.randn(self.reservoir_dim, self.rff_dim).astype(np.float32)
        
        # 3. 储备池权重（确保谱半径）
        W = self.rng.randn(self.reservoir_dim, self.reservoir_dim).astype(np.float32)
        eigmax = float(np.abs(np.linalg.eigvals(W)).max())
        self.W = W * np.float32(self.spectral_radius / (eigmax + 1e-12))
    
    def encode(self, X: np.ndarray, batch_size: int = 2000) -> np.ndarray:
        """
        编码输入数据
        
        Args:
            X: 输入数据 (n_samples, input_dim)
            batch_size: 批处理大小
            
        Returns:
            encoded: 编码后的特征 (n_samples, reservoir_dim)
        """
        n_samples = len(X)
        encoded_features = []
        
        print(f"编码 {n_samples} 个样本...")
        
        for i in range(0, n_samples, batch_size):
            batch = X[i:i+batch_size]
            batch_encoded = self._encode_batch(batch)
            encoded_features.append(batch_encoded)
            
            # 显示进度
            if (i // batch_size) % 10 == 0:
                print(f"  进度: {min(i+batch_size, n_samples)}/{n_samples}")
        
        encoded = np.vstack(encoded_features)
        print(f"编码完成: {encoded.shape}")
        
        return encoded
    
    def _encode_batch(self, batch: np.ndarray) -> np.ndarray:
        """编码单个批次"""
        batch_size = len(batch)
        
        # 1. 随机傅里叶特征变换
        phi = self.scale * np.cos(batch @ self.W_rff + self.b_rff)
        
        # 2. ESN状态演化
        states = np.zeros((batch_size, self.reservoir_dim), dtype=np.float32)
        
        for i in range(batch_size):
            # 计算新的状态
            h = np.tanh(self.W_in @ phi[i] + self.W @ states[i])
            # 泄漏积分
            states[i] = (1.0 - self.leaking_rate) * states[i] + self.leaking_rate * h
        
        return states
    
    def get_config(self) -> dict:
        """获取ESN配置"""
        return {
            'input_dim': self.input_dim,
            'rff_dim': self.rff_dim,
            'reservoir_dim': self.reservoir_dim,
            'spectral_radius': self.spectral_radius,
            'leaking_rate': self.leaking_rate,
            'seed': self.seed
        }


class MultiESNEncoder:
    """多ESN编码器（用于集成）"""
    
    def __init__(self):
        """初始化多ESN编码器"""
        self.encoders = []
        self.configs = []
    
    def add_encoder(self, encoder: ESNEncoder):
        """添加ESN编码器"""
        self.encoders.append(encoder)
        self.configs.append(encoder.get_config())
    
    def encode_all(self, X: np.ndarray, batch_size: int = 2000) -> list:
        """
        使用所有ESN编码数据
        
        Args:
            X: 输入数据
            batch_size: 批处理大小
            
        Returns:
            encoded_list: 每个ESN的编码结果列表
        """
        encoded_list = []
        
        for i, encoder in enumerate(self.encoders):
            print(f"\n编码器 {i+1}/{len(self.encoders)}:")
            encoded = encoder.encode(X, batch_size)
            encoded_list.append(encoded)
        
        return encoded_list
    
    def encode_concatenated(self, X: np.ndarray, batch_size: int = 2000) -> np.ndarray:
        """
        编码并拼接所有特征
        
        Args:
            X: 输入数据
            batch_size: 批处理大小
            
        Returns:
            concatenated: 拼接后的特征
        """
        encoded_list = self.encode_all(X, batch_size)
        concatenated = np.concatenate(encoded_list, axis=1)
        
        print(f"\n特征拼接: {concatenated.shape}")
        print(f"  总特征维度: {concatenated.shape[1]}")
        
        return concatenated
    
    def get_ensemble_configs(self) -> list:
        """获取所有ESN配置"""
        return self.configs


def create_diverse_esn_ensemble(input_dim: int, n_esn: int = 3) -> MultiESNEncoder:
    """
    创建多样性ESN集成
    
    Args:
        input_dim: 输入维度
        n_esn: ESN数量
        
    Returns:
        ensemble: 多ESN编码器
    """
    ensemble = MultiESNEncoder()
    
    # 基础配置
    base_configs = [
        {'rff_dim': 256, 'reservoir_dim': 200, 'spectral_radius': 0.9, 'leaking_rate': 0.8, 'seed': 42},
        {'rff_dim': 256, 'reservoir_dim': 180, 'spectral_radius': 0.85, 'leaking_rate': 0.7, 'seed': 142},
        {'rff_dim': 256, 'reservoir_dim': 150, 'spectral_radius': 0.8, 'leaking_rate': 0.6, 'seed': 242},
        {'rff_dim': 256, 'reservoir_dim': 220, 'spectral_radius': 0.95, 'leaking_rate': 0.9, 'seed': 342},
        {'rff_dim': 256, 'reservoir_dim': 170, 'spectral_radius': 0.75, 'leaking_rate': 0.5, 'seed': 442}
    ]
    
    # 创建指定数量的ESN
    for i in range(min(n_esn, len(base_configs))):
        config = base_configs[i]
        encoder = ESNEncoder(
            input_dim=input_dim,
            rff_dim=config['rff_dim'],
            reservoir_dim=config['reservoir_dim'],
            spectral_radius=config['spectral_radius'],
            leaking_rate=config['leaking_rate'],
            seed=config['seed']
        )
        ensemble.add_encoder(encoder)
    
    print(f"\n创建了 {len(ensemble.encoders)} 个多样性ESN")
    
    return ensemble


def test_esn_encoder():
    """测试ESN编码器"""
    print("测试ESN编码器...")
    
    try:
        # 创建测试数据
        np.random.seed(42)
        n_samples = 100
        input_dim = 64
        X = np.random.randn(n_samples, input_dim).astype(np.float32)
        
        # 测试单个ESN
        print("\n1. 测试单个ESN:")
        esn = ESNEncoder(
            input_dim=input_dim,
            rff_dim=128,
            reservoir_dim=100,
            spectral_radius=0.9,
            leaking_rate=0.8,
            seed=42
        )
        
        encoded = esn.encode(X, batch_size=50)
        print(f"  输入形状: {X.shape}")
        print(f"  输出形状: {encoded.shape}")
        
        # 测试多ESN集成
        print("\n2. 测试多ESN集成:")
        ensemble = create_diverse_esn_ensemble(input_dim, n_esn=3)
        
        concatenated = ensemble.encode_concatenated(X, batch_size=50)
        print(f"  拼接特征形状: {concatenated.shape}")
        
        # 测试配置获取
        print("\n3. 测试配置获取:")
        config = esn.get_config()
        print(f"  ESN配置: {config}")
        
        ensemble_configs = ensemble.get_ensemble_configs()
        print(f"  集成配置数量: {len(ensemble_configs)}")
        
        print("\n测试通过!")
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_esn_encoder()