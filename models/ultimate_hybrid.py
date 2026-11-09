"""
终极混合模型定义
结合ESN特征和传统特征的深度残差网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class UltimateHybridModel(nn.Module):
    """
    终极混合模型
    结合ESN特征、传统特征和深度残差网络
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: Tuple[int, int, int, int] = (1024, 512, 256, 128),
                 dropout_rates: Tuple[float, float, float, float] = (0.6, 0.5, 0.4, 0.3),
                 num_classes: int = 11):
        """
        初始化终极混合模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度 (4层)
            dropout_rates: Dropout率 (4层)
            num_classes: 输出类别数
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rates = dropout_rates
        self.num_classes = num_classes
        
        # 验证参数
        assert len(hidden_dims) == 4, "需要4个隐藏层维度"
        assert len(dropout_rates) == 4, "需要4个Dropout率"
        
        # 主网络路径
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])
        
        # 残差连接路径
        self.res_fc = nn.Linear(input_dim, hidden_dims[2])
        self.res_bn = nn.BatchNorm1d(hidden_dims[2])
        
        # 融合后网络
        self.fc4 = nn.Linear(hidden_dims[2] * 2, hidden_dims[3])  # 注意：输入是主路径+残差路径
        self.bn4 = nn.BatchNorm1d(hidden_dims[3])
        
        self.fc5 = nn.Linear(hidden_dims[3], num_classes)
        
        # Dropout层
        self.dropout1 = nn.Dropout(dropout_rates[0])
        self.dropout2 = nn.Dropout(dropout_rates[1])
        self.dropout3 = nn.Dropout(dropout_rates[2])
        self.dropout4 = nn.Dropout(dropout_rates[3])
        
        # 初始化权重
        self._initialize_weights()
        
        print(f"终极混合模型初始化:")
        print(f"  输入维度: {input_dim}")
        print(f"  网络结构: {input_dim} -> {hidden_dims[0]} -> {hidden_dims[1]} -> {hidden_dims[2]} -> {hidden_dims[3]} -> {num_classes}")
        print(f"  残差连接: {input_dim} -> {hidden_dims[2]}")
        print(f"  Dropout率: {dropout_rates}")
        print(f"  参数数量: {self._count_parameters():,}")
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _count_parameters(self) -> int:
        """计算参数数量"""
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, input_dim)
            
        Returns:
            output: 输出logits (batch_size, num_classes)
        """
        # 残差路径
        x_res = F.relu(self.res_bn(self.res_fc(x)))
        x_res = self.dropout3(x_res)
        
        # 主路径
        x_main = F.relu(self.bn1(self.fc1(x)))
        x_main = self.dropout1(x_main)
        
        x_main = F.relu(self.bn2(self.fc2(x_main)))
        x_main = self.dropout2(x_main)
        
        x_main = F.relu(self.bn3(self.fc3(x_main)))
        x_main = self.dropout3(x_main)
        
        # 融合残差
        x_fused = torch.cat([x_main, x_res], dim=1)
        
        # 继续处理
        x_fused = F.relu(self.bn4(self.fc4(x_fused)))
        x_fused = self.dropout4(x_fused)
        
        output = self.fc5(x_fused)
        
        return output
    
    def get_config(self) -> dict:
        """获取模型配置"""
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rates': self.dropout_rates,
            'num_classes': self.num_classes,
            'num_parameters': self._count_parameters()
        }


class BaselineESNModel(nn.Module):
    """
    基线ESN模型
    简单的MLP分类器
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: Tuple[int, int] = (128, 64),
                 dropout_rate: float = 0.3,
                 num_classes: int = 11):
        """
        初始化基线ESN模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度 (2层)
            dropout_rate: Dropout率
            num_classes: 输出类别数
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        
        # 网络层
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        
        self.fc3 = nn.Linear(hidden_dims[1], num_classes)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # 初始化权重
        self._initialize_weights()
        
        print(f"基线ESN模型初始化:")
        print(f"  输入维度: {input_dim}")
        print(f"  网络结构: {input_dim} -> {hidden_dims[0]} -> {hidden_dims[1]} -> {num_classes}")
        print(f"  Dropout率: {dropout_rate}")
        print(f"  参数数量: {self._count_parameters():,}")
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _count_parameters(self) -> int:
        """计算参数数量"""
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, input_dim)
            
        Returns:
            output: 输出logits (batch_size, num_classes)
        """
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        output = self.fc3(x)
        
        return output
    
    def get_config(self) -> dict:
        """获取模型配置"""
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'num_classes': self.num_classes,
            'num_parameters': self._count_parameters()
        }


def test_models():
    """测试模型"""
    print("测试模型...")
    
    try:
        # 测试数据
        batch_size = 32
        input_dim = 571
        num_classes = 11
        
        X = torch.randn(batch_size, input_dim)
        
        # 测试终极混合模型
        print("\n1. 测试终极混合模型:")
        model = UltimateHybridModel(
            input_dim=input_dim,
            hidden_dims=(1024, 512, 256, 128),
            dropout_rates=(0.6, 0.5, 0.4, 0.3),
            num_classes=num_classes
        )
        
        output = model(X)
        print(f"  输入形状: {X.shape}")
        print(f"  输出形状: {output.shape}")
        print(f"  模型配置: {model.get_config()}")
        
        # 测试基线模型
        print("\n2. 测试基线ESN模型:")
        baseline = BaselineESNModel(
            input_dim=200,  # ESN特征维度
            hidden_dims=(128, 64),
            dropout_rate=0.3,
            num_classes=num_classes
        )
        
        X_baseline = torch.randn(batch_size, 200)
        output_baseline = baseline(X_baseline)
        print(f"  输入形状: {X_baseline.shape}")
        print(f"  输出形状: {output_baseline.shape}")
        print(f"  模型配置: {baseline.get_config()}")
        
        print("\n测试通过!")
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_models()