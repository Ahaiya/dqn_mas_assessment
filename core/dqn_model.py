"""
Layer 2: DQN Policy Network (决策控制层)
=========================================
功能: 接收 StateEncoder 的状态向量，输出动作价值 (Q-Values)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Deep Q-Network (MLP架构)

    输入: State Vector (Tensor [Batch, 6]) -> 来自 StateEncoder
    输出: Q-Values (Tensor [Batch, 3])     -> 对应 3 个宏观动作
    """

    def __init__(self, state_dim=6, action_dim=3, hidden_dim=64):
        super(DQN, self).__init__()

        # 定义全连接网络 (Fully Connected Layers)
        # 输入层 -> 隐藏层 1
        self.fc1 = nn.Linear(state_dim, hidden_dim)

        # 隐藏层 1 -> 隐藏层 2 (增加非线性表达能力)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # 隐藏层 2 -> 输出层 (Action Q-Values)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        # 初始化权重 
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        """
        前向传播逻辑
        x: 状态向量 batch
        return: 动作价值 Q(s, a)
        """
        # 第一层 + ReLU 激活
        x = F.relu(self.fc1(x))

        # 第二层 + ReLU 激活
        x = F.relu(self.fc2(x))

        # 输出层 (Q值无需激活函数，因为是回归值，可正可负)
        return self.fc3(x)

    @classmethod
    def default(cls):
        """工厂方法: 返回符合当前架构标准的默认模型
            【ht】
            具体的作用就是 在统一的地方设置默认参数，这样如果要修改也只需要改这个地方就可以，即所谓的工厂方法
            工厂方法是一种 创建型设计模式。通俗地说，它是一个 **“专门负责造对象的接口”**。
        """
        # 对应 StateEncoder 的 6 维特征
        # 对应 Technical Route 中的 3 个动作 (0:终止, 1:辩论, 2:提示) [cite: 248]
        return cls(state_dim=6, action_dim=3, hidden_dim=64)