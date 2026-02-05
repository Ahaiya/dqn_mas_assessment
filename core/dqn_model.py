"""
Layer 2: DQN Policy Network (决策控制层)
=========================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Deep Q-Network (MLP架构)
    输入: State Vector (Tensor [Batch, 6])
    输出: Q-Values (Tensor [Batch, 2]) -> [Submit, Debate]
    """

    def __init__(self, state_dim=6, action_dim=2, hidden_dim=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    @classmethod
    def default(cls):
        # Action Dim = 2 (0:Submit, 1:Debate)
        return cls(state_dim=6, action_dim=2, hidden_dim=64)