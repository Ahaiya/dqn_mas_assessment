"""
Layer 2: DQN Agent (决策智能体)
=========================================
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from core.dqn_model import DQN
from config.loader import global_config

# 兼容 LangSmith
try:
    from langsmith import traceable
except ImportError:
    def traceable(**kwargs):
        def decorator(func): return func

        return decorator


class DQNAgent:
    def __init__(self):
        # 1. 从 Loader 获取配置
        config = global_config.get("training", {})

        lr = config.get("learning_rate", 0.001)
        gamma = config.get("gamma", 0.95)
        buffer_size = config.get("buffer_size", 5000)

        # 2. 初始化网络
        self.policy_net = DQN.default()
        self.target_net = DQN.default()
        self.target_net.load_state_dict(self.policy_net.state_dict())   # 将 policy_net 的所有参数（权重、偏置）完整复制给 target_net
        self.target_net.eval()  # 在反向传播时不需要计算梯度

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size)

        self.gamma = gamma
        self.action_space = [0, 1]  # 0:Submit, 1:Debate

    @traceable(run_type="tool", name="DQN_Get_Q_Values")
    def get_q_values(self, state_tensor: torch.Tensor):
        with torch.no_grad():
            # 单样本 batch 的处理
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            # 前向传播，输出结果 q_values 的形状通常是 [1, Action_Dim]（例如 [1, 2] -> [[0.8, 0.2]]）
            q_values = self.policy_net(state_tensor)
            return q_values.squeeze().tolist()  # 格式处理

    @traceable(run_type="tool", name="DQN_Inference")
    def select_action(self, state_tensor: torch.Tensor, epsilon: float = 0.1) -> int:
        # 探索机制 (Exploration)，即“瞎探索”
        if random.random() < epsilon:
            return random.choice(self.action_space)

        # 利用机制 (Exploitation）
        with torch.no_grad():
            # 维度处理
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            # 获取 Q 值，
            q_values = self.policy_net(state_tensor)
            # 贪婪选择 (Argmax)
            return q_values.argmax().item() # .item()：将 PyTorch 的 0 维张量转换为 Python 的整数（int）

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    @traceable(run_type="embedding", name="DQN_Training_Step")
    def update_policy(self, batch_size=None):
        if batch_size is None:
            batch_size = global_config.get("training", {}).get("batch_size", 32)

        if len(self.memory) < batch_size:
            return None

        batch = random.sample(self.memory, batch_size)  # 随机采样，打破数据的时间相关性（Correlation），让训练更稳定
        # 解包、格式对齐
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*batch)
        batch_state = torch.stack(batch_state)
        batch_action = torch.tensor(batch_action, dtype=torch.long).unsqueeze(1)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float32).unsqueeze(1)
        batch_next_state = torch.stack(batch_next_state)
        batch_done = torch.tensor(batch_done, dtype=torch.float32).unsqueeze(1)

        # 计算当前 Q 值 (Predicted Q)
        q_values = self.policy_net(batch_state)
        current_q = q_values.gather(1, batch_action)    # .gather(dim, index) 在 dim 这个维度上，按 index 指定的位置取值

        # 计算目标 Q 值 (Target Q) - 贝尔曼方程
        with torch.no_grad():
            next_q_values = self.target_net(batch_next_state)
            max_next_q = next_q_values.max(1)[0].unsqueeze(1)
            expected_q = batch_reward + (self.gamma * max_next_q * (1 - batch_done))

        loss = F.mse_loss(current_q, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 软更新,相比每隔 C 步暴力覆盖，这能让训练更平滑
        tau = 0.01
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

        return loss.item()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location='cpu'))