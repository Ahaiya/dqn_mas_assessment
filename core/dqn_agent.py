"""
Layer 2: DQN Agent (决策智能体)
=========================================
功能: 封装神经网络，实现 ε-greedy 决策、经验回放与梯度更新。
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
from core.dqn_model import DQN

# 🌟 [新增] 引入 LangSmith 装饰器
from langsmith import traceable

class DQNAgent:
    def __init__(self, learning_rate=0.001, gamma=0.95, buffer_size=5000):
        """
        初始化智能体
        :param learning_rate: 学习率 (Alpha)
        :param gamma: 折扣因子 (Gamma)，决定看重眼前利益还是长远利益
        :param buffer_size: 经验回放池大小
        """
        # 1. 初始化大脑 (Policy Network)
        self.policy_net = DQN.default()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # 2. 初始化记忆 (Replay Buffer)
        self.memory = deque(maxlen=buffer_size)

        # 3. 超参数
        self.gamma = gamma
        self.action_space = [0, 1, 2]  # 0:终止, 1:辩论, 2:提示

    # 🌟 [建议] 增加监控，这对于分析模型为什么做决定至关重要
    @traceable(run_type="tool", name="DQN_Get_Q_Values")
    def get_q_values(self, state_tensor: torch.Tensor):
        """
        获取当前状态下的 Q 值，用于 LangSmith 可视化监控
        """
        with torch.no_grad():
            # 确保维度匹配 [Batch, Dim] --- unsqueeze(0)插入了 batch 维度
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            # 转成普通 Python 列表 [0.12, -0.5, 0.8]
            return q_values.squeeze().tolist()

    # 🌟 [修改] 加上 @traceable
    # run_type="tool" 表示这在 LangSmith 里会被显示为一个“工具调用”
    # name="DQN_Select" 给它起个易读的名字
    @traceable(run_type="tool", name="DQN_Inference")
    def select_action(self, state_tensor: torch.Tensor, epsilon: float = 0.1) -> int:
        """
        核心决策逻辑 (ε-greedy 策略)
        :param state_tensor: 状态向量 (Layer 1 Output)
        :param epsilon: 探索率 (0.0~1.0)，训练初期通常较高，后期降低
        :return: 动作索引 (0, 1, 2)
        """
        # 策略 A: 探索 (Explore) - 随机瞎选，为了发现新可能性
        if random.random() < epsilon:
            return random.choice(self.action_space)

        # 策略 B: 利用 (Exploit) - 听大脑的，选 Q 值最大的
        with torch.no_grad():
            # state_tensor 维度可能是 [6], 需要扩充为 [1, 6] 放入网络
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)

            q_values = self.policy_net(state_tensor)
            # 返回 Q 值最大的动作索引
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """
        记忆存储: 将一段经历存入回放池
        """
        self.memory.append((state, action, reward, next_state, done))

    # 🌟 [修改] 加上 @traceable 用于监控训练过程
    @traceable(run_type="embedding", name="DQN_Training_Step")
    def update_policy(self, batch_size=32):
        """
        自我训练: 从记忆中随机抽取片段，反向传播更新大脑
        (这是 Phase 5 训练阶段的核心，Phase 2/3 暂时只调用接口)
        """
        if len(self.memory) < batch_size:
            return None # 经验太少，先不学

        # 1. 随机抽样
        batch = random.sample(self.memory, batch_size)

        # 2. 解包数据
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*batch)

        batch_state = torch.stack(batch_state)
        batch_action = torch.tensor(batch_action).unsqueeze(1)
        batch_reward = torch.tensor(batch_reward).unsqueeze(1)
        batch_next_state = torch.stack(batch_next_state)
        batch_done = torch.tensor(batch_done, dtype=torch.float32).unsqueeze(1)

        # 3. 计算当前 Q 值 (Q_expected)
        # gather: 提取出实际执行的那个动作对应的 Q 值
        q_values = self.policy_net(batch_state)
        current_q = q_values.gather(1, batch_action)

        # 4. 计算目标 Q 值 (Q_target) -> Bellman Equation
        with torch.no_grad():
            next_q_values = self.policy_net(batch_next_state)
            max_next_q = next_q_values.max(1)[0].unsqueeze(1)
            # Q_target = Reward + Gamma * Max(Next_Q) * (1 - Done)
            expected_q = batch_reward + (self.gamma * max_next_q * (1 - batch_done))

        # 5. 计算损失 (MSE Loss)
        loss = F.mse_loss(current_q, expected_q)

        # 6. 梯度下降
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))