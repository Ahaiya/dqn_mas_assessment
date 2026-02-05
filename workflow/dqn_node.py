"""
Layer 2 Node: DQN Decision
==========================
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.encoder import state_encoder
from core.dqn_agent import DQNAgent

# 初始化全局 Agent
global_dqn_agent = DQNAgent()


def dqn_decision_node(state: dict):
    # 1. 读取上下文
    reviews = state.get("reviews", []) # 本轮所有专家的评价列表
    current_round = state.get("current_round", 1)

    # 2. 状态编码
    state_tensor = state_encoder.encode(reviews, current_round)
    state_list = state_tensor.tolist()

    # 3. 获取动态探索率
    epsilon = state.get("epsilon", 0.05)

    # 4. 决策
    action = global_dqn_agent.select_action(state_tensor, epsilon=epsilon)

    # 5. 监控信息，获取当前状态的所有 Q 值，用于观察神经网络的偏好
    try:
        q_values = global_dqn_agent.get_q_values(state_tensor)
    except:
        q_values = [0.0, 0.0]

    debug_info = {
        "State_Var": round(state_list[1], 4),
        "Q_Submit": round(q_values[0], 3),
        "Q_Debate": round(q_values[1], 3),
        "Decision": ["Submit", "Debate"][action]
    }

    return {
        "dqn_action": action,
        "current_round": current_round + 1,
        "dqn_trace": [(state_tensor, action)],
        "dqn_debug_info": debug_info
    }