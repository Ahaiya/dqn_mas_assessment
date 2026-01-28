"""
Layer 2 Node: DQN Decision (LangGraph V1.0+ Standard)
=====================================================
负责读取 Layer 3 的所有评价，编码为状态，并做出决策。
"""
import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.encoder import state_encoder
from core.dqn_agent import DQNAgent

# 初始化全局 Agent
global_dqn_agent = DQNAgent()

def dqn_decision_node(state: dict):
    print("\n>>> 🤖 [DQN] 正在观察局势 (Monitoring)...")

    # 1. 读取累积状态
    # 得益于 state.py 的 Reducer，这里的 reviews 包含了本轮所有专家按顺序生成的评价
    reviews = state.get("reviews", [])
    current_round = state.get("current_round", 1)

    # 2. 编码 (Layer 1)
    state_tensor = state_encoder.encode(reviews, current_round)
    state_list = state_tensor.tolist()

    # 3. 决策 (Layer 2)
    ## 获取 Q 值用于监控
    try:
        q_values = global_dqn_agent.get_q_values(state_tensor)
    except Exception:
        # 兼容性保护，防止 agent 没更新导致崩溃
        q_values = [0.0, 0.0, 0.0]

    ## Epsilon-Greedy 动作选择
    action = global_dqn_agent.select_action(state_tensor, epsilon=0.1)

    # 4. 构造监控数据
    debug_info = {
        "📊 State_Features": {
            "0_Mean_Score": round(state_list[0], 2),
            "1_Variance":   round(state_list[1], 4),
            "2_Min_Score":  round(state_list[2], 2),
            "3_Confidence": round(state_list[3], 2),
            "4_Round_Prog": round(state_list[4], 2)
        },
        "🧠 Brain_Analysis": {
            "Q_Action0_Submit": round(q_values[0], 3),
            "Q_Action1_Debate": round(q_values[1], 3),
            "Q_Action2_Hint":   round(q_values[2], 3)
        },
        "🎯 Final_Decision": action,
        "Decision_Meaning": ["Submit", "Debate", "Hint"][action]
    }

    # 控制台输出
    print(f"    📊 分歧度(Var): {debug_info['📊 State_Features']['1_Variance']}")
    print(f"    🎯 决策(Action): {action} ({debug_info['Decision_Meaning']})")

    # 5. 返回更新
    ## 注意: 这里不返回 reviews，只更新控制字段
    return {
        "dqn_action": action,
        "current_round": current_round + 1,
        "dqn_debug_info": debug_info
    }