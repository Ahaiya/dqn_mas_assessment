"""
Dynamic Graph Construction
==========================
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END, START
from workflow.state import GraphState
from workflow.nodes import make_agent_node, debate_fanout_node
from workflow.dqn_node import dqn_decision_node
from config.loader import global_config  # 🌟 引用 Config Loader

# 1. 初始化
workflow = StateGraph(GraphState)

config_agents = global_config.get("agents", [])
agent_names = [cfg["name"] for cfg in config_agents]

# 2. 注册节点
## A. 广播/循环入口节点
workflow.add_node("debate_fanout", debate_fanout_node)
## B. 专家节点 (根据配置动态生成)
for name in agent_names:
    workflow.add_node(name, make_agent_node(name))
## C. 决策节点 (DQN)
workflow.add_node("dqn_decision", dqn_decision_node)

# 3. 定义边，逻辑：START -> Fanout -> Agents(并行) -> DQN -> (路由判断)
## 启动 -> 广播
workflow.add_edge(START, "debate_fanout")
## 广播 -> 所有专家
for name in agent_names:
    workflow.add_edge("debate_fanout", name)
## 所有专家 -> DQN (汇聚)
for name in agent_names:
    workflow.add_edge(name, "dqn_decision")


# 4. 条件路由
def route_after_decision(state: GraphState):
    """
    根据 DQN 的决策决定下一步走向
    """
    action = state.get("dqn_action", 0)
    current_round = state.get("current_round", 1)

    # 从配置读取最大轮次
    max_rounds = global_config.get("global_settings", {}).get("max_rounds", 6)

    # 强制熔断
    if current_round > max_rounds:
        print(f"🛑 达到最大轮次 ({max_rounds}) -> 强制结束")
        return "end"

    # Action 1: Debate
    if action == 1:
        return "debate_fanout"

    # Action 0: Submit
    return "end"

# 注册路由
workflow.add_conditional_edges(
    "dqn_decision",
    route_after_decision,
    {
        "debate_fanout": "debate_fanout",   # 如果返回 debate_fanout，走这里
        "end": END                          # 如果返回 end，走这里
    }
)

mas_graph = workflow.compile()