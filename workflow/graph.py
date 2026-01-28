"""
Dynamic Graph Construction (ASAP Edition)
=========================================
基于 mas_config.yaml 动态构建并行评估网络。
拓扑结构：
START -> [Fanout] -> [Agents Parallel] -> [DQN] -> [Loop/End]
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from langgraph.graph import StateGraph, END, START
from workflow.state import GraphState
from workflow.nodes import make_agent_node, debate_fanout_node
from workflow.dqn_node import dqn_decision_node
from agents.factory import agent_factory

# 1. 初始化图
workflow = StateGraph(GraphState)

# 2. 读取配置，获取所有专家名称
# 注意：此时不需要指定 set_id，我们只需要知道有哪几种角色即可
# 实际运行时，Factory 会根据数据里的 set_id 动态切换内部 Prompt
config_agents = agent_factory.config.get("agents", [])
agent_names = [cfg["name"] for cfg in config_agents]

print(f"📊 Graph: 检测到 {len(agent_names)} 个专家角色 {agent_names}")

# 3. 注册节点
# A. 广播节点 (入口 & 循环点)
workflow.add_node("debate_fanout", debate_fanout_node)

# B. 专家节点 (循环注册)
for name in agent_names:
    workflow.add_node(name, make_agent_node(name))

# C. 决策节点
workflow.add_node("dqn_decision", dqn_decision_node)

# 4. 定义边 (Edges)
# 逻辑：无论是刚开始(START)还是辩论回来，都先经过 fanout，然后广播给所有专家

# START -> Fanout
workflow.add_edge(START, "debate_fanout")

# Fanout -> 所有专家 (并行)
for name in agent_names:
    workflow.add_edge("debate_fanout", name)

# 所有专家 -> DQN (汇聚)
for name in agent_names:
    workflow.add_edge(name, "dqn_decision")


# 5. 条件路由 (DQN 决策)
def route_after_decision(state: GraphState):
    action = state.get("dqn_action", 0)
    current_round = state.get("current_round", 1)

    # 获取最大轮次配置
    max_rounds = agent_factory.config.get("global_settings", {}).get("max_rounds", 6)

    # 熔断
    if current_round > max_rounds:
        print(f"🛑 达到最大轮次 ({max_rounds}) -> 强制提交")
        return "end"

    # 决策逻辑
    if action == 1 or action == 2:  # 1:Debate, 2:Hint
        # 增加轮次计数由 State Reducer 或 DQN Node 处理，这里只负责路由
        return "debate_fanout"
    else:  # 0: Submit
        print(f"✅ 达成共识/提交 -> 结束")
        return "end"


workflow.add_conditional_edges(
    "dqn_decision",
    route_after_decision,
    {
        "debate_fanout": "debate_fanout",
        "end": END
    }
)

# 6. 编译
mas_graph = workflow.compile()