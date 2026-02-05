"""
Graph Nodes Implementation
==========================
"""
import sys
import os
from typing import Dict, Any

# 确保引用路径正确
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.factory import agent_factory
from workflow.state import GraphState

def make_agent_node(agent_name: str):
    """
    工厂函数：创建一个特定角色的执行节点
    """
    def agent_node(state: GraphState) -> Dict[str, Any]:
        # 1. 获取输入
        subject = state["submission"]
        reviews = state.get("reviews", [])

        # 2. 获取 Agent 实例 (支持 Mock/Real 自动切换)
        set_id = subject.metadata.get("set_id", 1)  # set_id 用于加载对应的量规
        agent = agent_factory.get_agent_by_name(agent_name, set_id)

        # 3. 执行评估
        result = agent.run(subject, previous_reviews=reviews)   # 传入历史 reviews 供 Agent 进行辩论参考

        # 4. 返回增量更新
        ## 注意：这里返回的是列表 [result]，配合 state 中的 operator.add 实现追加
        return {"reviews": [result]}

    return agent_node

def debate_fanout_node(state: GraphState):
    """
    广播/路由节点
    目前主要用于打印日志，标记新一轮的开始
    """
    # 可以在这里做一些简单的日志打印
    # current_round = state.get("current_round", 1)
    # print(f"---  Round {current_round} Start ---")
    return {} # 不修改状态，仅作为路由锚点