"""
Workflow Nodes (Dynamic Factory Pattern)
========================================
不再硬编码具体的 Agent 节点函数。
而是提供一个工厂函数，根据 Agent 名称动态生成 LangGraph 节点。
"""
import functools
from typing import Dict, Any, List

from workflow.state import GraphState
from core.schemas import AgentOutput
from agents.factory import agent_factory


def _get_latest_peer_reviews(state: GraphState) -> List[AgentOutput]:
    """[Helper] 获取上一轮的评价历史"""
    reviews = state.get("reviews", [])
    if not reviews:
        return []

    # 动态计算上一轮的数量
    # 假设每轮每个专家都发言一次，那么上一轮的评论数 = 专家总数
    # 为了保险，我们取最近产生的一批评论
    num_agents = len(agent_factory.config.get("agents", []))
    return reviews[-num_agents:] if len(reviews) >= num_agents else []


def _run_generic_agent(state: GraphState, agent_name: str) -> Dict[str, Any]:
    """
    通用 Agent 执行逻辑
    """
    print(f"    🏃 [{agent_name}] 节点启动...")
    subject = state["submission"]
    current_round = state.get("current_round", 1)
    set_id = subject.metadata.get("set_id", 1)

    # 1. 获取上一轮历史 (如果是辩论轮次)
    history = []
    if current_round > 1:
        history = _get_latest_peer_reviews(state)

    # 2. 从工厂获取实例
    agent_instance = agent_factory.get_agent_by_name(agent_name, set_id=set_id)
    if not agent_instance:
        print(f"❌ 错误: 找不到名为 {agent_name} 的 Agent 实例")
        return {}

    # 3. 执行
    try:
        result = agent_instance.run(subject, previous_reviews=history)
        return {"reviews": [result]}
    except Exception as e:
        print(f"❌ [{agent_name}] 运行崩溃: {e}")
        return {}


def make_agent_node(agent_name: str):
    """
    [高阶函数] 创建一个绑定了 agent_name 的节点函数。
    LangGraph 需要节点函数接受 state 并返回 dict。
    """
    # 使用 partial 固定 agent_name 参数
    node_func = functools.partial(_run_generic_agent, agent_name=agent_name)
    # 设置函数名，方便 LangSmith 显示
    node_func.__name__ = f"node_{agent_name}"
    return node_func


def debate_fanout_node(state: GraphState) -> Dict[str, Any]:
    """
    [广播节点] 仅仅打印日志，用于连接路由
    """
    print(f"\n📢 [System] 开启新一轮辩论 (Round {state.get('current_round', '?')})...")
    return {}
