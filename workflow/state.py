"""
Workflow State Definition (LangGraph V1.0+ Standard)
====================================================
定义多智能体系统的运行时状态。
关键升级：使用 Reducer (operator.add) 实现列表的自动累积。
"""

import operator
from typing import TypedDict, List, Any, Optional, Annotated
from core.schemas import AgentOutput, EvaluationSubject


class GraphState(TypedDict):
    """
    全图共享状态 (Graph State)
    """
    # 1. 输入数据 (只读)
    # 覆盖模式: 每次只有一份作业
    submission: EvaluationSubject

    # 2. 评估结果池 (Append Only / 增量更新)
    # 【V1.2.6 核心】Annotated[List, operator.add]
    # 含义: 当节点返回 {"reviews": [new_item]} 时，
    # 框架会自动执行: old_list + [new_item]
    # 解决了并发或串行执行时数据覆盖的问题。
    reviews: Annotated[List[AgentOutput], operator.add]

    # 3. 控制与监控字段 (覆盖模式)
    current_round: int              # 当前轮次
    dqn_action: int                 # DQN 决策 (0/1/2)
    dqn_debug_info: Optional[dict]  # 监控数据

    # 4. 历史记录 (自动追加)
    # 用于归档每一轮的完整状态，供下一轮辩论参考
    history: Annotated[List[Any], operator.add]