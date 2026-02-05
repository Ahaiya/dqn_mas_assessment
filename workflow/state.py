"""
Graph State Definition
======================
"""
import operator
from typing import List, Annotated, Any, Dict, Optional, Tuple, TypedDict
from core.schemas import AgentOutput, EvaluationSubject

class GraphState(TypedDict):
    # 1. 核心输入
    submission: EvaluationSubject

    # 2. 专家评价历史 (核心记忆)
    ## 使用 operator.add 实现增量更新 (Append模式)， 这样多轮辩论的记录会一直累加，供 Agent 参考
    reviews: Annotated[List[AgentOutput], operator.add]

    # 3. 流程控制
    current_round: int

    # 4. DQN 决策相关
    dqn_action: int         # 当前动作 (0:Submit, 1:Debate)
    epsilon: float          # 当前探索率 (由外部传入)

    # 5. 训练轨迹 (Hindsight Experience Replay)
    ## 使用 add 累加，记录整个 Episode 中每一次 DQN 的决策
    ## 格式: List[Tuple[StateTensor, Action]]
    dqn_trace: Annotated[List[Tuple[Any, int]], operator.add]

    # 6. 调试信息 (可选，覆盖模式即可)
    dqn_debug_info: Optional[Dict]