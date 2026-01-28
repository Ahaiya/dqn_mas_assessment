"""
Layer 1: State Encoder (感知与输入层)
=========================================
功能: 将 Layer 3 (MAS) 的非结构化评估结果转化为 Layer 2 (DQN) 可读的结构化张量。

"""

import numpy as np
import torch
from typing import List
from core.schemas import AgentOutput


class StateEncoder:
    def __init__(self, feature_dim: int = 6):
        """
        初始化状态编码器
        :param feature_dim: 状态向量维度，默认为 6
        """
        self.feature_dim = feature_dim

    def encode(self, reviews: List[AgentOutput], current_round: int) -> torch.Tensor:
        """
        将多智能体评审结果编码为状态向量 St

        特征工程设计 (6维向量):
        [0] 平均分 (Mean Score)      -> 归一化 (0-1): 反映整体质量
        [1] 分歧度 (Variance)        -> 归一化 (0-1): DQN 决策最核心依据 (是否辩论?)
        [2] 最低分 (Min Score)       -> 归一化 (0-1): 反映是否存在致命短板 (木桶效应)
        [3] 平均自信度 (Confidence)  -> 原值 (0-1):   反映专家是否犹豫
        [4] 轮次进度 (Round Progress)-> 归一化 (0-1): 告知模型时间紧迫性 (防止死循环)
        [5] 预留位 (Padding)         -> 0.0
        """
        # 0. 处理初始空状态 (第一轮尚未开始时)
        if not reviews:
            return torch.zeros(self.feature_dim, dtype=torch.float32)

        # 1. 提取数值特征
        # 提取每个专家的总分和置信度
        scores = [r.overall_score for r in reviews]
        confidences = [r.confidence for r in reviews]

        # 转换为 numpy 数组处理
        scores_np = np.array(scores) if scores else np.array([0.0])
        conf_np = np.array(confidences) if confidences else np.array([0.0])

        # 2. 特征计算与归一化 (适配 0-5 分制)
        # Max Score = 5.0

        # [0] 平均分归一化
        mean_score = np.mean(scores_np) / 5.0

        # [1] 方差归一化
        # 在 0-5 分制下，最大方差约为 6.25 (即 {0, 5} 极端对立的情况)。
        # 除以 5.0 可将其映射到 0-1.25 左右的合理区间，保留了分歧的敏感度。
        variance_score = np.var(scores_np) / 5.0

        # [2] 最低分归一化
        min_score = np.min(scores_np) / 5.0

        # [3] 平均自信度 (本身即为 0-1)
        avg_conf = np.mean(conf_np)

        # [4] 轮次特征 (假设最大允许 6 轮辩论，避免无限循环)
        # 随着轮次增加，该值趋近于 1，DQN 应倾向于 "终止/提交" 以获得时间奖励
        max_rounds = 6.0
        norm_round = min(current_round / max_rounds, 1.0)

        # [5] 预留位
        padding = 0.0

        # 3. 组装张量
        state_vector = np.array([
            mean_score,
            variance_score,
            min_score,
            avg_conf,
            norm_round,
            padding
        ], dtype=np.float32)

        # 返回 PyTorch Tensor，无需梯度 (因为这是环境状态输入)
        return torch.tensor(state_vector, dtype=torch.float32)


# 单例导出，供 Graph Node 调用
state_encoder = StateEncoder()