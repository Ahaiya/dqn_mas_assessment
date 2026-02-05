"""
Mock Agent for Offline Training
===============================
"""
import random
import numpy as np
from typing import Optional
from core.schemas import AgentOutput, EvaluationSubject, ScoreItem
from config.loader import global_config


class MockAgent:
    def __init__(self, role_name: str):
        self.role_name = role_name

        # 读取模拟参数
        sim_config = global_config.get("simulation", {})
        self.convergence_rate = sim_config.get("convergence_rate", 0.8) # 辩论有效的概率（例如 0.8 表示 80% 的概率分数会变准）
        self.noise_level = sim_config.get("noise_level", 1.0) # 噪声水平（例如 1.0 表示初始瞎猜时，分数的标准差是 1.0 分）
        self.convergence_speed = sim_config.get("convergence_speed", 0.5) # 收敛速度（例如 0.5 表示每次辩论误差减少一半）

    def run(self, subject: EvaluationSubject, previous_reviews: Optional[list] = None) -> AgentOutput:
        # 1. 计算 GT (归一化到 0-5)
        # 提取原始分
        meta = subject.metadata
        raw_score = meta.get("original_score", 0)
        max_score = meta.get("raw_max_score", 10)

        if max_score == 0: max_score = 10
        # 比例缩放
        gt_score = (raw_score / max_score) * 5.0
        gt_score = max(0.0, min(5.0, gt_score))

        # 2. 决定当前分数
        current_score = 0.0
        if not previous_reviews:
            # A. 第一轮：GT + 高斯噪声（盲猜）
            noise = np.random.normal(0, self.noise_level)
            current_score = gt_score + noise
        else:
            # B. 辩论轮：查找上一轮自己的分数
            last_me = next((r for r in previous_reviews if r.role == self.role_name), None) # next(...) 是一个查找器，找到 role 匹配的上一条评论

            # 如果没找到，就重新生成
            last_score = last_me.overall_score if last_me else (gt_score + np.random.normal(0, self.noise_level))

            if random.random() < self.convergence_rate:
                # 有效辩论：向 GT 靠近
                diff = gt_score - last_score
                current_score = last_score + (diff * self.convergence_speed)
                current_score += np.random.normal(0, 0.1)  # 微小抖动
            else:
                # 无效辩论：随机波动
                current_score = last_score + np.random.normal(0, self.noise_level * 0.5)

        # 3. 截断边界
        current_score = max(0.0, min(5.0, current_score))

        # 4. 构造输出
        return AgentOutput(
            role=self.role_name,
            overall_score=round(current_score, 2),
            confidence=0.9,
            thought_process=f"[Mock] Simulation based on GT={gt_score:.2f}",
            scores=[
                ScoreItem(indicator="Mock_Metric", score=round(current_score, 2), evidence="N/A", comment="Simulated")
            ]
        )