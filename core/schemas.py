"""
Core Schemas (Generic Multi-View Evaluator)
===========================================
通用评价框架的核心数据结构。
升级适配: ASAP 数据集 (引用材料、元数据、隐私标记处理)
"""
from enum import Enum
from typing import List, Optional, Any, Dict

# 兼容 Pydantic V2
try:
    from langchain_core.pydantic_v1 import BaseModel, Field
except ImportError:
    from pydantic import BaseModel, Field


# ==========================================
# 1. 待评主体 (The Subject)
# ==========================================

class ArtifactType(str, Enum):
    """通用材料类型"""
    TEXT_CONTENT = "text_content"  # 作文、合同、简历等纯文本
    SOURCE_CODE = "source_code"  # 代码文件
    CONVERSATION = "conversation_log"  # 对话记录
    DOCUMENT = "document"  # PDF/Word 解析后的文档
    OTHER = "other"


class AssessmentArtifact(BaseModel):
    """单个工件 (文件/段落)"""
    type: ArtifactType = Field(..., description="材料类型")
    content: str = Field(..., description="主要内容")
    filename: str = Field(..., description="文件名或标题")
    description: Optional[str] = Field(None, description="上下文说明 (如：题目要求)")


class EvaluationSubject(BaseModel):
    """
    泛化的待评主体。包含主体内容、参考材料和元数据。
    """
    subject_id: str = Field(..., description="唯一标识符 (ID)")

    # 核心内容
    artifacts: List[AssessmentArtifact] = Field(..., description="包含的所有待评材料")

    # [ASAP 核心升级] 参考材料 (原文)
    # 对于 Set 3-6 (Source Dependent)，这里存放阅读原文。
    # 对于 Set 1,2,7,8 (独立写作)，这里为空。
    reference_text: Optional[str] = Field(None, description="参考阅读材料/原文 (Source Text)")

    # 元数据 (Set ID, Max Score, Topic Context)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外的上下文元数据")

    def to_markdown_context(self) -> str:
        """动态生成 Prompt 上下文"""
        context_parts = [f"============= 📂 评估对象 (ID: {self.subject_id}) ============="]

        # 1. 注入题目背景 (Prompt/Context)
        if "context" in self.metadata:
            context_parts.append(f"📋 【题目要求/背景】\n{self.metadata['context']}\n")

        # 2. [关键] 注入阅读原文 (如果存在)
        if self.reference_text:
            context_parts.append(
                f"📖 【参考阅读材料 (Source Text)】\n请仔细阅读以下原文，评估学生是否准确引用或理解了文章：\n\n{self.reference_text}\n")
            context_parts.append("-" * 30)

        # 3. 遍历学生提交材料
        for artifact in self.artifacts:
            title = artifact.filename
            desc = f" ({artifact.description})" if artifact.description else ""
            section = f"### 📄 学生提交内容: {title}{desc}\n```\n{artifact.content}\n```"
            context_parts.append(section)

        context_parts.append("==================================================================")
        return "\n\n".join(context_parts)


# ==========================================
# 2. 评价结果 (The Feedback)
# ==========================================

class ScoreItem(BaseModel):
    """原子评分项"""
    indicator: str = Field(..., description="指标名称/代码 (e.g. Grammar, Logic)")
    score: float = Field(..., description="得分")
    evidence: str = Field(..., description="原文证据")
    comment: str = Field(..., description="评价理由")


class AgentOutput(BaseModel):
    """
    智能体输出
    关键修改：role 字段不再是 Literal，而是开放的 str
    """
    role: str = Field(
        ...,
        description="Agent的角色名称 (e.g. 'Structure_Expert', 'Content_Expert')"
    )

    thought_process: str = Field(..., description="思维链 (CoT)")
    scores: List[ScoreItem] = Field(..., description="分项打分详情")

    # 归一化后的通用得分 (0.0 - 5.0)
    overall_score: float = Field(..., ge=0, le=5, description="综合得分 (归一化到 0-5)")

    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="置信度")

    class Config:
        extra = "forbid"

    def get_low_score_items(self, threshold: float = 3.0) -> List[ScoreItem]:
        return [item for item in self.scores if item.score < threshold]