"""
Generic Agent Implementation
============================
万能评估代理。
它不再硬编码角色 (Architect/Strategist)，而是根据传入的配置动态扮演角色。
"""
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

# 引入新的通用 Schema
from core.schemas import AgentOutput, EvaluationSubject
from config.model_factory import get_core_model


class GenericAgent:
    def __init__(self, role_name: str, system_prompt: str, temperature: float = 0.0):
        """
        初始化万能代理
        :param role_name: 角色名称 (e.g. "Content_Expert")
        :param system_prompt: 已经注入了量规的完整 System Prompt
        :param temperature: 模型温度
        """
        self.role_name = role_name

        # 1. 初始化模型
        self.llm = get_core_model(temperature=temperature)

        # 2. 绑定结构化输出 (Schema)
        # 注意: AgentOutput 现在是通用的，role 字段是 str 类型，可以兼容任何角色
        self.structured_llm = self.llm.with_structured_output(AgentOutput)

        # 3. 构建 Prompt
        # input_data 将填入 EvaluationSubject.to_markdown_context() 的结果
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "【待评估对象】\n{input_data}")
        ])

        # 4. 组装链
        self.chain: Runnable = self.prompt | self.structured_llm

    def run(self, subject: EvaluationSubject, previous_reviews: Optional[list] = None) -> AgentOutput:
        """
        执行评估
        :param subject: 泛化的待评主体 (原 StudentSubmission)
        :param previous_reviews: (可选) 上一轮辩论历史
        """
        print(f"🤖 [{self.role_name}] 正在评估 {subject.subject_id} ...")

        # 1. 准备上下文
        context_str = subject.to_markdown_context()

        # 2. (可选) 注入辩论历史
        # 如果有 previous_reviews，我们需要将其拼接到 input_data 或 system prompt 中
        # 这里简化处理：直接拼接到用户输入的开头
        if previous_reviews:
            history_text = self._format_history(previous_reviews)
            final_input = f"【上一轮专家组意见 (请仔细阅读并反思)】\n{history_text}\n\n{context_str}"
        else:
            final_input = context_str

        # 3. 执行调用
        result = self.chain.invoke({"input_data": final_input})

        # 4. 强制修正角色名 (保持数据一致性)
        if result.role != self.role_name:
            result.role = self.role_name

        return result

    def _format_history(self, reviews) -> str:
        """简单的历史格式化"""
        text = ""
        for r in reviews:
            text += f"> {r.role}: {r.overall_score}分 | {r.thought_process[:50]}...\n"
        return text