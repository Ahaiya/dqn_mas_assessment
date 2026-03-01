"""
Generic Agent Implementation
============================
"""
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from core.schemas import AgentOutput, EvaluationSubject
from config.model_factory import get_core_model


class GenericAgent:
    def __init__(self, role_name: str, system_prompt: str, temperature: float = 0.0):
        """
        :param role_name: 角色名称 (e.g. "Content_Expert")
        :param system_prompt: 已经注入了量规的完整 System Prompt
        :param temperature: 模型温度
        """
        self.role_name = role_name

        # 1. 初始化模型
        self.llm = get_core_model(temperature=temperature)

        # 2. 绑定结构化输出 (Schema)
        self.structured_llm = self.llm.with_structured_output(AgentOutput)  # 强制 LLM 输出符合 AgentOutput 定义的 JSON 结构

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
        :param subject: 泛化的待评主体
        :param previous_reviews: (可选) 上一轮辩论历史
        """
        # 0.控制输出
        ## 计算当前轮次 (假设每轮3个专家)
        round_idx = (len(previous_reviews) // 3) + 1 if previous_reviews else 1
        ## 打印开始信号 (使用 \r 可以在同一行覆盖，或者直接用简单的 print)
        print(f"  [Round_{round_idx}] {self.role_name} 正在思考...", end="\r", flush=True)

        # 1. 准备上下文 (Markdown 格式)
        context_str = subject.to_markdown_context()

        # 2. (可选) 注入辩论历史
        # 如果有 previous_reviews，将其拼接到用户输入的开头，作为“上下文线索”
        if previous_reviews:
            history_text = self._format_history(previous_reviews)
            final_input = f"【上一轮专家组意见 (请仔细阅读并反思)】\n{history_text}\n\n{context_str}"
        else:
            final_input = context_str

        # 3. 执行调用
        try:
            result = self.chain.invoke({"input_data": final_input})

            # 🌟 [关键修复] 检查是否为 None
            if result is None:
                raise ValueError(f"Agent {self.role_name} returned None (JSON parsing failed or empty response)")

        except Exception as e:
            # 打印错误并向上抛出，train.py 的循环会捕获它并跳过当前 Episode
            print(f"\n❌ Agent {self.role_name} failed: {e}")
            raise e

        # 4. 强制修正角色名 (保持数据一致性，防止 LLM 幻觉篡改角色名)
        if result.role != self.role_name:
            result.role = self.role_name

        # 🌟 分数归一化（防御性措施）
        # 获取该 Set 的原始满分 (e.g., Set 1 是 12, Set 6 是 4)
        raw_max = subject.metadata.get("raw_max_score", 0)

        # 只有当 raw_max 存在且不为 0 时才尝试修正
        if raw_max > 0:
            # 原则：信任 Prompt，只做防御性措施
            if result.overall_score > 5.0:
                result.overall_score = (result.overall_score / raw_max) * 5.0
        # 最后的最后，确保分数在 0-5 之间
        result.overall_score = max(0.0, min(5.0, result.overall_score))

        # 打印关键结果：角色、分数、置信度
        print(f"  ✅ [Round_{round_idx}] {self.role_name.ljust(16)}: 评分 {result.overall_score:<4} (Conf: {result.confidence})")
        return result

    def _format_history(self, reviews) -> str:
        """格式化历史评价，供当前 Agent 参考"""
        text = ""
        for r in reviews:
            thought_snippet = r.thought_process[:300] + ("..." if len(r.thought_process) > 300 else "")
            text += f"> 【{r.role}】打分: {r.overall_score}\n  观点摘要: {thought_snippet}\n"
        return text
