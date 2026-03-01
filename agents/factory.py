"""
Agent Factory (Dynamic Rubric Loader & Mock Support)
====================================================
"""
import os
from typing import List, Dict, Any
from agents.generic_agent import GenericAgent
from agents.mock_agent import MockAgent
from config.loader import global_config

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUBRIC_DIR = os.path.join(BASE_DIR, "data", "rubrics")


class AgentFactory:
    def __init__(self):
        self.config = global_config
        self.agents_cache: Dict[str, Any] = {}

    def _load_rubric_content(self, set_id: int) -> str:
        filename = f"set_{set_id}.md"
        path = os.path.join(RUBRIC_DIR, filename)
        if not os.path.exists(path):
            return "（暂无特定量规，请基于常识评分）"
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def get_agent_by_name(self, name: str, set_id: int):
        """
        获取 Agent 实例 (支持 Mock 切换)
        """
        run_mode = self.config.get("run_mode", "production")

        if run_mode == "mock_training":
            # 🟢 Mock 模式：使用简单的缓存策略
            mock_key = f"mock_{name}"
            if mock_key not in self.agents_cache:
                self.agents_cache[mock_key] = MockAgent(role_name=name)
            return self.agents_cache[mock_key]

        # 🟡 生产模式
        key = f"set_{set_id}_{name}"
        if key not in self.agents_cache:
            self.get_agents(set_id)
        return self.agents_cache[key]

    def get_agents(self, set_id: int) -> List[Any]:
        run_mode = self.config.get("run_mode", "production")
        agent_names = [cfg["name"] for cfg in self.config.get("agents", [])]

        if run_mode == "mock_training":
            return [self.get_agent_by_name(name, set_id) for name in agent_names]

        # 生产模式逻辑：读取量规 -> 注入 Prompt -> 创建实例
        # print(f" Factory: 正在为 Set {set_id} 初始化专家组...")
        rubric_content = self._load_rubric_content(set_id)
        created_agents = []

        for agent_cfg in self.config.get("agents", []):
            name = agent_cfg["name"]
            template = agent_cfg["system_prompt_template"]
            full_system_prompt = template.replace("{rubric_content}", rubric_content)

            agent = GenericAgent(role_name=name, system_prompt=full_system_prompt, temperature=0.0)
            self.agents_cache[f"set_{set_id}_{name}"] = agent
            created_agents.append(agent)

        return created_agents


agent_factory = AgentFactory()