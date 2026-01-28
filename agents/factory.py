"""
Agent Factory (Dynamic Rubric Loader)
=====================================
负责根据当前的 essay_set_id，加载对应的量规文件，
并将其注入到 GenericAgent 的 System Prompt 中。
"""
import yaml
import os
from typing import List, Dict
from agents.generic_agent import GenericAgent

# 路径定义
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "mas_config.yaml")
RUBRIC_DIR = os.path.join(BASE_DIR, "data", "rubrics")  # 🌟 新的量规目录

class AgentFactory:
    def __init__(self, config_path: str = CONFIG_PATH):
        self.config = self._load_config(config_path)
        # 缓存：避免重复创建，key 为 "set_id:agent_name"
        self.agents_cache: Dict[str, GenericAgent] = {}

    def _load_config(self, path: str) -> dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ 配置文件未找到: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_rubric_content(self, set_id: int) -> str:
        """
        🌟 核心逻辑：根据 Set ID 读取对应的 .md 文件
        """
        filename = f"set_{set_id}.md"
        path = os.path.join(RUBRIC_DIR, filename)

        # 容错：如果找不到特定 Set 的量规，回退到通用量规
        if not os.path.exists(path):
            print(f"⚠️ 警告: 未找到 Set {set_id} 的量规文件，使用默认空量规。")
            return "（暂无特定量规，请基于常识评分）"

        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def get_agents(self, set_id: int) -> List[GenericAgent]:
        """
        获取适用于特定 Set 的专家组。
        如果缓存里有，直接返回；如果没有，现场生产。
        """
        # 1. 检查缓存 (我们假设一组专家只能服务于一个 Set，因为 Prompt 变了)
        cache_key = f"set_{set_id}_content_expert" # 检查其中一个即可
        if cache_key in self.agents_cache:
            # 从缓存中捞出这一组
            return [self.agents_cache[f"set_{set_id}_{agent_cfg['name']}"]
                    for agent_cfg in self.config.get("agents", [])]

        # 2. 现场生产
        print(f"🏭 Factory: 正在为 Set {set_id} 初始化专家组...")
        rubric_content = self._load_rubric_content(set_id)
        created_agents = []

        for agent_cfg in self.config.get("agents", []):
            name = agent_cfg["name"]
            template = agent_cfg["system_prompt_template"]

            # 🌟 动态注入：将 {rubric_content} 替换为当前 Set 的真实规则
            full_system_prompt = template.replace("{rubric_content}", rubric_content)

            agent = GenericAgent(
                role_name=name,
                system_prompt=full_system_prompt,
                temperature=0.0
            )

            # 存入缓存
            self.agents_cache[f"set_{set_id}_{name}"] = agent
            created_agents.append(agent)

        return created_agents

    def get_agent_by_name(self, name: str, set_id: int) -> GenericAgent:
        """
        精确获取某一个专家 (用于 Node 执行时)
        """
        key = f"set_{set_id}_{name}"
        if key not in self.agents_cache:
            # 如果没找到，说明还没初始化，强制初始化一组
            self.get_agents(set_id)
        return self.agents_cache[key]

# 单例导出
agent_factory = AgentFactory()