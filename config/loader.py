"""
Config Loader
=============
统一负责配置文件的加载与路径解析。
"""
import yaml
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")


class ConfigLoader:
    def __init__(self, path: str):
        self._config = self._load(path)

    def _load(self, path: str) -> dict:
        if not os.path.exists(path):
            print(f"⚠️ Warning: Config file not found at {path}. CWD: {os.getcwd()}")
            return {}

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @property
    def config(self) -> dict:
        return self._config


# 单例导出，供全项目使用
_loader = ConfigLoader(CONFIG_PATH)
global_config = _loader.config