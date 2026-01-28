# 进一步封装 init_chat_model
# 目的是: 工程解耦、统一管控
import os

from langchain.chat_models import init_chat_model
from functools import lru_cache
from dotenv import load_dotenv, find_dotenv

# 加载环境变量
load_dotenv(find_dotenv(), override=True)

@lru_cache(maxsize=4) # 🌟 最佳实践: 缓存模型实例，避免重复初始化开销
def get_core_model(temperature: float = 0.0):
    """
    [单例工厂] 返回统一配置的 Chat Model。
    底层使用 LangChain 1.0 的 init_chat_model。
    """
    ## openai 类的 API 接口
    # 1. 集中读取配置 (从 .env)
    model_name = os.getenv("MODEL_NAME")
    base_url = os.getenv("OPENAI_API_BASE", None)

    # 2. 统一初始化
    print(f"🏭 ModelFactory: Loading {model_name} ...")

    if base_url:
        model = init_chat_model(
            model=model_name,
            temperature=temperature,
            openai_api_base=base_url
        )
    else:
        model = init_chat_model(
            model=model_name,
            temperature=temperature,
        )
    return model