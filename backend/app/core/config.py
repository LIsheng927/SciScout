"""全局配置。用环境变量驱动,方便在 OpenAI / 通义千问 / DeepSeek 之间切换而不改代码。"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""
    llm_base_url: str | None = None  # 留空 = 官方 OpenAI;填 dashscope/deepseek 的 base_url 即可切换
    llm_model: str = "gpt-4o-mini"

    max_llm_calls_per_run: int = 6  # 防止 demo/测试时意外打空余额

    arxiv_max_results: int = 5

    # 免费注册即可拿到:https://www.semanticscholar.org/product/api#Partner-Form
    # 留空也能用(走匿名共享配额),但更容易被429限流;填了之后是"专属配额",更稳定
    semantic_scholar_api_key: str = ""

    # ---- RAG 相关配置 ----
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 800       # 每个文本块大概多少字符
    chunk_overlap: int = 100    # 相邻块之间重叠多少字符,避免语义被硬切断
    rag_top_k: int = 3          # 检索时取相关度最高的几块
    qdrant_path: str = "./qdrant_data"  # 本地持久化存储路径(不用额外装数据库软件)


@lru_cache
def get_settings() -> Settings:
    return Settings()
