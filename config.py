"""
Configuration for the Plagiarism Check Agent.
All values can be overridden via environment variables.

Quick start:
  export OPENAI_API_KEY=sk-...           # LLM API key (required)
  export OPENAI_MODEL=gpt-4o             # LLM model (default: deepseek-chat)
  export OPENAI_BASE_URL=https://...     # custom endpoint (optional)
  export EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5  # embedding model
"""
import os
from pathlib import Path


class Config:
    # ── LLM (OpenAI-compatible) ──────────────────────────────────────────
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "deepseek-chat")
    OPENAI_BASE_URL: str = os.environ.get("OPENAI_BASE_URL",
                                          "https://api.deepseek.com/v1")
    MAX_TOKENS: int = int(os.environ.get("MAX_TOKENS", "16000"))
    THINKING_BUDGET: int = int(os.environ.get("THINKING_BUDGET", "8000"))

    # ── Embedding ────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = os.environ.get(
        "EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5"
    )
    EMBEDDING_DEVICE: str = os.environ.get("EMBEDDING_DEVICE", "cpu")

    # ── Vector Search ────────────────────────────────────────────────────
    SIMILARITY_THRESHOLD: float = float(
        os.environ.get("SIMILARITY_THRESHOLD", "0.75")
    )
    TOP_K_RESULTS: int = int(os.environ.get("TOP_K_RESULTS", "5"))
    CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", "64"))

    # ── Paths ────────────────────────────────────────────────────────────
    TRAJECTORIES_DIR: Path = Path(
        os.environ.get("TRAJECTORIES_DIR", "./trajectories")
    )
    VECTOR_DB_DIR: Path = Path(
        os.environ.get("VECTOR_DB_DIR", "./vector_db")
    )
    OUTPUT_DIR: Path = Path(
        os.environ.get("OUTPUT_DIR", "./output")
    )

    # ── Execution ────────────────────────────────────────────────────────
    DRY_RUN: bool = os.environ.get("DRY_RUN", "false").lower() == "true"
    # 设置llm思考的最大迭代次数，防止无限循环
    MAX_ITERATIONS: int = int(os.environ.get("MAX_ITERATIONS", "50"))
    # 工具返回结果的最大 token 预算，防止上下文溢出
    MAX_TOOL_RESULT_TOKENS: int = int(
        os.environ.get("MAX_TOOL_RESULT_TOKENS", "6000")
    )

    # ── Memory ─────────────────────────────────────────────────────────
    # 记忆数据目录，按类型分文件存储：
    #   chunk_hotspots.jsonl    — 跨论文热点片段
    #   common_knowledge.jsonl  — 领域常识（不应标记为重复）
    #   false_positives.jsonl   — 人工纠错记录
    #   author_profiles.jsonl   — 用户画像与查重历史
    MEMORY_DATA_DIR: Path = Path(
        os.environ.get("MEMORY_DATA_DIR", "./memory_data")
    )

    # ── External APIs ──────────────────────────────────────────────────
    # Semantic Scholar API key (optional, for paper recommendations)
    # Free tier: https://www.semanticscholar.org/product/api
    SEMANTIC_SCHOLAR_API_KEY: str = os.environ.get(
        "SEMANTIC_SCHOLAR_API_KEY", ""
    )
