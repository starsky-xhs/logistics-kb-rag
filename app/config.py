"""运行时配置。

刻意不引入 pydantic-settings：配置项不多，手写解析更容易解释，也让核心包保持
零第三方依赖。所有环境变量都在 ``.env.example`` 中有对应说明。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Final

from .errors import ConfigError

_TRUE_VALUES: Final = {"1", "true", "yes", "on"}
_FALSE_VALUES: Final = {"0", "false", "no", "off"}


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or not value.strip() else value.strip()


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} 必须是整数，当前值为 {raw!r}") from exc
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} 必须在 {minimum} 到 {maximum} 之间，当前值为 {value}")
    return value


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} 必须是数字，当前值为 {raw!r}") from exc
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} 必须在 {minimum} 到 {maximum} 之间，当前值为 {value}")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ConfigError(f"{name} 必须是布尔值（true/false），当前值为 {raw!r}")


@dataclass(slots=True)
class Settings:
    """服务全部可调参数。

    默认值对应完全离线的演示模式：哈希向量 + 模板生成，不需要网络和密钥。
    """

    db_path: Path = Path("data/kb.sqlite3")
    top_k: int = 5
    candidate_k: int = 20
    chunk_size: int = 650
    chunk_overlap: int = 120
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    rrf_k: int = 60
    embed_provider: str = "hash"
    embed_dim: int = 512
    embed_salt: str = "lkb-hash-v1"
    embed_batch_size: int = 16
    llm_provider: str = "mock"
    llm_api_base: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    embed_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embed_api_key: str = ""
    embed_model: str = "text-embedding-v4"
    request_timeout: float = 20.0
    max_retries: int = 2
    temperature: float = 0.2
    max_tokens: int = 800
    max_context_chars: int = 6000
    rerank_provider: str = "heuristic"
    rerank_api_base: str = "https://api.siliconflow.cn/v1"
    rerank_api_key: str = ""
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_fallback: bool = True
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path).expanduser()
        if not isinstance(self.top_k, int) or self.top_k < 1:
            raise ConfigError("top_k 必须是正整数")
        if self.candidate_k < self.top_k:
            raise ConfigError("candidate_k 不能小于 top_k")
        if self.chunk_size < 120:
            raise ConfigError("chunk_size 不能小于 120，否则中文语义会被切碎")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ConfigError("chunk_overlap 必须大于等于 0 且小于 chunk_size")
        if self.chunk_overlap > self.chunk_size // 2:
            raise ConfigError("chunk_overlap 不应超过 chunk_size 的一半")
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigError("temperature 必须在 0 到 2 之间")
        if not 0.0 < self.request_timeout <= 300.0:
            raise ConfigError("request_timeout 必须在 0 到 300 秒之间")
        if not 0 <= self.max_retries <= 5:
            raise ConfigError("max_retries 必须在 0 到 5 之间")
        if self.embed_provider not in {"hash", "http"}:
            raise ConfigError("EMBED_PROVIDER 只支持 hash 或 http")
        if self.llm_provider not in {"mock", "http"}:
            raise ConfigError("LLM_PROVIDER 只支持 mock 或 http")
        if self.rerank_provider not in {"off", "heuristic", "http"}:
            raise ConfigError("RERANK_PROVIDER 只支持 off、heuristic 或 http")
        if self.embed_provider == "http" and not self.embed_api_key:
            raise ConfigError("EMBED_PROVIDER=http 时必须设置 EMBED_API_KEY")
        if self.llm_provider == "http" and not self.llm_api_key:
            raise ConfigError("LLM_PROVIDER=http 时必须设置 LLM_API_KEY")
        if self.rerank_provider == "http" and not self.rerank_api_key:
            raise ConfigError("RERANK_PROVIDER=http 时必须设置 RERANK_API_KEY")

    @classmethod
    def from_env(cls, **overrides: object) -> Settings:
        """从环境变量构造配置，``overrides`` 供测试与 demo 定向覆盖。"""

        data: dict[str, object] = {
            "db_path": Path(_env_str("KB_DB_PATH", "data/kb.sqlite3")),
            "top_k": _env_int("RAG_TOP_K", 5, minimum=1, maximum=50),
            "candidate_k": _env_int("RAG_CANDIDATE_K", 20, minimum=1, maximum=200),
            "chunk_size": _env_int("RAG_CHUNK_SIZE", 650, minimum=120, maximum=5000),
            "chunk_overlap": _env_int("RAG_CHUNK_OVERLAP", 120, minimum=0, maximum=2000),
            "bm25_k1": _env_float("BM25_K1", 1.5, minimum=0.1, maximum=5.0),
            "bm25_b": _env_float("BM25_B", 0.75, minimum=0.0, maximum=1.0),
            "rrf_k": _env_int("RRF_K", 60, minimum=1, maximum=1000),
            "embed_provider": _env_str("EMBED_PROVIDER", "hash").lower(),
            "embed_dim": _env_int("EMBED_DIM", 512, minimum=64, maximum=4096),
            "embed_salt": _env_str("EMBED_SALT", "lkb-hash-v1"),
            "embed_batch_size": _env_int("EMBED_BATCH_SIZE", 16, minimum=1, maximum=256),
            "llm_provider": _env_str("LLM_PROVIDER", "mock").lower(),
            "llm_api_base": _env_str("LLM_API_BASE", "https://api.deepseek.com/v1"),
            "llm_api_key": _env_str("LLM_API_KEY", ""),
            "llm_model": _env_str("LLM_MODEL", "deepseek-chat"),
            "embed_api_base": _env_str(
                "EMBED_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
            "embed_api_key": _env_str("EMBED_API_KEY", ""),
            "embed_model": _env_str("EMBED_MODEL", "text-embedding-v4"),
            "request_timeout": _env_float("LLM_TIMEOUT", 20.0, minimum=0.5, maximum=300.0),
            "max_retries": _env_int("LLM_MAX_RETRIES", 2, minimum=0, maximum=5),
            "temperature": _env_float("LLM_TEMPERATURE", 0.2, minimum=0.0, maximum=2.0),
            "max_tokens": _env_int("LLM_MAX_TOKENS", 800, minimum=64, maximum=8192),
            "max_context_chars": _env_int(
                "RAG_MAX_CONTEXT_CHARS", 6000, minimum=500, maximum=50000
            ),
            "rerank_provider": _env_str("RERANK_PROVIDER", "heuristic").lower(),
            "rerank_api_base": _env_str("RERANK_API_BASE", "https://api.siliconflow.cn/v1"),
            "rerank_api_key": _env_str("RERANK_API_KEY", ""),
            "rerank_model": _env_str("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
            "rerank_fallback": _env_bool("RERANK_FALLBACK", True),
            "log_level": _env_str("LOG_LEVEL", "INFO").upper(),
        }
        known = {field.name for field in fields(cls)}
        unknown = set(overrides) - known
        if unknown:
            raise ConfigError(f"未知配置项：{', '.join(sorted(unknown))}")
        data.update(overrides)
        return cls(**data)  # type: ignore[arg-type]

    def describe(self) -> dict[str, str | int | float | bool]:
        """返回可安全打印的配置摘要，绝不包含 API Key。"""

        return {
            "db_path": str(self.db_path),
            "top_k": self.top_k,
            "candidate_k": self.candidate_k,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embed_provider": self.embed_provider,
            "llm_provider": self.llm_provider,
            "rerank_provider": self.rerank_provider,
            "llm_model": self.llm_model if self.llm_provider != "mock" else "mock",
        }
