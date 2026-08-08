"""HTTP 请求/响应模型。

这些模型与领域 dataclass 解耦：领域层可以独立测试，不必依赖 pydantic；
HTTP 字段一旦变化，也不会反向污染检索与生成代码。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class IngestRequest(BaseModel):
    path: str = Field(min_length=1, description="服务器可访问的目录路径")


class IngestWarnings(BaseModel):
    file: str
    reason: str


class IngestResponse(BaseModel):
    root: str
    scanned: int
    added: int
    updated: int
    skipped: int
    failed: int
    chunks: int
    elapsed_ms: float
    errors: list[IngestWarnings] = Field(default_factory=list)
    warnings: list[IngestWarnings] = Field(default_factory=list)


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    source: str | None = None


class CitationOut(BaseModel):
    index: int
    source: str
    heading_path: list[str]
    chunk_id: str
    snippet: str


class HitOut(BaseModel):
    source: str
    heading_path: list[str]
    chunk_id: str
    text: str
    score: float
    explain: dict[str, object]


class SearchResponse(BaseModel):
    question: str
    hits: list[HitOut]


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[HistoryMessage] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    version: str
    documents: int
    chunks: int
    embedder: str
    generator: str


class StatsResponse(BaseModel):
    documents: int
    chunks: int
    embedding_dim: int
    revision: int
    db_path: str
    sources: list[dict[str, object]]
    config: dict[str, object]
    embedder: str
    generator: str
    recent_queries: list[dict[str, object]]
