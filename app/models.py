"""领域数据模型。

这些 dataclass 是存储层、检索层、生成层与 API 层之间的共同语言。
刻意不依赖 pydantic，保证核心包在零第三方依赖下可运行。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

ExplainValue = str | int | float | bool | None


@dataclass(slots=True)
class Document:
    """一篇已解析的源文档。"""

    doc_id: str
    source: str
    title: str
    content: str
    content_hash: str
    mtime: float
    size: int
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.content)


@dataclass(slots=True)
class Chunk:
    """标题感知分块后的最小检索单元。"""

    chunk_id: str
    doc_id: str
    source: str
    title: str
    heading_path: tuple[str, ...]
    ordinal: int
    text: str

    @property
    def heading_text(self) -> str:
        return " > ".join(self.heading_path)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def display_title(self) -> str:
        """用于引用展示的「文件名 §章节」。"""

        if self.heading_path:
            return f"{self.source} §{self.heading_text}"
        return f"{self.source} §{self.title}"


@dataclass(slots=True)
class SearchHit:
    """一次混合检索的最终命中。"""

    chunk: Chunk
    score: float
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    vector_rank: int | None = None
    bm25_rank: int | None = None
    explain: dict[str, ExplainValue] = field(default_factory=dict)

    @property
    def citation_label(self) -> str:
        return self.chunk.display_title


@dataclass(slots=True)
class Citation:
    """回答中的编号引用。"""

    index: int
    source: str
    heading_path: tuple[str, ...]
    chunk_id: str
    snippet: str

    @property
    def label(self) -> str:
        if self.heading_path:
            return f"{self.source} §{' > '.join(self.heading_path)}"
        return self.source


@dataclass(slots=True)
class Answer:
    """RAG 服务对单个问题的完整回答。"""

    question: str
    text: str
    citations: list[Citation]
    refused: bool
    latency_ms: float
    token_estimate: int
    hits: list[SearchHit] = field(default_factory=list)

    @property
    def citation_sources(self) -> list[str]:
        seen: set[str] = set()
        sources: list[str] = []
        for citation in self.citations:
            if citation.source not in seen:
                seen.add(citation.source)
                sources.append(citation.source)
        return sources


@dataclass(slots=True)
class ChatMessage:
    """多轮对话中的一条历史消息。"""

    role: Literal["user", "assistant"]
    content: str

    def validate(self) -> None:
        if self.role not in {"user", "assistant"}:
            raise ValueError(f"不支持的消息角色：{self.role}")
        if not self.content.strip():
            raise ValueError("历史消息内容不能为空")


@dataclass(slots=True)
class IngestReport:
    """一次目录入库的可审计结果。"""

    root: str
    scanned: int = 0
    added: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    chunks: int = 0
    elapsed_ms: float = 0.0
    errors: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[tuple[str, str]] = field(default_factory=list)

    @property
    def changed(self) -> int:
        return self.added + self.updated

    @property
    def ok(self) -> bool:
        return self.failed == 0

    def summary(self) -> str:
        return (
            f"扫描 {self.scanned} 个文件，新增 {self.added}，更新 {self.updated}，"
            f"跳过 {self.skipped}，失败 {self.failed}，写入 {self.chunks} 个分块，"
            f"耗时 {self.elapsed_ms:.0f} ms"
        )


@dataclass(slots=True)
class QueryLog:
    """单次查询的观测记录，用于本地排障与成本估算。"""

    kind: Literal["search", "answer"]
    query: str
    latency_ms: float
    hit_count: int
    refused: bool
    token_estimate: int
    top_source: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "query": self.query,
            "latency_ms": round(self.latency_ms, 2),
            "hit_count": self.hit_count,
            "refused": self.refused,
            "token_estimate": self.token_estimate,
            "top_source": self.top_source,
        }


def estimate_tokens(text: str) -> int:
    """粗略估算 token：中文按 1 字 1 token，英文按 4 字符 1 token。"""

    if not text:
        return 0
    chinese = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    other = len(text) - chinese
    return chinese + max(1, math.ceil(other / 4))
