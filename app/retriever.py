"""混合检索与可解释重排。

流程：
1. 向量通道取 ``candidate_k`` 个候选；
2. BM25 通道取同样数量候选；
3. 用 Reciprocal Rank Fusion（RRF，k=60）融合两个排名；
4. 可选的启发式重排：章节标题命中、查询词覆盖率、分块长度惩罚；
5. 输出带 ``explain`` 的 ``SearchHit``，方便客服排障和 demo 展示。

RRF 只看排名、不直接相加两个通道的原始分数，因此不需要假设余弦相似度和
BM25 分数处于同一量纲；重排则是一个独立的、可关闭的业务相关性校正。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any, Final

from .bm25 import BM25Index, tokenize
from .config import Settings
from .embed import Embedder
from .errors import RetrievalError
from .models import Chunk, SearchHit
from .store import SQLiteStore

logger = logging.getLogger(__name__)

_QUESTION_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "什么",
        "怎么",
        "如何",
        "多少",
        "多久",
        "哪些",
        "哪个",
        "是否",
        "可以",
        "需要",
        "请问",
        "一般",
        "分别",
        "以及",
        "相关",
        "进行",
        "这个",
        "那个",
        "我们",
        "你们",
        "它们",
        "的话",
        "怎样",
        "怎么办",
    }
)
_RRF_MIN_SCORE: Final = 1e-9


def content_terms(text: str) -> set[str]:
    """去掉疑问词后的内容词，用于覆盖率与拒答判断。"""

    return {
        token
        for token in tokenize(text)
        if token not in _QUESTION_STOPWORDS and not token.isspace()
    }


class HybridRetriever:
    """向量 + BM25 的混合检索器。"""

    def __init__(
        self,
        store: SQLiteStore,
        embedder: Embedder,
        settings: Settings,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        source: str | None = None,
        heading_prefix: Sequence[str] | None = None,
        rerank: bool | None = None,
    ) -> list[SearchHit]:
        """执行混合检索；``rerank=None`` 时使用配置默认值。"""

        query = query.strip()
        limit = top_k or self.settings.top_k
        if not query or limit <= 0:
            return []

        chunks = self.store.list_chunks(source=source, heading_prefix=heading_prefix)
        if not chunks:
            return []
        candidate_k = min(len(chunks), max(limit, self.settings.candidate_k))

        # 向量通道
        query_vector = self.embedder.embed_query(query)
        vector_matches = self.store.vector_search(
            query_vector,
            candidate_k,
            source=source,
            heading_prefix=heading_prefix,
        )
        vector_rank = {
            match.chunk.chunk_id: (rank, match.score)
            for rank, match in enumerate(vector_matches, start=1)
        }

        # BM25 通道。标题重复一次，让章节词在打分中略占优势。
        index_texts = [
            f"{chunk.heading_text}\n{chunk.heading_text}\n{chunk.title}\n{chunk.text}"
            for chunk in chunks
        ]
        bm25 = BM25Index(index_texts, k1=self.settings.bm25_k1, b=self.settings.bm25_b)
        bm25_results = bm25.search(query, candidate_k)
        bm25_rank = {
            chunks[index].chunk_id: (rank, score)
            for rank, (index, score) in enumerate(bm25_results, start=1)
        }

        # RRF 融合
        merged: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            vector_info = vector_rank.get(chunk.chunk_id)
            bm25_info = bm25_rank.get(chunk.chunk_id)
            if vector_info is None and bm25_info is None:
                continue
            rrf = 0.0
            if vector_info:
                rrf += 1.0 / (self.settings.rrf_k + vector_info[0])
            if bm25_info:
                rrf += 1.0 / (self.settings.rrf_k + bm25_info[0])
            merged[chunk.chunk_id] = {
                "chunk": chunk,
                "rrf": rrf,
                "vector_rank": vector_info[0] if vector_info else 0,
                "vector_score": float(vector_info[1]) if vector_info else 0.0,
                "bm25_rank": bm25_info[0] if bm25_info else 0,
                "bm25_score": float(bm25_info[1]) if bm25_info else 0.0,
            }

        if not merged:
            return []
        max_rrf = max(float(item["rrf"]) for item in merged.values()) or 1.0
        query_terms = content_terms(query)

        should_rerank = self.settings.rerank_provider != "off" if rerank is None else rerank
        rerank_scores: dict[str, float] = {}
        rerank_mode = "off"
        if should_rerank:
            if self.settings.rerank_provider == "http":
                try:
                    rerank_scores = self._http_rerank(query, merged)
                    rerank_mode = "http"
                except RetrievalError as exc:
                    if not self.settings.rerank_fallback:
                        raise
                    logger.warning(
                        "HTTP 重排失败，回退到启发式重排",
                        extra={"error": str(exc)},
                    )
                    rerank_scores = self._heuristic_scores(query_terms, merged)
                    rerank_mode = "heuristic-fallback"
            else:
                rerank_scores = self._heuristic_scores(query_terms, merged)
                rerank_mode = "heuristic"

        hits: list[SearchHit] = []
        for chunk_id, item in merged.items():
            chunk = item["chunk"]
            assert not isinstance(chunk, (str, int, float))  # 类型收窄，便于 mypy
            rrf = float(item["rrf"])
            rrf_norm = rrf / max_rrf
            rerank_score = rerank_scores.get(chunk_id, 0.0)
            if rerank_mode == "off":
                final_score = rrf
            else:
                final_score = 0.45 * rrf_norm + 0.55 * rerank_score
            section_heading = self._section_heading(chunk)
            explain = {
                "vector_rank": int(item["vector_rank"]),
                "vector_score": round(float(item["vector_score"]), 6),
                "bm25_rank": int(item["bm25_rank"]),
                "bm25_score": round(float(item["bm25_score"]), 6),
                "rrf_score": round(rrf, 6),
                "rrf_norm": round(rrf_norm, 6),
                "rerank_mode": rerank_mode,
                "rerank_score": round(rerank_score, 6),
                "content_term_coverage": round(
                    self._coverage(query_terms, f"{section_heading} {chunk.text}"), 4
                ),
                "heading_hit": self._heading_hit(query_terms, section_heading),
                "title_hit": self._title_hit(query_terms, chunk.title),
                "length_fit": self._length_fit(chunk.char_count),
                "chunk_chars": chunk.char_count,
                "citation": chunk.display_title,
            }
            hits.append(
                SearchHit(
                    chunk=chunk,
                    score=round(final_score, 8),
                    vector_score=float(item["vector_score"]),
                    bm25_score=float(item["bm25_score"]),
                    rrf_score=rrf,
                    rerank_score=rerank_score,
                    vector_rank=int(item["vector_rank"]) or None,
                    bm25_rank=int(item["bm25_rank"]) or None,
                    explain=explain,
                )
            )

        hits.sort(key=lambda hit: (-hit.score, hit.chunk.source, hit.chunk.ordinal))
        return hits[:limit]

    @staticmethod
    def _coverage(query_terms: set[str], text: str) -> float:
        if not query_terms:
            return 0.0
        text_terms = set(tokenize(text))
        return len(query_terms & text_terms) / len(query_terms)

    @staticmethod
    def _section_heading(chunk: Chunk) -> str:
        """去掉文档 H1，只保留章节路径，避免所有分块都蹭到标题词。"""

        path = tuple(chunk.heading_path)
        # 单层路径就是文档 H1，所有分块都带它，不能当作章节命中信号。
        if len(path) > 1:
            return " > ".join(path[1:])
        return ""

    @staticmethod
    def _heading_hit(query_terms: set[str], heading: str) -> float:
        """章节标题命中率；按标题长度归一，避免长标题白送分。"""

        if not query_terms or not heading:
            return 0.0
        heading_terms = set(tokenize(heading))
        if not heading_terms:
            return 0.0
        return len(query_terms & heading_terms) / len(heading_terms)

    @staticmethod
    def _title_hit(query_terms: set[str], title: str) -> float:
        if not query_terms:
            return 0.0
        title_terms = set(tokenize(title))
        return len(query_terms & title_terms) / max(1, min(4, len(query_terms)))

    @staticmethod
    def _length_fit(char_count: int) -> float:
        """知识库分块以 200~500 字最适合作为回答依据。"""

        return max(0.0, 1.0 - abs(char_count - 350) / 700)

    def _heuristic_scores(
        self,
        query_terms: set[str],
        merged: dict[str, dict[str, Any]],
    ) -> dict[str, float]:
        """章节标题 25%、词覆盖 50%、文档标题 15%、长度 10%。"""

        scores: dict[str, float] = {}
        for chunk_id, item in merged.items():
            chunk = item["chunk"]
            section_heading = self._section_heading(chunk)
            coverage = self._coverage(query_terms, f"{section_heading} {chunk.text}")
            heading = self._heading_hit(query_terms, section_heading)
            title = self._title_hit(query_terms, chunk.title)
            length_fit = self._length_fit(chunk.char_count)
            scores[chunk_id] = 0.50 * coverage + 0.25 * heading + 0.15 * title + 0.10 * length_fit
        return scores

    def _http_rerank(
        self,
        query: str,
        merged: dict[str, dict[str, Any]],
    ) -> dict[str, float]:
        try:
            import httpx  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RetrievalError("HTTP 重排需要可选依赖 httpx") from exc

        chunk_ids = list(merged)
        documents = []
        for chunk_id in chunk_ids:
            chunk = merged[chunk_id]["chunk"]
            assert not isinstance(chunk, (str, int, float))
            documents.append(f"{chunk.heading_text}\n{chunk.text}")

        base = self.settings.rerank_api_base.rstrip("/")
        endpoint = base if base.endswith("/rerank") else f"{base}/rerank"
        payload = {
            "model": self.settings.rerank_model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
        }
        headers = {
            "Authorization": f"Bearer {self.settings.rerank_api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        try:
            with httpx.Client(timeout=self.settings.request_timeout) as client:
                for attempt in range(self.settings.max_retries + 1):
                    try:
                        response = client.post(endpoint, json=payload, headers=headers)
                        if response.status_code in {408, 429, 500, 502, 503, 504}:
                            raise RetrievalError(f"重排服务暂不可用，HTTP {response.status_code}")
                        if response.status_code >= 400:
                            raise RetrievalError(
                                f"重排请求失败，HTTP {response.status_code}：{response.text[:160]}"
                            )
                        records = response.json().get("results")
                        if not isinstance(records, list):
                            raise RetrievalError("重排响应缺少 results 数组")
                        scores = {chunk_id: 0.0 for chunk_id in chunk_ids}
                        for record in records:
                            if not isinstance(record, dict):
                                continue
                            index = int(record.get("index", -1))
                            if not 0 <= index < len(chunk_ids):
                                continue
                            raw = record.get("relevance_score", record.get("score", 0.0))
                            scores[chunk_ids[index]] = float(raw)
                        return scores
                    except (RetrievalError, httpx.HTTPError, ValueError) as exc:
                        last_error = exc
                        if attempt >= self.settings.max_retries:
                            break
                        time.sleep(min(4.0, 0.5 * (2**attempt)))
        except httpx.HTTPError as exc:  # 建连阶段的错误
            last_error = exc
        raise RetrievalError(f"HTTP 重排失败：{last_error}")
