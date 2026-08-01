"""RAG 服务编排层。

职责边界：
* ``ingest_dir`` 负责文件扫描、哈希去重、解析、分块、向量化、落库；
* ``search`` / ``answer`` / ``stream_answer`` 负责检索与生成编排、指标记录；
* CLI、demo、FastAPI 只做参数解析与展示，不直接操作 SQLite 或模型客户端。

这样测试可以绕开 Web 层直接验证业务，也避免 API schema 与领域模型耦合。
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import deque
from collections.abc import Iterator, Sequence
from pathlib import Path

from .chunker import MarkdownChunker
from .config import Settings
from .embed import Embedder, build_embedder
from .errors import KbError, LoaderError, OptionalDependencyError, ValidationError
from .generator import GenerationResult, Generator, StreamEvent, build_generator
from .loaders import file_sha256, infer_title, iter_source_files, load_file, relative_source
from .logging_utils import configure_logging
from .models import (
    Answer,
    ChatMessage,
    Document,
    IngestReport,
    QueryLog,
    SearchHit,
    estimate_tokens,
)
from .retriever import HybridRetriever
from .store import SQLiteStore

logger = logging.getLogger(__name__)


class RAGService:
    """知识库问答服务门面。"""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        store: SQLiteStore | None = None,
        embedder: Embedder | None = None,
        generator: Generator | None = None,
        retriever: HybridRetriever | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        configure_logging(self.settings.log_level)
        self.store = store or SQLiteStore(self.settings.db_path)
        self.store.init_schema()
        self.embedder = embedder or build_embedder(self.settings)
        self.generator = generator or build_generator(self.settings)
        self.retriever = retriever or HybridRetriever(self.store, self.embedder, self.settings)
        self._query_logs: deque[QueryLog] = deque(maxlen=200)
        self._log_lock = threading.Lock()
        logger.info(
            "RAG 服务初始化完成",
            extra={
                "db_path": str(self.settings.db_path),
                "embedder": self.embedder.name,
                "generator": self.generator.name,
            },
        )

    # ------------------------------------------------------------------ 入库
    def ingest_dir(self, root: str | Path) -> IngestReport:
        """增量入库一个目录，单个文件失败不会中断整批任务。"""

        source_root = Path(root).expanduser().resolve()
        report = IngestReport(root=str(source_root))
        started = time.perf_counter()
        try:
            files = iter_source_files(source_root)
        except LoaderError:
            raise
        logger.info("开始扫描目录", extra={"root": str(source_root), "files": len(files)})

        for path in files:
            report.scanned += 1
            try:
                source = relative_source(path, source_root)
                digest = file_sha256(path)
                if self.store.get_document_hash(source) == digest:
                    report.skipped += 1
                    logger.debug("文件未变化，跳过", extra={"source": source})
                    continue

                text, metadata = load_file(path)
                title = infer_title(text, path.stem)
                stat = path.stat()
                document = Document(
                    doc_id=hashlib.sha256(source.encode("utf-8")).hexdigest()[:24],
                    source=source,
                    title=title,
                    content=text,
                    content_hash=digest,
                    mtime=stat.st_mtime,
                    size=stat.st_size,
                    metadata=metadata,
                )
                chunker = MarkdownChunker(
                    chunk_size=self.settings.chunk_size,
                    overlap=self.settings.chunk_overlap,
                )
                chunks = chunker.split(text, doc_id=document.doc_id, source=source, title=title)
                if not chunks:
                    raise LoaderError("文档没有产生有效分块，请检查正文是否为空")
                embed_texts = [
                    f"{chunk.title}\n{chunk.heading_text}\n{chunk.text}" for chunk in chunks
                ]
                vectors = self.embedder.embed_documents(embed_texts)
                status, written = self.store.save_document(document, chunks, vectors)
                if status == "inserted":
                    report.added += 1
                elif status == "updated":
                    report.updated += 1
                else:
                    report.skipped += 1
                report.chunks += written
                logger.info(
                    "文档入库完成",
                    extra={
                        "source": source,
                        "status": status,
                        "chunks": written,
                        "chars": len(text),
                    },
                )
            except OptionalDependencyError as exc:
                report.skipped += 1
                report.warnings.append((str(path.name), str(exc)))
                logger.warning(
                    "缺少可选依赖，跳过文件",
                    extra={"file": path.name, "reason": str(exc)},
                )
            except LoaderError as exc:
                report.failed += 1
                report.errors.append((str(path.name), str(exc)))
                logger.error("文档处理失败", extra={"file": path.name, "reason": str(exc)})
            except KbError as exc:
                report.failed += 1
                report.errors.append((str(path.name), str(exc)))
                logger.error("文档入库失败", extra={"file": path.name, "reason": str(exc)})
            except Exception as exc:  # 隔离单文件异常，避免整批回滚
                report.failed += 1
                report.errors.append((str(path.name), f"未预期错误：{exc}"))
                logger.exception("文档入库出现未预期错误", extra={"file": path.name})

        report.elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "目录入库结束",
            extra={
                "root": str(source_root),
                "scanned": report.scanned,
                "added": report.added,
                "updated": report.updated,
                "skipped": report.skipped,
                "failed": report.failed,
                "chunks": report.chunks,
                "elapsed_ms": round(report.elapsed_ms, 1),
            },
        )
        return report

    # ------------------------------------------------------------------ 检索
    def search(
        self,
        question: str,
        *,
        top_k: int | None = None,
        source: str | None = None,
        heading_prefix: Sequence[str] | None = None,
    ) -> list[SearchHit]:
        """检索相关分块，并记录延迟。"""

        if not question.strip():
            raise ValidationError("检索问题不能为空")
        started = time.perf_counter()
        hits = self.retriever.search(
            question,
            top_k=top_k,
            source=source,
            heading_prefix=heading_prefix,
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        self._record_query(
            QueryLog(
                kind="search",
                query=question,
                latency_ms=elapsed,
                hit_count=len(hits),
                refused=False,
                token_estimate=estimate_tokens(question),
                top_source=hits[0].chunk.source if hits else None,
            )
        )
        return hits

    # ------------------------------------------------------------------ 回答
    def answer(
        self,
        question: str,
        history: Sequence[ChatMessage] | None = None,
    ) -> Answer:
        """同步回答，用于 CLI 与普通 HTTP 调用。"""

        if not question.strip():
            raise ValidationError("问题不能为空")
        self._validate_history(history)
        started = time.perf_counter()
        hits = self.retriever.search(question)
        result: GenerationResult = self.generator.generate(question, hits, history)
        latency_ms = (time.perf_counter() - started) * 1000.0
        answer = Answer(
            question=question,
            text=result.text,
            citations=result.citations,
            refused=result.refused,
            latency_ms=latency_ms,
            token_estimate=result.token_estimate,
            hits=hits,
        )
        self._record_query(
            QueryLog(
                kind="answer",
                query=question,
                latency_ms=latency_ms,
                hit_count=len(hits),
                refused=answer.refused,
                token_estimate=answer.token_estimate,
                top_source=hits[0].chunk.source if hits else None,
            )
        )
        logger.info(
            "问题回答完成",
            extra={
                "question": question,
                "hits": len(hits),
                "refused": answer.refused,
                "citations": len(answer.citations),
                "latency_ms": round(latency_ms, 1),
                "token_estimate": answer.token_estimate,
            },
        )
        return answer

    def stream_answer(
        self,
        question: str,
        history: Sequence[ChatMessage] | None = None,
    ) -> Iterator[StreamEvent]:
        """流式回答；事件类型与 API SSE 一一对应。"""

        if not question.strip():
            raise ValidationError("问题不能为空")
        self._validate_history(history)
        started = time.perf_counter()
        try:
            hits = self.retriever.search(question)
        except KbError:
            logger.exception("流式回答前检索失败", extra={"question": question})
            raise
        token_estimate = 0
        refused = False
        try:
            for event in self.generator.stream(question, hits, history):
                if event.kind == "done":
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    token_estimate = int(event.data.get("token_estimate", 0))
                    refused = bool(event.data.get("refused", False))
                    event.data["latency_ms"] = round(latency_ms, 2)
                    event.data["hits"] = [_hit_explain(hit) for hit in hits[:5]]
                    self._record_query(
                        QueryLog(
                            kind="answer",
                            query=question,
                            latency_ms=latency_ms,
                            hit_count=len(hits),
                            refused=refused,
                            token_estimate=token_estimate,
                            top_source=hits[0].chunk.source if hits else None,
                        )
                    )
                yield event
        finally:
            logger.info(
                "流式回答结束",
                extra={
                    "question": question,
                    "hits": len(hits),
                    "refused": refused,
                    "token_estimate": token_estimate,
                    "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
                },
            )

    # ------------------------------------------------------------------ 其它
    def stats(self) -> dict[str, object]:
        """合并持久化统计、运行配置与最近查询。"""

        data = dict(self.store.stats())
        data.update(
            {
                "config": self.settings.describe(),
                "embedder": self.embedder.name,
                "generator": self.generator.name,
                "recent_queries": self.recent_queries(limit=10),
            }
        )
        return data

    def recent_queries(self, limit: int = 20) -> list[dict[str, object]]:
        if limit <= 0:
            return []
        with self._log_lock:
            logs = list(self._query_logs)[-limit:]
        return [log.as_dict() for log in logs]

    def close(self) -> None:
        close = getattr(self.embedder, "close", None)
        if callable(close):
            close()

    def _record_query(self, log: QueryLog) -> None:
        with self._log_lock:
            self._query_logs.append(log)

    @staticmethod
    def _validate_history(history: Sequence[ChatMessage] | None) -> None:
        if history is None:
            return
        if len(history) > 20:
            raise ValidationError("历史消息过多，最多保留 20 条")
        for message in history:
            try:
                message.validate()
            except ValueError as exc:
                raise ValidationError(str(exc)) from exc


def _hit_explain(hit: SearchHit) -> dict[str, object]:
    return {
        "chunk_id": hit.chunk.chunk_id,
        "source": hit.chunk.source,
        "heading_path": list(hit.chunk.heading_path),
        "score": round(hit.score, 6),
        "explain": dict(hit.explain),
    }
