"""SQLite 持久化。

设计取舍：

* 不引入 ORM。两张表、少量查询，手写 SQL 更透明，也更容易做参数化。
* 向量以 float32 BLOB 存储，余弦相似度在 Python 中计算。当前定位是单机
  知识库（万级分块），这样部署零额外组件；超过规模后应换 pgvector。
* 每个操作独立建连接并开启 WAL，避免 FastAPI 多线程共享同一个 sqlite3
  连接；写操作使用事务，失败自动回滚。
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from array import array
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .errors import IngestError, RetrievalError
from .models import Chunk, Document

SCHEMA_VERSION = "1"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    source       TEXT NOT NULL UNIQUE,
    title        TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    content      TEXT NOT NULL,
    mtime        REAL NOT NULL,
    size         INTEGER NOT NULL,
    metadata     TEXT NOT NULL DEFAULT '{}',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      TEXT PRIMARY KEY,
    doc_id        TEXT NOT NULL,
    source        TEXT NOT NULL,
    title         TEXT NOT NULL,
    heading_path  TEXT NOT NULL,
    ordinal       INTEGER NOT NULL,
    text          TEXT NOT NULL,
    embedding     BLOB NOT NULL,
    embedding_dim INTEGER NOT NULL,
    created_at    TEXT NOT NULL,
    FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks (source);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_ordinal ON chunks (doc_id, ordinal);

CREATE TABLE IF NOT EXISTS corpus_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def vector_to_blob(vector: Sequence[float]) -> bytes:
    """float32 小端 BLOB；SQLite 不做字节序转换，这里显式约定。"""

    data = array("f", (float(value) for value in vector))
    if sys.byteorder != "little":
        data.byteswap()
    return data.tobytes()


def blob_to_vector(blob: bytes) -> array:
    data = array("f")
    data.frombytes(blob)
    if sys.byteorder != "little":
        data.byteswap()
    return data


def _decode_heading(raw: str) -> tuple[str, ...]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return ()
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        source=row["source"],
        title=row["title"],
        heading_path=_decode_heading(row["heading_path"]),
        ordinal=int(row["ordinal"]),
        text=row["text"],
    )


def _now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


@dataclass(slots=True)
class VectorMatch:
    """向量通道的中间结果。"""

    chunk: Chunk
    score: float


class SQLiteStore:
    """文档与分块的 SQLite 存储。"""

    def __init__(self, db_path: str | Path, *, busy_timeout_ms: int = 5000) -> None:
        self.db_path = Path(db_path)
        self.busy_timeout_ms = busy_timeout_ms
        self._initialized = False

    def _open(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(
                str(self.db_path),
                timeout=self.busy_timeout_ms / 1000,
                isolation_level=None,
            )
        except sqlite3.Error as exc:
            raise IngestError(f"无法打开数据库 {self.db_path}：{exc}") from exc
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {int(self.busy_timeout_ms)}")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._open()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def _reader(self) -> Iterator[sqlite3.Connection]:
        """只读连接也显式关闭，避免长时间运行时依赖 GC 回收句柄。"""

        connection = self._open()
        try:
            yield connection
        finally:
            connection.close()

    def init_schema(self) -> None:
        """建表并启用 WAL；可重复调用。"""

        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            connection = self._open()
            try:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.execute("PRAGMA synchronous = NORMAL")
                connection.executescript(SCHEMA_SQL)
                connection.execute(
                    "INSERT INTO corpus_meta (key, value) VALUES ('revision', '0') "
                    "ON CONFLICT(key) DO NOTHING"
                )
                connection.execute(
                    "INSERT INTO corpus_meta (key, value) VALUES ('schema_version', ?) "
                    "ON CONFLICT(key) DO NOTHING",
                    (SCHEMA_VERSION,),
                )
            finally:
                connection.close()
        except sqlite3.Error as exc:
            raise IngestError(f"初始化数据库失败：{exc}") from exc
        self._initialized = True

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self.init_schema()

    def corpus_revision(self) -> int:
        """语料版本号，每次写操作递增。"""

        self._ensure_initialized()
        with self._reader() as connection:
            row = connection.execute(
                "SELECT value FROM corpus_meta WHERE key = 'revision'"
            ).fetchone()
        try:
            return int(row["value"]) if row else 0
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _bump_revision(connection: sqlite3.Connection) -> None:
        connection.execute(
            "INSERT INTO corpus_meta (key, value) VALUES ('revision', '1') "
            "ON CONFLICT(key) DO UPDATE SET value = CAST(corpus_meta.value AS INTEGER) + 1"
        )

    def get_document_hash(self, source: str) -> str | None:
        """返回已入库文件的 SHA-256；不存在时返回 None。"""

        self._ensure_initialized()
        with self._reader() as connection:
            row = connection.execute(
                "SELECT content_hash FROM documents WHERE source = ?", (source,)
            ).fetchone()
        return str(row["content_hash"]) if row else None

    def save_document(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> tuple[str, int]:
        """原子写入文档与全部分块。

        返回 ``("inserted"|"updated"|"unchanged", 分块数)``。同一 source 的
        content_hash 未变时直接跳过，这是增量入库的最后一道幂等保护。
        """

        self._ensure_initialized()
        if len(chunks) != len(vectors):
            raise IngestError("分块数量与向量数量不一致")
        if not chunks:
            raise IngestError("文档没有产生任何分块")
        for chunk, vector in zip(chunks, vectors, strict=True):
            if chunk.doc_id != document.doc_id:
                raise IngestError("分块 doc_id 与文档不一致")
            if chunk.source != document.source:
                raise IngestError("分块 source 与文档不一致")
            if not vector:
                raise IngestError(f"分块 {chunk.chunk_id} 的向量为空")
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise IngestError("同一文档的分块向量维度不一致")

        try:
            with self._transaction() as connection:
                existing = connection.execute(
                    "SELECT doc_id, content_hash FROM documents WHERE source = ?",
                    (document.source,),
                ).fetchone()
                if existing and existing["content_hash"] == document.content_hash:
                    return "unchanged", 0

                status = "updated" if existing else "inserted"
                if existing:
                    connection.execute("DELETE FROM documents WHERE source = ?", (document.source,))

                now = _now()
                connection.execute(
                    """
                    INSERT INTO documents
                        (doc_id, source, title, content_hash, content, mtime, size,
                         metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document.doc_id,
                        document.source,
                        document.title,
                        document.content_hash,
                        document.content,
                        document.mtime,
                        document.size,
                        json.dumps(document.metadata, ensure_ascii=False, sort_keys=True),
                        now,
                        now,
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO chunks
                        (chunk_id, doc_id, source, title, heading_path, ordinal,
                         text, embedding, embedding_dim, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            chunk.chunk_id,
                            chunk.doc_id,
                            chunk.source,
                            chunk.title,
                            json.dumps(chunk.heading_path, ensure_ascii=False),
                            chunk.ordinal,
                            chunk.text,
                            vector_to_blob(vector),
                            len(vector),
                            now,
                        )
                        for chunk, vector in zip(chunks, vectors, strict=True)
                    ],
                )
                self._bump_revision(connection)
                return status, len(chunks)
        except sqlite3.Error as exc:
            raise IngestError(f"写入文档失败（{document.source}）：{exc}") from exc

    def delete_document(self, source: str) -> bool:
        """按 source 删除文档及其分块。"""

        self._ensure_initialized()
        try:
            with self._transaction() as connection:
                cursor = connection.execute("DELETE FROM documents WHERE source = ?", (source,))
                if cursor.rowcount:
                    self._bump_revision(connection)
                    return True
                return False
        except sqlite3.Error as exc:
            raise IngestError(f"删除文档失败（{source}）：{exc}") from exc

    def list_chunks(
        self,
        *,
        source: str | None = None,
        heading_prefix: Sequence[str] | None = None,
    ) -> list[Chunk]:
        """列出分块，可按 source 精确过滤、按 heading_path 前缀过滤。"""

        self._ensure_initialized()
        sql = "SELECT chunk_id, doc_id, source, title, heading_path, ordinal, text FROM chunks"
        params: list[Any] = []
        if source:
            sql += " WHERE source = ?"
            params.append(source)
        sql += " ORDER BY source, ordinal"
        try:
            with self._reader() as connection:
                rows = connection.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise RetrievalError(f"读取分块失败：{exc}") from exc

        prefix = tuple(heading_prefix) if heading_prefix else ()
        chunks = [_row_to_chunk(row) for row in rows]
        if prefix:
            chunks = [chunk for chunk in chunks if chunk.heading_path[: len(prefix)] == prefix]
        return chunks

    def vector_search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        *,
        source: str | None = None,
        heading_prefix: Sequence[str] | None = None,
    ) -> list[VectorMatch]:
        """按余弦相似度返回 top-k；过滤条件与 ``list_chunks`` 一致。"""

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if not query_vector:
            raise RetrievalError("查询向量不能为空")
        self._ensure_initialized()

        sql = (
            "SELECT chunk_id, doc_id, source, title, heading_path, ordinal, text, "
            "embedding, embedding_dim FROM chunks"
        )
        params: list[Any] = []
        if source:
            sql += " WHERE source = ?"
            params.append(source)
        sql += " ORDER BY source, ordinal"

        try:
            with self._reader() as connection:
                rows = connection.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise RetrievalError(f"向量检索读取失败：{exc}") from exc

        query_norm = math.sqrt(sum(float(value) ** 2 for value in query_vector))
        if query_norm == 0:
            return []
        prefix = tuple(heading_prefix) if heading_prefix else ()
        scored: list[VectorMatch] = []
        expected_dim = len(query_vector)

        for row in rows:
            chunk = _row_to_chunk(row)
            if prefix and chunk.heading_path[: len(prefix)] != prefix:
                continue
            if int(row["embedding_dim"]) != expected_dim:
                raise RetrievalError(
                    "查询向量与库中向量维度不一致，可能是更换了 EMBED_PROVIDER 或 "
                    "EMBED_MODEL；请重新执行入库。"
                )
            stored = blob_to_vector(row["embedding"])
            stored_norm = math.sqrt(sum(float(value) ** 2 for value in stored))
            if stored_norm == 0:
                continue
            dot = sum(
                float(left) * float(right) for left, right in zip(query_vector, stored, strict=True)
            )
            scored.append(VectorMatch(chunk=chunk, score=dot / (query_norm * stored_norm)))

        scored.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return scored[:top_k]

    def stats(self) -> dict[str, Any]:
        """返回可展示的语料统计。"""

        self._ensure_initialized()
        with self._reader() as connection:
            document_count = int(
                connection.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
            )
            chunk_count = int(
                connection.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
            )
            source_rows = connection.execute(
                """
                SELECT c.source, COUNT(*) AS chunks, d.title
                FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
                GROUP BY c.source, d.title
                ORDER BY c.source
                """
            ).fetchall()
            dim_row = connection.execute("SELECT MAX(embedding_dim) AS dim FROM chunks").fetchone()
            revision_row = connection.execute(
                "SELECT value FROM corpus_meta WHERE key = 'revision'"
            ).fetchone()

        try:
            db_bytes = self.db_path.stat().st_size
        except OSError:
            db_bytes = 0
        try:
            revision = int(revision_row["value"]) if revision_row else 0
        except (TypeError, ValueError):
            revision = 0
        return {
            "documents": document_count,
            "chunks": chunk_count,
            "sources": [
                {
                    "source": row["source"],
                    "title": row["title"],
                    "chunks": int(row["chunks"]),
                }
                for row in source_rows
            ],
            "embedding_dim": int(dim_row["dim"]) if dim_row and dim_row["dim"] else 0,
            "revision": revision,
            "db_path": str(self.db_path),
            "db_bytes": db_bytes,
        }
