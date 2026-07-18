"""SQLite 存储测试：向量 BLOB、过滤、增量覆盖与 WAL。"""

from __future__ import annotations

import sqlite3
import unittest

from app.errors import IngestError, RetrievalError
from app.models import Chunk, Document
from app.store import SQLiteStore, blob_to_vector, vector_to_blob
from tests.support import temporary_workspace


def _document(source: str, content_hash: str, title: str = "测试文档") -> Document:
    return Document(
        doc_id="doc-" + source,
        source=source,
        title=title,
        content="正文",
        content_hash=content_hash,
        mtime=1.0,
        size=2,
        metadata={"format": "md"},
    )


def _chunk(
    source: str,
    chunk_id: str,
    title: str,
    heading: tuple[str, ...],
    ordinal: int,
    text: str,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-" + source,
        source=source,
        title=title,
        heading_path=heading,
        ordinal=ordinal,
        text=text,
    )


class TestSQLiteStore(unittest.TestCase):
    def setUp(self) -> None:
        workspace = self.enterContext(temporary_workspace("store"))
        self.db_path = workspace / "kb.sqlite3"
        self.store = SQLiteStore(self.db_path)
        self.store.init_schema()
        self.document = _document("a.md", "hash-a")
        self.chunks = [
            _chunk("a.md", "c1", "测试文档", ("测试文档", "退款"), 1, "退款一至三个工作日到账"),
            _chunk("a.md", "c2", "测试文档", ("测试文档", "运费"), 2, "运费按体积重计算"),
        ]
        self.store.save_document(self.document, self.chunks, [[1.0, 0.0], [0.0, 1.0]])

    def test_stats_and_listing(self) -> None:
        stats = self.store.stats()
        self.assertEqual(stats["documents"], 1)
        self.assertEqual(stats["chunks"], 2)
        self.assertEqual(stats["embedding_dim"], 2)
        self.assertEqual(stats["sources"][0]["source"], "a.md")

        chunks = self.store.list_chunks()
        self.assertEqual([chunk.chunk_id for chunk in chunks], ["c1", "c2"])

    def test_vector_search_orders_by_cosine(self) -> None:
        results = self.store.vector_search([1.0, 0.0], top_k=2)
        self.assertEqual(results[0].chunk.chunk_id, "c1")
        self.assertAlmostEqual(results[0].score, 1.0, places=5)
        self.assertEqual(results[1].chunk.chunk_id, "c2")

    def test_source_and_heading_filters(self) -> None:
        self.assertEqual(self.store.vector_search([1.0, 0.0], 5, source="missing.md"), [])
        only_heading = self.store.list_chunks(heading_prefix=("测试文档", "运费"))
        self.assertEqual(len(only_heading), 1)
        self.assertEqual(only_heading[0].chunk_id, "c2")
        no_match = self.store.list_chunks(heading_prefix=("其他",))
        self.assertEqual(no_match, [])

    def test_same_hash_is_idempotent(self) -> None:
        revision = self.store.corpus_revision()
        status, written = self.store.save_document(
            self.document, self.chunks, [[1.0, 0.0], [0.0, 1.0]]
        )
        self.assertEqual((status, written), ("unchanged", 0))
        self.assertEqual(self.store.corpus_revision(), revision)

    def test_changed_hash_replaces_document(self) -> None:
        revision = self.store.corpus_revision()
        updated = _document("a.md", "hash-b")
        new_chunks = [_chunk("a.md", "c3", "测试文档", ("测试文档", "新章节"), 1, "新的内容")]
        status, written = self.store.save_document(updated, new_chunks, [[0.5, 0.5]])
        self.assertEqual((status, written), ("updated", 1))
        self.assertGreater(self.store.corpus_revision(), revision)
        stats = self.store.stats()
        self.assertEqual(stats["documents"], 1)
        self.assertEqual(stats["chunks"], 1)
        self.assertEqual(self.store.get_document_hash("a.md"), "hash-b")

    def test_delete_cascades_chunks(self) -> None:
        self.assertTrue(self.store.delete_document("a.md"))
        self.assertFalse(self.store.delete_document("a.md"))
        self.assertEqual(self.store.stats()["chunks"], 0)

    def test_dimension_mismatch_is_reported(self) -> None:
        with self.assertRaises(RetrievalError):
            self.store.vector_search([1.0, 0.0, 0.0], top_k=2)

    def test_invalid_save_is_rejected(self) -> None:
        with self.assertRaises(IngestError):
            self.store.save_document(self.document, self.chunks, [[1.0, 0.0]])
        with self.assertRaises(IngestError):
            self.store.save_document(self.document, [], [])

    def test_vector_blob_roundtrip(self) -> None:
        blob = vector_to_blob([0.25, -0.5, 1.0])
        restored = blob_to_vector(blob)
        self.assertEqual(len(restored), 3)
        self.assertAlmostEqual(restored[1], -0.5, places=6)

    def test_wal_mode_is_enabled(self) -> None:
        connection = sqlite3.connect(self.db_path)
        try:
            mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(mode.lower(), "wal")


if __name__ == "__main__":
    unittest.main()
