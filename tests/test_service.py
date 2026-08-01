"""RAGService 测试：真实入库、幂等、更新、问答与并发检索。"""

from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor

from app.errors import LoaderError, ValidationError
from app.models import ChatMessage
from app.service import RAGService
from tests.support import (
    DAMAGE_DOC,
    FREIGHT_DOC,
    REFUND_DOC,
    make_settings,
    temporary_workspace,
    write_doc,
)


class TestRAGService(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_path = self.enterContext(temporary_workspace("service"))
        self.docs_dir = self.tmp_path / "docs"
        self.docs_dir.mkdir()
        write_doc(self.docs_dir, "售后.md", REFUND_DOC)
        write_doc(self.docs_dir, "运费.md", FREIGHT_DOC)
        write_doc(self.docs_dir, "理赔.md", DAMAGE_DOC)
        self.settings = make_settings(self.tmp_path)
        self.service = RAGService(self.settings)

    def tearDown(self) -> None:
        self.service.close()

    def test_ingest_and_duplicate_skip(self) -> None:
        first = self.service.ingest_dir(self.docs_dir)
        self.assertEqual(first.scanned, 3)
        self.assertEqual(first.added, 3)
        self.assertGreaterEqual(first.chunks, 3)
        self.assertTrue(first.ok)
        second = self.service.ingest_dir(self.docs_dir)
        self.assertEqual(second.added, 0)
        self.assertEqual(second.skipped, 3)
        self.assertEqual(second.chunks, 0)

    def test_changed_file_is_updated_and_reindexed(self) -> None:
        self.service.ingest_dir(self.docs_dir)
        old_hash = self.service.store.get_document_hash("售后.md")
        write_doc(self.docs_dir, "售后.md", REFUND_DOC + "\n补充：退款到账后短信通知。\n")
        report = self.service.ingest_dir(self.docs_dir)
        self.assertEqual(report.updated, 1)
        self.assertNotEqual(self.service.store.get_document_hash("售后.md"), old_hash)
        self.assertGreater(report.chunks, 0)

    def test_answer_has_citation(self) -> None:
        self.service.ingest_dir(self.docs_dir)
        answer = self.service.answer("退款多久能到账？")
        self.assertFalse(answer.refused)
        self.assertTrue(answer.citations)
        self.assertIn("[1]", answer.text)
        self.assertIn("1 至 3 个工作日", answer.text)
        self.assertGreater(answer.latency_ms, 0)
        self.assertTrue(answer.hits)
        self.assertTrue(answer.citation_sources)

    def test_unanswerable_question_is_refused(self) -> None:
        self.service.ingest_dir(self.docs_dir)
        answer = self.service.answer("国际海运整柜到鹿特丹的清关流程是什么？")
        self.assertTrue(answer.refused)
        self.assertEqual(answer.citations, [])

    def test_search_records_query_log(self) -> None:
        self.service.ingest_dir(self.docs_dir)
        hits = self.service.search("破损理赔材料")
        self.assertTrue(hits)
        logs = self.service.recent_queries()
        self.assertTrue(logs)
        self.assertEqual(logs[-1]["kind"], "search")
        self.assertGreaterEqual(int(logs[-1]["hit_count"]), 1)
        stats = self.service.stats()
        self.assertEqual(stats["documents"], 3)
        self.assertIn("config", stats)

    def test_history_and_empty_question_validation(self) -> None:
        self.service.ingest_dir(self.docs_dir)
        with self.assertRaises(ValidationError):
            self.service.answer("   ")
        with self.assertRaises(ValidationError):
            self.service.answer("退款多久到账？", [ChatMessage(role="user", content=" ")])
        too_many = [ChatMessage(role="user", content="问题") for _ in range(21)]
        with self.assertRaises(ValidationError):
            self.service.answer("退款多久到账？", too_many)

    def test_missing_directory_raises(self) -> None:
        with self.assertRaises(LoaderError):
            self.service.ingest_dir(self.tmp_path / "not-exists")

    def test_concurrent_searches_are_safe(self) -> None:
        self.service.ingest_dir(self.docs_dir)
        queries = ["退款到账", "体积重怎么算", "理赔材料", "七天无理由"]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(self.service.search, queries))
        self.assertEqual(len(results), 4)
        self.assertTrue(all(isinstance(hits, list) for hits in results))
        self.assertTrue(all(hits for hits in results))


if __name__ == "__main__":
    unittest.main()
