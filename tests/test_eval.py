"""评测测试：真实运行入库、检索、生成并检查指标与报告。"""

from __future__ import annotations

import json
import unittest

from app.errors import ValidationError
from app.eval import evaluate, load_gold_set, render_markdown
from app.service import RAGService
from tests.support import (
    FREIGHT_DOC,
    REFUND_DOC,
    make_settings,
    temporary_workspace,
    write_doc,
)


class TestEvaluation(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_path = self.enterContext(temporary_workspace("eval"))
        self.docs_dir = self.tmp_path / "docs"
        self.docs_dir.mkdir()
        write_doc(self.docs_dir, "售后.md", REFUND_DOC)
        write_doc(self.docs_dir, "运费.md", FREIGHT_DOC)
        self.gold_path = self.tmp_path / "gold.jsonl"
        samples = [
            {
                "id": "g1",
                "question": "退款多久能到账？",
                "expected_source": "售后.md",
                "key_points": ["1 至 3 个工作日"],
            },
            {
                "id": "g2",
                "question": "体积重按什么公式计算？",
                "expected_source": "运费.md",
                "key_points": ["除以 6000"],
            },
            {
                "id": "g3",
                "question": "国际海运整柜的清关流程是什么？",
                "expected_source": None,
                "key_points": [],
            },
        ]
        self.gold_path.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in samples) + "\n",
            encoding="utf-8",
        )
        self.service = RAGService(make_settings(self.tmp_path))
        self.service.ingest_dir(self.docs_dir)

    def tearDown(self) -> None:
        self.service.close()

    def test_metrics_are_computed_from_real_run(self) -> None:
        report = evaluate(self.service, self.gold_path, ks=(1, 3))
        self.assertEqual(report.counts["total"], 3)
        self.assertEqual(report.counts["answerable"], 2)
        self.assertEqual(report.counts["unanswerable"], 1)
        self.assertEqual(report.metrics["recall@1"], 1.0)
        self.assertEqual(report.metrics["recall@3"], 1.0)
        self.assertEqual(report.metrics["citation_hit_rate"], 1.0)
        self.assertEqual(report.metrics["refusal_accuracy"], 1.0)
        self.assertGreaterEqual(report.metrics["mrr"], 0.99)
        self.assertEqual(len(report.cases), 3)
        self.assertTrue(all(case.latency_ms > 0 for case in report.cases))

    def test_markdown_report_contains_metrics_and_cases(self) -> None:
        report = evaluate(self.service, self.gold_path, ks=(1, 3))
        markdown = render_markdown(report)
        self.assertIn("# 物流知识库 RAG 评测报告", markdown)
        self.assertIn("Recall@1", markdown)
        self.assertIn("拒答判断准确率", markdown)
        self.assertIn("g3", markdown)

    def test_report_is_written_to_disk(self) -> None:
        output = self.tmp_path / "nested" / "EVAL_REPORT.md"
        evaluate(self.service, self.gold_path, ks=(1,), report_path=output)
        self.assertTrue(output.exists())
        self.assertIn("Recall@1", output.read_text(encoding="utf-8"))

    def test_duplicate_ids_are_rejected(self) -> None:
        bad = self.tmp_path / "bad.jsonl"
        bad.write_text(
            json.dumps({"id": "x", "question": "q", "expected_source": None})
            + "\n"
            + json.dumps({"id": "x", "question": "q2", "expected_source": None}),
            encoding="utf-8",
        )
        with self.assertRaises(ValidationError):
            load_gold_set(bad)

    def test_missing_gold_file_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            load_gold_set(self.tmp_path / "missing.jsonl")


if __name__ == "__main__":
    unittest.main()
