"""BM25 测试：中文 bigram、英文词、排序与边界条件。"""

from __future__ import annotations

import unittest

from app.bm25 import BM25Index, tokenize


class TestTokenizer(unittest.TestCase):
    def test_chinese_bigrams_and_english_words(self) -> None:
        tokens = tokenize("退货 refund 2026")
        self.assertIn("退货", tokens)
        self.assertNotIn("退款", tokens)
        self.assertIn("refund", tokens)
        self.assertIn("2026", tokens)

    def test_single_chinese_char(self) -> None:
        self.assertEqual(tokenize("退"), ["退"])

    def test_punctuation_is_ignored(self) -> None:
        self.assertEqual(tokenize("，。！？"), [])


class TestBM25Index(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = [
            "退款到账时效：原路退回一般 1 至 3 个工作日到账。",
            "体积重按长宽高除以 6000 计算，实重与体积重取较大值。",
            "破损理赔需要运单号、六面照片、价值证明和索赔函。",
            "完全无关的一段天气描述，今天晴朗。",
        ]
        self.index = BM25Index(self.documents, k1=1.5, b=0.75)

    def test_ranking_returns_relevant_document(self) -> None:
        results = self.index.search("退款多久到账", top_k=2)
        self.assertTrue(results)
        self.assertEqual(results[0][0], 0)
        self.assertGreater(results[0][1], 0)

    def test_material_query_hits_material_document(self) -> None:
        results = self.index.search("破损理赔要哪些材料", top_k=3)
        self.assertEqual(results[0][0], 2)

    def test_english_query(self) -> None:
        index = BM25Index(["refund policy and return window", "shipping weight rules"])
        results = index.search("refund window", top_k=1)
        self.assertEqual(results[0][0], 0)

    def test_top_k_is_respected(self) -> None:
        results = self.index.search("货物 运费 退款", top_k=1)
        self.assertLessEqual(len(results), 1)

    def test_empty_query_and_empty_corpus(self) -> None:
        self.assertEqual(self.index.search("", top_k=5), [])
        self.assertEqual(self.index.search("   ", top_k=5), [])
        empty = BM25Index([])
        self.assertEqual(empty.search("退款", top_k=5), [])
        self.assertEqual(empty.idf("退款"), 0.0)

    def test_idf_is_positive(self) -> None:
        self.assertGreater(self.index.idf("退款"), 0.0)
        self.assertGreater(self.index.idf("从未出现的词"), 0.0)

    def test_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            BM25Index(["a"], k1=0)
        with self.assertRaises(ValueError):
            BM25Index(["a"], b=1.2)


if __name__ == "__main__":
    unittest.main()
