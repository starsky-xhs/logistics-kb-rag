"""混合检索测试：RRF、explain、过滤与重排开关。"""

from __future__ import annotations

import hashlib
import unittest

from app.embed import HashEmbedder
from app.models import Chunk, Document
from app.retriever import HybridRetriever, content_terms
from app.store import SQLiteStore
from tests.support import make_settings, temporary_workspace


class TestHybridRetriever(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_path = self.enterContext(temporary_workspace("retriever"))
        self.settings = make_settings(self.tmp_path)
        self.store = SQLiteStore(self.settings.db_path)
        self.store.init_schema()
        self.embedder = HashEmbedder(dim=512)
        self.documents = {
            "refund.md": (
                "售后政策",
                ("售后政策", "退款到账时效"),
                "退货验收通过后，退款在 1 个工作日内提交，原路退回一般 1 至 3 个工作日到账。",
            ),
            "freight.md": (
                "运费说明",
                ("运费说明", "体积重"),
                "标准快递体积重按长宽高除以 6000 计算，实重与体积重取较大值。",
            ),
            "damage.md": (
                "破损理赔",
                ("破损理赔", "所需材料"),
                "申请破损理赔需要运单号、包装六面照片、价值证明和索赔函。",
            ),
        }
        for ordinal, (source, (title, heading, text)) in enumerate(self.documents.items(), start=1):
            document = Document(
                doc_id=hashlib.sha256(source.encode()).hexdigest()[:16],
                source=source,
                title=title,
                content=text,
                content_hash=f"hash-{ordinal}",
                mtime=1.0,
                size=len(text),
            )
            chunk = Chunk(
                chunk_id=f"chunk-{ordinal}",
                doc_id=document.doc_id,
                source=source,
                title=title,
                heading_path=heading,
                ordinal=1,
                text=text,
            )
            embed_text = f"{title}\n{' > '.join(heading)}\n{text}"
            self.store.save_document(document, [chunk], [self.embedder.embed_query(embed_text)])
        self.retriever = HybridRetriever(self.store, self.embedder, self.settings)

    def test_refund_query_hits_refund_chunk_with_explain(self) -> None:
        hits = self.retriever.search("退款多久能到账", top_k=3)
        self.assertTrue(hits)
        self.assertEqual(hits[0].chunk.source, "refund.md")
        explain = hits[0].explain
        for key in (
            "vector_rank",
            "vector_score",
            "bm25_rank",
            "bm25_score",
            "rrf_score",
            "rerank_score",
            "content_term_coverage",
        ):
            self.assertIn(key, explain)
        self.assertGreater(hits[0].score, 0)
        self.assertEqual(hits[0].explain["rerank_mode"], "heuristic")

    def test_freight_and_damage_queries(self) -> None:
        self.assertEqual(
            self.retriever.search("体积重怎么计算", top_k=1)[0].chunk.source, "freight.md"
        )
        self.assertEqual(
            self.retriever.search("破损理赔要哪些材料", top_k=1)[0].chunk.source,
            "damage.md",
        )

    def test_top_k_and_source_filter(self) -> None:
        hits = self.retriever.search("理赔材料", top_k=1, source="damage.md")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].chunk.source, "damage.md")
        self.assertEqual(self.retriever.search("理赔材料", source="missing.md"), [])

    def test_heading_prefix_filter(self) -> None:
        hits = self.retriever.search("退款到账", heading_prefix=("售后政策", "退款到账时效"))
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].chunk.source, "refund.md")

    def test_rerank_can_be_disabled(self) -> None:
        hits = self.retriever.search("退款到账", top_k=3, rerank=False)
        self.assertTrue(hits)
        self.assertEqual(hits[0].explain["rerank_mode"], "off")
        self.assertAlmostEqual(hits[0].score, hits[0].rrf_score, places=7)

    def test_empty_store_and_empty_query(self) -> None:
        empty_dir = self.tmp_path / "empty"
        empty_dir.mkdir()
        settings = make_settings(empty_dir, db_path=empty_dir / "empty.sqlite3")
        store = SQLiteStore(settings.db_path)
        store.init_schema()
        retriever = HybridRetriever(store, self.embedder, settings)
        self.assertEqual(retriever.search("退款", top_k=3), [])
        self.assertEqual(self.retriever.search("   ", top_k=3), [])

    def test_content_terms_drop_question_stopwords(self) -> None:
        terms = content_terms("退款需要多久")
        self.assertIn("退款", terms)
        self.assertNotIn("需要", terms)
        self.assertNotIn("多久", terms)


if __name__ == "__main__":
    unittest.main()
