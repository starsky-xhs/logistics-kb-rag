"""生成器测试：抽取式回答、强制引用、拒答与流式事件。"""

from __future__ import annotations

import unittest

from app.generator import REFUSAL_TEXT, MockGenerator
from app.models import ChatMessage, Chunk, SearchHit


def _hit(
    text: str,
    *,
    source: str = "01-售后与退货政策.md",
    heading: tuple[str, ...] = ("售后与退货政策", "退货时效", "退款到账时效"),
    title: str = "售后与退货政策",
    rerank: float = 0.7,
) -> SearchHit:
    chunk = Chunk(
        chunk_id=f"chunk-{source}-{hash(text) & 0xFFFF}",
        doc_id="doc-1",
        source=source,
        title=title,
        heading_path=heading,
        ordinal=1,
        text=text,
    )
    return SearchHit(chunk=chunk, score=0.8, rerank_score=rerank, rrf_score=0.03)


class TestMockGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = MockGenerator()

    def test_extractive_answer_has_citation(self) -> None:
        hit = _hit(
            "仓库验收通过后，退款在 1 个工作日内提交支付渠道。"
            "原路退回一般 1 至 3 个工作日到账；对公转账为 3 至 5 个工作日。"
        )
        result = self.generator.generate("退款多久能到账？", [hit])
        self.assertFalse(result.refused)
        self.assertIn("[1]", result.text)
        self.assertIn("1 至 3 个工作日", result.text)
        self.assertEqual(len(result.citations), 1)
        self.assertEqual(result.citations[0].source, hit.chunk.source)
        self.assertIn("退货时效", result.citations[0].label)
        self.assertIn("参考来源", result.text)
        self.assertGreater(result.token_estimate, 0)

    def test_no_hits_is_refused(self) -> None:
        result = self.generator.generate("退款多久到账？", [])
        self.assertTrue(result.refused)
        self.assertEqual(result.citations, [])
        self.assertEqual(result.text, REFUSAL_TEXT)

    def test_irrelevant_hit_is_refused(self) -> None:
        hit = _hit(
            "国际海运整柜需要提前订舱，具体船期以船公司通知为准。",
            heading=("售后与退货政策", "国际运输"),
        )
        result = self.generator.generate("生鲜冷链的温控区间是多少度？", [hit])
        self.assertTrue(result.refused)
        self.assertEqual(result.citations, [])

    def test_material_question_expands_list_items(self) -> None:
        hit = _hit(
            "1. 运单号、订单号及发货清单；\n"
            "2. 破损货物全景照片、包装六面照片；\n"
            "3. 货物价值证明，包括采购发票或销售合同；\n"
            "4. 签收单或签收记录；\n"
            "5. 加盖公章的索赔函。",
            source="03-货物破损理赔流程.md",
            heading=("货物破损理赔流程", "所需材料"),
            title="货物破损理赔流程",
        )
        result = self.generator.generate("申请理赔需要哪些材料？", [hit])
        self.assertFalse(result.refused)
        self.assertIn("运单号", result.text)
        self.assertIn("签收单", result.text)
        self.assertIn("索赔函", result.text)
        self.assertGreaterEqual(result.text.count("[1]"), 4)

    def test_multiple_hits_are_numbered(self) -> None:
        hits = [
            _hit("退款在 1 至 3 个工作日到账。"),
            _hit(
                "退款未到账时应先核对支付渠道流水，再提供退款单号。",
                heading=("售后与退货政策", "异常处理"),
            ),
        ]
        result = self.generator.generate("退款到账要多久，没到账怎么办？", hits)
        self.assertFalse(result.refused)
        self.assertIn("[1]", result.text)
        self.assertIn("[2]", result.text)
        self.assertEqual([item.index for item in result.citations], [1, 2])

    def test_stream_emits_citation_token_done(self) -> None:
        hit = _hit("退款一般 1 至 3 个工作日到账。")
        events = list(self.generator.stream("退款多久到账？", [hit]))
        kinds = [event.kind for event in events]
        self.assertIn("citation", kinds)
        self.assertIn("token", kinds)
        self.assertEqual(kinds[-1], "done")
        done = events[-1].data
        self.assertFalse(done["refused"])
        self.assertTrue(done["citations"])
        self.assertIn("1 至 3 个工作日", done["text"])

    def test_history_is_accepted(self) -> None:
        hit = _hit("退款一般 1 至 3 个工作日到账。")
        history = [ChatMessage(role="user", content="我问的是退款")]
        result = self.generator.generate("多久到账？", [hit], history)
        self.assertFalse(result.refused)


if __name__ == "__main__":
    unittest.main()
