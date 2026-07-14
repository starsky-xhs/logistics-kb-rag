"""分块器测试：标题路径、代码块、超长段落滑窗与确定性。"""

from __future__ import annotations

import unittest

from app.chunker import MarkdownChunker


class TestChunker(unittest.TestCase):
    def test_heading_path_is_preserved(self) -> None:
        text = (
            "# 售后政策\n\n"
            "总则内容。\n\n"
            "## 退货\n\n"
            "退货说明。\n\n"
            "### 时效\n\n"
            "客户应在 7 个自然日内寄回。\n\n"
            "## 运费\n\n"
            "运费说明。\n"
        )
        chunks = MarkdownChunker(chunk_size=500, overlap=50).split(
            text, doc_id="d1", source="policy.md", title="售后政策"
        )
        paths = {chunk.heading_path for chunk in chunks}
        self.assertIn(("售后政策", "退货", "时效"), paths)
        self.assertIn(("售后政策", "运费"), paths)
        self.assertNotIn(("售后政策", "退货", "时效", "运费"), paths)

    def test_headings_inside_code_fence_are_not_headings(self) -> None:
        text = "# 文档\n\n## 示例\n\n说明\n\n```bash\n# 这是注释，不是标题\n```\n"
        chunks = MarkdownChunker(chunk_size=500, overlap=50).split(
            text, doc_id="d1", source="a.md", title="文档"
        )
        self.assertTrue(chunks)
        self.assertEqual({chunk.heading_path for chunk in chunks}, {("文档", "示例")})
        self.assertIn("# 这是注释，不是标题", chunks[0].text)

    def test_long_paragraph_is_split_without_exceeding_window(self) -> None:
        paragraph = "".join(f"第{i}句说明退款时效与验收规则。" for i in range(1, 61))
        chunker = MarkdownChunker(chunk_size=200, overlap=40)
        chunks = chunker.split(
            f"# 文档\n\n## 长段落\n\n{paragraph}\n",
            doc_id="d1",
            source="a.md",
            title="文档",
        )
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.char_count <= 200 for chunk in chunks))
        self.assertTrue(all(chunk.heading_path == ("文档", "长段落") for chunk in chunks))
        self.assertEqual([chunk.ordinal for chunk in chunks], list(range(1, len(chunks) + 1)))

    def test_chunk_ids_are_deterministic(self) -> None:
        text = "# 文档\n\n## 章节\n\n一段足够长的中文内容，用于生成稳定分块。\n"
        first = MarkdownChunker(300, 50).split(text, doc_id="doc", source="a.md", title="文档")
        second = MarkdownChunker(300, 50).split(text, doc_id="doc", source="a.md", title="文档")
        self.assertEqual([chunk.chunk_id for chunk in first], [chunk.chunk_id for chunk in second])
        self.assertEqual(len({chunk.chunk_id for chunk in first}), len(first))

    def test_no_heading_uses_empty_path(self) -> None:
        chunks = MarkdownChunker(300, 50).split(
            "没有标题的一段正文。", doc_id="d", source="a.txt", title="a"
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].heading_path, ())

    def test_empty_text_returns_no_chunks(self) -> None:
        self.assertEqual(
            MarkdownChunker(300, 50).split("", doc_id="d", source="a.md", title="a"), []
        )

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MarkdownChunker(chunk_size=100, overlap=10)
        with self.assertRaises(ValueError):
            MarkdownChunker(chunk_size=200, overlap=200)
        with self.assertRaises(ValueError):
            MarkdownChunker(chunk_size=300, overlap=200)


if __name__ == "__main__":
    unittest.main()
