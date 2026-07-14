"""加载器测试：编码回退、标题推断、目录扫描与文件哈希。"""

from __future__ import annotations

import hashlib
import importlib.util
import unittest

from app.errors import LoaderError, OptionalDependencyError
from app.loaders import (
    file_sha256,
    infer_title,
    iter_source_files,
    load_file,
    relative_source,
)
from tests.support import temporary_workspace, write_doc


class TestLoaders(unittest.TestCase):
    def setUp(self) -> None:
        self.root = self.enterContext(temporary_workspace("loaders"))
        self.docs = self.root / "docs"
        self.docs.mkdir()

    def test_markdown_and_title(self) -> None:
        path = write_doc(self.docs, "a.md", "# 售后政策\n\n正文内容。\n")
        text, metadata = load_file(path)
        self.assertIn("正文内容", text)
        self.assertEqual(metadata["format"], "md")
        self.assertEqual(infer_title(text, "fallback"), "售后政策")

    def test_gb18030_fallback(self) -> None:
        path = self.docs / "gbk.txt"
        path.write_bytes("中文编码测试".encode("gb18030"))
        text, metadata = load_file(path)
        self.assertIn("中文编码测试", text)
        self.assertEqual(metadata["encoding"], "gb18030")

    def test_iter_files_ignores_hidden_children(self) -> None:
        write_doc(self.docs, "a.md", "# a\n")
        write_doc(self.docs, "b.txt", "b")
        write_doc(self.docs, ".hidden", "ignored")
        write_doc(self.docs, ".cache/c.md", "# hidden")
        (self.docs / "image.png").write_bytes(b"not supported")
        files = iter_source_files(self.docs)
        self.assertEqual([path.name for path in files], ["a.md", "b.txt"])
        self.assertEqual(relative_source(files[0], self.docs), "a.md")

    def test_file_hash_matches_hashlib(self) -> None:
        path = write_doc(self.docs, "hash.md", "内容")
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(file_sha256(path), expected)

    def test_unsupported_and_missing_path(self) -> None:
        with self.assertRaises(LoaderError):
            load_file(write_doc(self.docs, "x.csv", "a,b"))
        with self.assertRaises(LoaderError):
            iter_source_files(self.root / "missing")

    @unittest.skipIf(importlib.util.find_spec("docx") is not None, "python-docx 已安装")
    def test_missing_optional_docx_dependency(self) -> None:
        path = write_doc(self.docs, "x.docx", "not a real docx")
        with self.assertRaises(OptionalDependencyError) as context:
            load_file(path)
        self.assertIn("python-docx", str(context.exception))


if __name__ == "__main__":
    unittest.main()
