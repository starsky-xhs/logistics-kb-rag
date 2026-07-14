"""标题层级感知的中文分块器。

普通定长切分会把「退货时效」的正文接到「运费标准」的标题下面，引用时无法定位
到正确章节。这里先按 Markdown 标题维护 ``heading_path``，再在章节内部按段落
打包；超长段落按中文句末标点滑窗切分，窗口与重叠长度可配置。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .models import Chunk

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SENTENCE_END_RE = re.compile(r"(?<=[。！？!?；;])\s*")
_FENCE_RE = re.compile(r"^(```|~~~)")


@dataclass(slots=True)
class _Section:
    heading_path: tuple[str, ...]
    lines: list[str]


class MarkdownChunker:
    """按标题路径分块，输出确定性的 ``Chunk`` 列表。"""

    def __init__(self, chunk_size: int = 650, overlap: int = 120) -> None:
        if chunk_size < 120:
            raise ValueError("chunk_size 不能小于 120")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap 必须大于等于 0 且小于 chunk_size")
        if overlap > chunk_size // 2:
            raise ValueError("overlap 不应超过 chunk_size 的一半")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(
        self,
        text: str,
        *,
        doc_id: str,
        source: str,
        title: str,
    ) -> list[Chunk]:
        """把一篇文档切为若干块；ordinal 在文档内从 0 递增。"""

        sections = self._sections(text)
        chunks: list[Chunk] = []
        ordinal = 0
        for section in sections:
            body = "\n".join(section.lines).strip()
            if not body:
                continue
            for piece in self._pack(body):
                ordinal += 1
                chunk_id = self._chunk_id(doc_id, ordinal, section.heading_path, piece)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        source=source,
                        title=title,
                        heading_path=section.heading_path,
                        ordinal=ordinal,
                        text=piece,
                    )
                )
        return chunks

    def _sections(self, text: str) -> list[_Section]:
        """扫描标题，生成「路径 + 正文行」。代码块里的 # 不会被误判。"""

        sections: list[_Section] = []
        headings: dict[int, str] = {}
        current_path: tuple[str, ...] = ()
        current_lines: list[str] = []
        in_fence = False

        def flush() -> None:
            if current_lines or current_path:
                sections.append(_Section(current_path, list(current_lines)))
                current_lines.clear()

        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            if _FENCE_RE.match(line.strip()):
                in_fence = not in_fence
                current_lines.append(line)
                continue
            match = None if in_fence else _HEADING_RE.match(line)
            if match:
                flush()
                level = len(match.group(1))
                headings[level] = match.group(2).strip()
                for deeper in [key for key in headings if key > level]:
                    del headings[deeper]
                current_path = tuple(headings[key] for key in sorted(headings))
            else:
                current_lines.append(line)
        flush()

        # 第一个 H1 之前若存在正文，也保留为无标题章节。
        return [section for section in sections if any(line.strip() for line in section.lines)]

    def _pack(self, body: str) -> list[str]:
        """按段落打包，尽量填满窗口，但不跨章节。"""

        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
        pieces: list[str] = []
        buffer = ""

        for paragraph in paragraphs:
            if len(paragraph) > self.chunk_size:
                if buffer:
                    pieces.append(buffer)
                    buffer = ""
                pieces.extend(self._split_long(paragraph))
                continue
            candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph
            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    pieces.append(buffer)
                buffer = paragraph

        if buffer:
            pieces.append(buffer)
        return [piece.strip() for piece in pieces if piece.strip()]

    def _split_long(self, paragraph: str) -> list[str]:
        """按句子滑窗；单句超长时退化为带重叠的硬切分。"""

        sentences = [
            sentence.strip() for sentence in _SENTENCE_END_RE.split(paragraph) if sentence.strip()
        ]
        if not sentences:
            return []
        windows: list[str] = []
        buffer = ""
        for sentence in sentences:
            if len(sentence) > self.chunk_size:
                if buffer:
                    windows.append(buffer)
                    buffer = ""
                windows.extend(self._hard_split(sentence))
                continue

            if not buffer:
                buffer = sentence
                continue
            if len(buffer) + 1 + len(sentence) <= self.chunk_size:
                buffer = f"{buffer} {sentence}"
                continue

            windows.append(buffer)
            carry = self._overlap_tail(buffer)
            buffer = f"{carry} {sentence}".strip() if carry else sentence
            if len(buffer) > self.chunk_size:
                # 重叠后仍超长，优先保证窗口不越界。
                windows.append(buffer[: self.chunk_size].strip())
                buffer = buffer[self.chunk_size - self.overlap :].strip()
        if buffer:
            windows.append(buffer)
        return [window.strip() for window in windows if window.strip()]

    def _hard_split(self, text: str) -> list[str]:
        step = max(1, self.chunk_size - self.overlap)
        return [
            text[start : start + self.chunk_size].strip()
            for start in range(0, len(text), step)
            if text[start : start + self.chunk_size].strip()
        ]

    def _overlap_tail(self, text: str) -> str:
        """从窗口尾部取不超过 overlap 的完整句子，避免从词中间截断。"""

        if self.overlap <= 0:
            return ""
        sentences = [sentence for sentence in _SENTENCE_END_RE.split(text) if sentence.strip()]
        tail: list[str] = []
        size = 0
        for sentence in reversed(sentences):
            sentence = sentence.strip()
            if size + len(sentence) > self.overlap:
                break
            tail.append(sentence)
            size += len(sentence)
        return " ".join(reversed(tail))

    @staticmethod
    def _chunk_id(doc_id: str, ordinal: int, heading_path: tuple[str, ...], text: str) -> str:
        payload = "\x1f".join((doc_id, str(ordinal), "/".join(heading_path), text))
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:24]


def split_markdown(
    text: str,
    *,
    doc_id: str,
    source: str,
    title: str,
    chunk_size: int = 650,
    overlap: int = 120,
) -> list[Chunk]:
    """函数式入口，方便测试和脚本调用。"""

    return MarkdownChunker(chunk_size=chunk_size, overlap=overlap).split(
        text, doc_id=doc_id, source=source, title=title
    )
