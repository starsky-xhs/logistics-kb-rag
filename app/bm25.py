"""纯标准库 BM25 检索。

中文没有天然空格，这里把连续汉字切成字符 bigram；英文和数字按词切分。
不引入 jieba 是为了让核心路径零依赖、结果完全确定，代价是召回粒度更粗，
因此线上再叠加向量检索做互补。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

_CJK_RE: Final = re.compile(r"[\u4e00-\u9fff]+")
_WORD_RE: Final = re.compile(r"[A-Za-z0-9]+(?:[._+\-][A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """中文 bigram + 英文/数字词，保留重复次数供 BM25 统计词频。"""

    tokens: list[str] = []
    for run in _CJK_RE.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    tokens.extend(match.group(0).lower() for match in _WORD_RE.finditer(text))
    return tokens


@dataclass(slots=True)
class _DocumentStat:
    term_freq: Counter[str]
    length: int


class BM25Index:
    """内存 BM25 索引。

    语料规模在万级分块以内时，内存索引足够；更大规模应换成倒排表或
    SQLite FTS5，而不是继续在 Python 里线性扫描。
    """

    def __init__(
        self,
        documents: Sequence[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not 0.0 <= b <= 1.0:
            raise ValueError("BM25 的 b 必须在 0 到 1 之间")
        if k1 <= 0:
            raise ValueError("BM25 的 k1 必须大于 0")
        self.k1 = k1
        self.b = b
        self._stats: list[_DocumentStat] = []
        self._doc_freq: Counter[str] = Counter()
        total_length = 0

        for document in documents:
            tokens = tokenize(document)
            frequencies = Counter(tokens)
            self._stats.append(_DocumentStat(frequencies, len(tokens)))
            total_length += len(tokens)
            self._doc_freq.update(frequencies.keys())

        self.size = len(self._stats)
        self.average_length = total_length / self.size if self.size else 0.0

    def idf(self, term: str) -> float:
        """BM25 的概率型 IDF，恒为正，避免高频词得到负分。"""

        if not self.size:
            return 0.0
        df = self._doc_freq.get(term, 0)
        return math.log(1.0 + (self.size - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """返回 ``(文档下标, 分数)``，按分数降序、下标升序排列。"""

        if top_k <= 0 or not self.size:
            return []
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms:
            return []
        results: list[tuple[int, float]] = []
        for index, stat in enumerate(self._stats):
            score = 0.0
            for term in terms:
                frequency = stat.term_freq.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * stat.length / (self.average_length or 1.0)
                )
                score += self.idf(term) * frequency * (self.k1 + 1.0) / denominator
            if score > 0:
                results.append((index, score))
        results.sort(key=lambda item: (-item[1], item[0]))
        return results[:top_k]

    def document_frequencies(self, terms: Iterable[str]) -> dict[str, int]:
        """调试用：查看查询词在语料中的文档频率。"""

        return {term: self._doc_freq.get(term, 0) for term in terms}
