"""向量化实现。

默认的 ``HashEmbedder`` 是确定性、可离线复现的哈希向量：中文取字符 bigram，
英文/数字取词，映射到固定维度并做 L2 归一化。它不理解语义，但足以让 demo、
测试和评测在没有模型的情况下跑通，也能作为真实向量服务的降级兜底。

``HttpEmbedder`` 走 OpenAI 兼容的 ``/embeddings`` 协议，带超时、指数退避重试
与批量请求；只有显式设置 ``EMBED_PROVIDER=http`` 时才会联网。
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from array import array
from collections import Counter
from collections.abc import Sequence
from typing import Final, Protocol, runtime_checkable

from .bm25 import tokenize as lexical_tokenize
from .config import Settings
from .errors import ConfigError, RetrievalError

Vector = array

_CJK_RE: Final = re.compile(r"[\u4e00-\u9fff]+")
_HTTP_RETRY_STATUS: Final = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


@runtime_checkable
class Embedder(Protocol):
    """向量化接口，便于在哈希与 HTTP 实现之间切换。"""

    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        """批量编码文档。"""

    def embed_query(self, text: str) -> Vector:
        """编码查询。"""


def _normalize(vector: Sequence[float]) -> Vector:
    result = array("f", (float(value) for value in vector))
    norm = math.sqrt(sum(value * value for value in result))
    if norm > 0:
        for index in range(len(result)):
            result[index] = result[index] / norm
    return result


class HashEmbedder:
    """基于哈希 trick 的确定性向量器。

    同一个词在文档与查询中一定落到相同维度；使用带 salt 的 BLAKE2b 而不是
    Python 内置 ``hash``，避免进程重启后哈希随机化导致库中向量失效。
    """

    name = "hash-bigram-v1"

    def __init__(self, dim: int = 512, salt: str = "lkb-hash-v1") -> None:
        if dim < 8:
            raise ConfigError("哈希向量维度至少为 8")
        self.dim = dim
        self.salt = salt

    def _features(self, text: str) -> list[str]:
        features: list[str] = []
        for run in _CJK_RE.findall(text):
            if len(run) == 1:
                features.append(f"zh:{run}")
            else:
                features.extend(f"zh:{run[index : index + 2]}" for index in range(len(run) - 1))
        for token in lexical_tokenize(text):
            if _CJK_RE.fullmatch(token):
                continue
            features.append(f"w:{token}")
        return features

    def _embed(self, text: str) -> Vector:
        frequencies = Counter(self._features(text))
        values = [0.0] * self.dim
        for feature, count in frequencies.items():
            digest = hashlib.blake2b(f"{self.salt}\x1f{feature}".encode(), digest_size=16).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            # 次线性词频，避免长文档因为重复词而淹没短文档。
            values[index] += sign * (1.0 + math.log(count))
        return _normalize(values)

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> Vector:
        return self._embed(text)


class HttpEmbedder:
    """OpenAI 兼容 embedding 客户端。

    ``base`` 可以是 ``https://host/v1``，也可以直接是完整的 ``/embeddings`` 地址。
    API Key 只从配置读取，不写日志、不写数据库。
    """

    name = "http-openai-compatible"

    def __init__(self, settings: Settings) -> None:
        if not settings.embed_api_key:
            raise ConfigError("EMBED_PROVIDER=http 时需要 EMBED_API_KEY")
        try:
            import httpx  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ConfigError(
                "HTTP embedding 需要可选依赖 httpx，请执行 `pip install httpx`"
            ) from exc
        self._httpx = httpx
        self.base_url = settings.embed_api_base
        self.api_key = settings.embed_api_key
        self.model = settings.embed_model
        self.timeout = settings.request_timeout
        self.batch_size = settings.embed_batch_size
        self.max_retries = settings.max_retries
        self.dim = 0
        self._client = httpx.Client(
            timeout=self.timeout,
            follow_redirects=True,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    def _endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        return base if base.endswith("/embeddings") else f"{base}/embeddings"

    def _request_batch(self, texts: Sequence[str]) -> list[Vector]:
        payload = {"model": self.model, "input": list(texts)}
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post(self._endpoint(), json=payload)
                if response.status_code in _HTTP_RETRY_STATUS:
                    raise RetrievalError(f"embedding 服务暂时不可用，HTTP {response.status_code}")
                if response.status_code >= 400:
                    detail = response.text[:200].replace("\n", " ")
                    raise RetrievalError(
                        f"embedding 请求失败，HTTP {response.status_code}：{detail}"
                    )
                data = response.json()
                records = data.get("data")
                if not isinstance(records, list) or not records:
                    raise RetrievalError("embedding 响应缺少 data 数组")
                if all(isinstance(item, dict) and "index" in item for item in records):
                    records = sorted(records, key=lambda item: int(item["index"]))
                if len(records) != len(texts):
                    raise RetrievalError(
                        f"embedding 返回数量不一致：请求 {len(texts)} 条，返回 {len(records)} 条"
                    )
                vectors: list[Vector] = []
                for item in records:
                    embedding = item.get("embedding") if isinstance(item, dict) else None
                    if not isinstance(embedding, list) or not embedding:
                        raise RetrievalError("embedding 向量为空或格式不正确")
                    if self.dim and len(embedding) != self.dim:
                        raise RetrievalError(
                            f"embedding 维度不一致：期望 {self.dim}，实际 {len(embedding)}"
                        )
                    self.dim = len(embedding)
                    vectors.append(_normalize(embedding))
                return vectors
            except (RetrievalError, self._httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(4.0, 0.5 * (2**attempt)))

        raise RetrievalError(f"embedding 调用失败，已重试 {self.max_retries} 次：{last_error}")

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise RetrievalError("embedding 输入不能为空字符串")
        vectors: list[Vector] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._request_batch(batch))
        return vectors

    def embed_query(self, text: str) -> Vector:
        vectors = self.embed_documents([text])
        return vectors[0]

    def close(self) -> None:
        self._client.close()


def build_embedder(settings: Settings) -> Embedder:
    """按配置构造向量器；配置错误会在启动阶段直接暴露。"""

    if settings.embed_provider == "hash":
        return HashEmbedder(dim=settings.embed_dim, salt=settings.embed_salt)
    if settings.embed_provider == "http":
        return HttpEmbedder(settings)
    raise ConfigError(f"未知的 EMBED_PROVIDER：{settings.embed_provider}")
