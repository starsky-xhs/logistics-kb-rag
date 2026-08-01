"""回答生成。

默认 ``MockGenerator`` 是抽取式的：从混合检索命中中挑选与问题最相关的依据句，
拼接成结构化中文回答，并强制附上 ``[n]`` 引用。它不调用模型，但遵守与真实
生成器相同的输入输出契约，因此 demo、单元测试和评测都能离线复现。

``HttpGenerator`` 面向 OpenAI 兼容的 ``/chat/completions``，支持 SSE 流式输出、
超时、有限重试、上下文预算和拒答标记。只有 ``LLM_PROVIDER=http`` 时才会联网。
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from .config import Settings
from .errors import ConfigError, GenerationError
from .models import ChatMessage, Citation, SearchHit, estimate_tokens
from .retriever import content_terms

logger = logging.getLogger(__name__)

REFUSAL_TEXT = (
    "抱歉，当前知识库中没有找到足以回答该问题的依据，因此不作推测性回答。\n"
    "建议补充更具体的关键词（如运单号规则、费用类型或业务环节），"
    "或转人工客服核实最新政策。"
)
_NO_ANSWER_MARKER = "NO_ANSWER"
_SENTENCE_RE = re.compile(r"(?<=[。！？!?；;])\s*|\n+")
_CITATION_RE = re.compile(r"\[(\d+)\]")
_NUMBER_HINT_RE = re.compile(r"\d|[一二三四五六七八九十百千万]|%|％|元|天|工作日|小时|千克|公斤")
_QUESTION_NUMBER_HINTS = ("多久", "多少", "几天", "比例", "标准", "费用", "价格", "时限", "期限")


@dataclass(slots=True)
class GenerationResult:
    """生成器的非流式结果。"""

    text: str
    refused: bool
    citations: list[Citation] = field(default_factory=list)
    token_estimate: int = 0


@dataclass(slots=True)
class StreamEvent:
    """SSE 事件：citation / token / done。"""

    kind: Literal["citation", "token", "done"]
    data: dict[str, Any]


class Generator(Protocol):
    """生成器协议。"""

    name: str

    def generate(
        self,
        question: str,
        hits: Sequence[SearchHit],
        history: Sequence[ChatMessage] | None = None,
    ) -> GenerationResult:
        """一次性生成完整回答。"""

    def stream(
        self,
        question: str,
        hits: Sequence[SearchHit],
        history: Sequence[ChatMessage] | None = None,
    ) -> Iterator[StreamEvent]:
        """流式生成。"""


def _split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for piece in _SENTENCE_RE.split(text):
        cleaned = re.sub(r"^\s*(?:[-*+]|\d+[.、)])\s*", "", piece).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        if len(cleaned) >= 4:
            sentences.append(cleaned)
    return sentences


def _citation_to_dict(citation: Citation) -> dict[str, Any]:
    return {
        "index": citation.index,
        "source": citation.source,
        "heading_path": list(citation.heading_path),
        "chunk_id": citation.chunk_id,
        "label": citation.label,
        "snippet": citation.snippet,
    }


def _insufficient_evidence(question: str, hits: Sequence[SearchHit]) -> bool:
    """词面证据门槛。

    哈希向量可能因碰撞召回无关片段，不能只看余弦分数。这里要求前三个命中
    至少有一个与问题有足够的实词覆盖；这是对 demo/评测中最关键的「拒答」
    能力的硬约束。
    """

    terms = content_terms(question)
    if not terms:
        return True
    best_coverage = 0.0
    best_matches = 0
    for hit in list(hits)[:3]:
        text = f"{hit.chunk.heading_text} {hit.chunk.title} {hit.chunk.text}"
        hit_terms = content_terms(text)
        matches = len(terms & hit_terms)
        coverage = matches / len(terms)
        best_coverage = max(best_coverage, coverage)
        best_matches = max(best_matches, matches)
    if best_coverage < 0.28:
        return True
    return best_matches < 2 and len(terms) > 2


class MockGenerator:
    """确定性的抽取式生成器。"""

    name = "mock-extractive"

    def generate(
        self,
        question: str,
        hits: Sequence[SearchHit],
        history: Sequence[ChatMessage] | None = None,
    ) -> GenerationResult:
        del history  # 抽取式生成不做多轮改写，保留接口兼容真实生成器。
        question = question.strip()
        if not question or not hits or _insufficient_evidence(question, hits):
            return GenerationResult(
                text=REFUSAL_TEXT,
                refused=True,
                citations=[],
                token_estimate=estimate_tokens(REFUSAL_TEXT),
            )

        terms = content_terms(question)
        used_hits: list[tuple[SearchHit, list[str]]] = []
        seen_sentences: list[set[str]] = []
        evidence_count = 0
        list_question = any(keyword in question for keyword in ("哪些", "材料", "清单", "所需"))
        candidate_hits = list(hits)[:3]
        if candidate_hits and candidate_hits[0].rerank_score > 0:
            # 明显低于首条的重排结果通常是向量噪声，不再拿来做抽取依据。
            floor = candidate_hits[0].rerank_score * 0.60
            candidate_hits = [
                hit for hit in candidate_hits if hit.rerank_score == 0 or hit.rerank_score >= floor
            ]

        for hit_position, hit in enumerate(candidate_hits, start=1):
            section_path = hit.chunk.heading_path[1:] if len(hit.chunk.heading_path) > 1 else ()
            heading_terms = content_terms(" > ".join(section_path))
            heading_bonus = (
                len(terms & heading_terms) / max(1, min(4, len(terms))) if terms else 0.0
            )
            section_relevant = bool(terms & heading_terms)
            sentence_candidates = _split_sentences(hit.chunk.text)
            scored: list[tuple[float, int, str]] = []
            for index, sentence in enumerate(sentence_candidates):
                overlap = terms & content_terms(sentence)
                if not overlap:
                    # 「需要哪些材料」这类问题，命中章节里的列表项不一定重复问题词；
                    # 标题已经匹配时，允许把章节内条目作为补充依据。
                    if not (hit_position == 1 and list_question and section_relevant):
                        continue
                    score = 0.06 + 0.05 * heading_bonus
                else:
                    coverage = len(overlap) / len(terms)
                    score = coverage + 0.20 * heading_bonus
                    if any(
                        hint in question for hint in _QUESTION_NUMBER_HINTS
                    ) and _NUMBER_HINT_RE.search(sentence):
                        score += 0.12
                    if 20 <= len(sentence) <= 180:
                        score += 0.03
                scored.append((score, index, sentence))

            special_list = hit_position == 1 and list_question and section_relevant
            if special_list:
                # 材料清单按原文顺序阅读才自然，不按词面覆盖分打乱条目。
                scored.sort(key=lambda item: item[1])
            else:
                scored.sort(key=lambda item: (-item[0], item[1], item[2]))
            picked: list[str] = []
            per_hit_limit = 6 if special_list else (3 if hit_position == 1 else 2)
            for score, _index, sentence in scored:
                signature = content_terms(sentence)
                if not signature:
                    continue
                if any(
                    len(signature & previous) / max(1, min(len(signature), len(previous))) >= 0.80
                    for previous in seen_sentences
                ):
                    continue
                if len(picked) >= per_hit_limit or evidence_count >= 6:
                    break
                if score < 0.05:
                    continue
                seen_sentences.append(signature)
                picked.append(sentence)
                evidence_count += 1
            if picked:
                used_hits.append((hit, picked))

        if not used_hits:
            return GenerationResult(
                text=REFUSAL_TEXT,
                refused=True,
                citations=[],
                token_estimate=estimate_tokens(REFUSAL_TEXT),
            )

        citations: list[Citation] = []
        citation_by_chunk: dict[str, Citation] = {}
        for hit, sentences in used_hits:
            citation = Citation(
                index=len(citations) + 1,
                source=hit.chunk.source,
                heading_path=hit.chunk.heading_path,
                chunk_id=hit.chunk.chunk_id,
                snippet=sentences[0],
            )
            citations.append(citation)
            citation_by_chunk[hit.chunk.chunk_id] = citation

        intro = _intro_for_question(question)
        lines = [intro, ""]
        point_index = 1
        for hit, sentences in used_hits:
            citation = citation_by_chunk[hit.chunk.chunk_id]
            # 同一片段可能贡献多条依据，只展示一个引用编号。
            for sentence in sentences:
                lines.append(f"{point_index}. {sentence} [{citation.index}]")
                point_index += 1
        lines.append("")
        lines.append("以上内容由知识库检索结果抽取整理，具体执行以最新合同和系统公示为准。")
        lines.append("")
        lines.append("参考来源：")
        for citation in citations:
            lines.append(f"[{citation.index}] {citation.label}")

        text = "\n".join(lines)
        return GenerationResult(
            text=text,
            refused=False,
            citations=citations,
            token_estimate=estimate_tokens(text),
        )

    def stream(
        self,
        question: str,
        hits: Sequence[SearchHit],
        history: Sequence[ChatMessage] | None = None,
    ) -> Iterator[StreamEvent]:
        result = self.generate(question, hits, history)
        for citation in result.citations:
            yield StreamEvent("citation", _citation_to_dict(citation))
        # 按小片段推送，模拟真实模型的增量输出；不是逐 token，但契约一致。
        for start in range(0, len(result.text), 18):
            yield StreamEvent("token", {"text": result.text[start : start + 18]})
        yield StreamEvent(
            "done",
            {
                "text": result.text,
                "refused": result.refused,
                "citations": [_citation_to_dict(item) for item in result.citations],
                "token_estimate": result.token_estimate,
                "generator": self.name,
            },
        )


def _intro_for_question(question: str) -> str:
    if any(keyword in question for keyword in ("多久", "几天", "时限", "期限", "时效")):
        return "根据知识库中的时效规定："
    if any(keyword in question for keyword in ("哪些", "材料", "需要准备", "怎么办", "如何办理")):
        return "根据知识库中的办理要求："
    if any(keyword in question for keyword in ("谁承担", "责任", "由谁", "赔付")):
        return "根据知识库中的责任划分与赔付规则："
    if any(keyword in question for keyword in ("多少", "比例", "费用", "价格", "怎么收")):
        return "根据知识库中的计费与赔付标准："
    return "根据知识库中的相关规定："


class HttpGenerator:
    """OpenAI 兼容 Chat Completions 生成器（含 SSE 流式）。"""

    name = "http-openai-compatible"

    def __init__(self, settings: Settings) -> None:
        if not settings.llm_api_key:
            raise ConfigError("LLM_PROVIDER=http 时需要 LLM_API_KEY")
        try:
            import httpx  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ConfigError("HTTP 生成需要可选依赖 httpx，请执行 `pip install httpx`") from exc
        self._httpx = httpx
        self.base_url = settings.llm_api_base
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model
        self.timeout = settings.request_timeout
        self.max_retries = settings.max_retries
        self.temperature = settings.temperature
        self.max_tokens = settings.max_tokens
        self.max_context_chars = settings.max_context_chars

    def _endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def _select_hits(self, hits: Sequence[SearchHit]) -> list[SearchHit]:
        selected: list[SearchHit] = []
        used = 0
        for hit in hits:
            block = f"[{len(selected) + 1}] {hit.chunk.display_title}\n{hit.chunk.text}\n\n"
            if selected and used + len(block) > self.max_context_chars:
                break
            if len(block) > self.max_context_chars:
                block = block[: self.max_context_chars]
            selected.append(hit)
            used += len(block)
            if len(selected) >= 5:
                break
        return selected

    def _build_messages(
        self,
        question: str,
        hits: Sequence[SearchHit],
        history: Sequence[ChatMessage] | None,
    ) -> tuple[list[dict[str, str]], list[Citation]]:
        selected = self._select_hits(hits)
        context_lines: list[str] = []
        citations: list[Citation] = []
        for index, hit in enumerate(selected, start=1):
            context_lines.append(f"[{index}] 来源：{hit.chunk.display_title}\n{hit.chunk.text}")
            citations.append(
                Citation(
                    index=index,
                    source=hit.chunk.source,
                    heading_path=hit.chunk.heading_path,
                    chunk_id=hit.chunk.chunk_id,
                    snippet=hit.chunk.text[:120],
                )
            )

        system = (
            "你是迅达物流的内部知识库问答助手。只能依据给定的『知识库资料』回答，"
            "不得编造政策、金额或时效。每个事实句后必须用 [编号] 标注依据。"
            "如果资料不足以回答，只输出 NO_ANSWER，不要输出其他内容。"
            "回答使用简洁、专业的中文，先给结论，再列依据。"
        )
        user_parts = ["知识库资料：", "\n\n".join(context_lines), ""]
        if history:
            history_lines = []
            budget = 1200
            for message in history[-6:]:
                message.validate()
                content = message.content.strip()
                if len(content) > 500:
                    content = content[:500] + "…"
                if budget - len(content) < 0:
                    break
                budget -= len(content)
                history_lines.append(f"{'用户' if message.role == 'user' else '助手'}：{content}")
            if history_lines:
                user_parts.extend(
                    ["历史对话（仅供指代消解，不可作为事实依据）：", "\n".join(history_lines), ""]
                )
        user_parts.append(f"问题：{question}")
        return (
            [
                {"role": "system", "content": system},
                {"role": "user", "content": "\n".join(user_parts)},
            ],
            citations,
        )

    def _iter_sse_text(self, messages: list[dict[str, str]]) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        last_error: Exception | None = None
        emitted = False
        for attempt in range(self.max_retries + 1):
            try:
                with self._httpx.Client(timeout=self.timeout) as client:
                    with client.stream(
                        "POST", self._endpoint(), json=payload, headers=headers
                    ) as response:
                        if response.status_code >= 400:
                            detail = response.read().decode("utf-8", errors="replace")[:200]
                            raise GenerationError(
                                f"生成接口返回 HTTP {response.status_code}：{detail}"
                            )
                        for raw_line in response.iter_lines():
                            line = raw_line.strip()
                            if not line or not line.startswith("data:"):
                                continue
                            chunk = line[5:].strip()
                            if chunk == "[DONE]":
                                return
                            try:
                                data = json.loads(chunk)
                            except json.JSONDecodeError:
                                logger.warning(
                                    "忽略无法解析的 SSE 数据块", extra={"chunk": chunk[:120]}
                                )
                                continue
                            if data.get("error"):
                                raise GenerationError(f"生成接口错误：{data['error']}")
                            for choice in data.get("choices", []):
                                delta = choice.get("delta") or {}
                                content = delta.get("content")
                                if content:
                                    emitted = True
                                    yield str(content)
                return
            except (GenerationError, self._httpx.HTTPError) as exc:
                last_error = exc
                if emitted or attempt >= self.max_retries:
                    break
                time.sleep(min(4.0, 0.5 * (2**attempt)))
        raise GenerationError(f"生成调用失败，已重试 {self.max_retries} 次：{last_error}")

    def _prepare(
        self, question: str, hits: Sequence[SearchHit], history: Sequence[ChatMessage] | None
    ) -> tuple[list[dict[str, str]], list[Citation]]:
        if not question.strip():
            raise GenerationError("问题不能为空")
        if not hits or _insufficient_evidence(question, hits):
            return [], []
        return self._build_messages(question, hits, history)

    def generate(
        self,
        question: str,
        hits: Sequence[SearchHit],
        history: Sequence[ChatMessage] | None = None,
    ) -> GenerationResult:
        if not question.strip():
            raise GenerationError("问题不能为空")
        if not hits or _insufficient_evidence(question, hits):
            return GenerationResult(
                text=REFUSAL_TEXT,
                refused=True,
                citations=[],
                token_estimate=estimate_tokens(REFUSAL_TEXT),
            )
        messages, citations = self._build_messages(question, hits, history)
        text_parts: list[str] = []
        for piece in self._iter_sse_text(messages):
            text_parts.append(piece)
        raw_text = "".join(text_parts).strip()
        return self._finalize(raw_text, citations)

    def _finalize(self, text: str, citations: list[Citation]) -> GenerationResult:
        if not text or _NO_ANSWER_MARKER in text[:80]:
            return GenerationResult(
                text=REFUSAL_TEXT,
                refused=True,
                citations=[],
                token_estimate=estimate_tokens(REFUSAL_TEXT),
            )
        used_indices = {int(match.group(1)) for match in _CITATION_RE.finditer(text)}
        used = [citation for citation in citations if citation.index in used_indices]
        if not used:
            # 模型漏标注时补齐来源清单，避免回答没有任何可审计出处。
            used = citations[:3]
            text += "\n\n参考来源：\n" + "\n".join(
                f"[{citation.index}] {citation.label}" for citation in used
            )
        return GenerationResult(
            text=text,
            refused=False,
            citations=used,
            token_estimate=estimate_tokens(text),
        )

    def stream(
        self,
        question: str,
        hits: Sequence[SearchHit],
        history: Sequence[ChatMessage] | None = None,
    ) -> Iterator[StreamEvent]:
        if not question.strip():
            raise GenerationError("问题不能为空")
        if not hits or _insufficient_evidence(question, hits):
            result = GenerationResult(
                text=REFUSAL_TEXT,
                refused=True,
                citations=[],
                token_estimate=estimate_tokens(REFUSAL_TEXT),
            )
            yield StreamEvent(
                "done",
                {
                    "text": result.text,
                    "refused": True,
                    "citations": [],
                    "token_estimate": result.token_estimate,
                    "generator": self.name,
                },
            )
            return

        messages, citations = self._build_messages(question, hits, history)
        for citation in citations:
            yield StreamEvent("citation", _citation_to_dict(citation))
        text_parts: list[str] = []
        for piece in self._iter_sse_text(messages):
            text_parts.append(piece)
            yield StreamEvent("token", {"text": piece})
        result = self._finalize("".join(text_parts).strip(), citations)
        if result.refused:
            # 回答被模型判为资料不足时，前端应以 done 事件覆盖此前展示的候选引用。
            yield StreamEvent(
                "done",
                {
                    "text": result.text,
                    "refused": True,
                    "citations": [],
                    "token_estimate": result.token_estimate,
                    "generator": self.name,
                },
            )
            return
        yield StreamEvent(
            "done",
            {
                "text": result.text,
                "refused": False,
                "citations": [_citation_to_dict(item) for item in result.citations],
                "token_estimate": result.token_estimate,
                "generator": self.name,
            },
        )


def build_generator(settings: Settings) -> Generator:
    """按配置构造生成器。"""

    if settings.llm_provider == "mock":
        return MockGenerator()
    if settings.llm_provider == "http":
        return HttpGenerator(settings)
    raise ConfigError(f"未知的 LLM_PROVIDER：{settings.llm_provider}")
