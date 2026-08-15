"""离线评测。

评测不依赖外部模型：使用真实入库、真实混合检索、真实生成器（默认 mock），
只把期望答案整理为 gold set。指标定义：

* Recall@k：期望来源文件是否出现在前 k 个检索结果中；
* MRR：第一个命中期望来源的排名的倒数均值；
* 引用命中率：最终回答的引用中是否包含期望来源；
* 拒答判断准确率：可回答问题未拒答、不可回答问题已拒答的比例；
* 关键点覆盖率：期望关键点的内容词在回答中的覆盖程度。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .errors import ValidationError
from .models import Answer
from .retriever import content_terms
from .service import RAGService

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(slots=True)
class GoldSample:
    """一条评测样本。"""

    sample_id: str
    question: str
    expected_source: str | None
    key_points: list[str]


@dataclass(slots=True)
class CaseResult:
    """单题评测结果。"""

    sample_id: str
    question: str
    expected_source: str | None
    top_sources: list[str]
    reciprocal_rank: float
    citation_hit: bool | None
    key_point_coverage: float
    refused: bool
    refusal_correct: bool
    latency_ms: float
    answer_chars: int


@dataclass(slots=True)
class EvalReport:
    """完整评测报告。"""

    gold_path: str
    generated_at: str
    ks: tuple[int, ...]
    metrics: dict[str, float]
    counts: dict[str, int]
    cases: list[CaseResult]
    config: dict[str, Any] = field(default_factory=dict)

    def metric_lines(self) -> list[tuple[str, str, str]]:
        lines: list[tuple[str, str, str]] = []
        for k in self.ks:
            lines.append(
                (
                    f"Recall@{k}",
                    _percent(self.metrics.get(f"recall@{k}", 0.0)),
                    "期望来源出现在前 k 个检索结果中",
                )
            )
        lines.extend(
            [
                ("MRR", f"{self.metrics.get('mrr', 0.0):.4f}", "首个正确来源排名倒数的均值"),
                (
                    "引用命中率",
                    _percent(self.metrics.get("citation_hit_rate", 0.0)),
                    "回答引用中包含期望来源的比例",
                ),
                (
                    "拒答判断准确率",
                    _percent(self.metrics.get("refusal_accuracy", 0.0)),
                    "可回答未拒答 + 不可回答已拒答",
                ),
                (
                    "关键点覆盖率",
                    _percent(self.metrics.get("key_point_coverage", 0.0)),
                    "期望关键点内容词在回答中的平均覆盖",
                ),
                (
                    "平均延迟",
                    f"{self.metrics.get('avg_latency_ms', 0.0):.1f} ms",
                    "单题检索 + 生成（含评测进程内开销）",
                ),
            ]
        )
        return lines


def load_gold_set(path: str | Path) -> list[GoldSample]:
    """读取并校验 JSONL gold set。"""

    gold_path = Path(path)
    if not gold_path.exists():
        raise ValidationError(f"gold set 不存在：{gold_path}")
    samples: list[GoldSample] = []
    seen: set[str] = set()
    with gold_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"gold set 第 {line_number} 行不是合法 JSON") from exc
            if not isinstance(data, dict):
                raise ValidationError(f"gold set 第 {line_number} 行必须是对象")
            sample_id = str(data.get("id", "")).strip()
            question = str(data.get("question", "")).strip()
            expected = data.get("expected_source")
            key_points = data.get("key_points", [])
            if not sample_id or not question:
                raise ValidationError(f"gold set 第 {line_number} 行缺少 id 或 question")
            if sample_id in seen:
                raise ValidationError(f"gold set 中 id 重复：{sample_id}")
            if expected is not None and not isinstance(expected, str):
                raise ValidationError(f"样本 {sample_id} 的 expected_source 必须是字符串或 null")
            if not isinstance(key_points, list) or not all(
                isinstance(item, str) for item in key_points
            ):
                raise ValidationError(f"样本 {sample_id} 的 key_points 必须是字符串数组")
            seen.add(sample_id)
            samples.append(
                GoldSample(
                    sample_id=sample_id,
                    question=question,
                    expected_source=expected,
                    key_points=[item for item in key_points if item.strip()],
                )
            )
    if not samples:
        raise ValidationError(f"gold set 为空：{gold_path}")
    return samples


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _display_path(value: str) -> str:
    """报告中优先展示仓库内相对路径，避免把本机绝对路径提交上去。"""

    path = Path(value)
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except (OSError, ValueError):
        return value


def _key_point_hit(point: str, answer: str) -> bool:
    point_terms = content_terms(point)
    if not point_terms:
        return False
    answer_terms = content_terms(answer)
    return len(point_terms & answer_terms) / len(point_terms) >= 0.5


def _score_case(sample: GoldSample, answer: Answer) -> CaseResult:
    top_sources: list[str] = []
    for hit in answer.hits:
        if hit.chunk.source not in top_sources:
            top_sources.append(hit.chunk.source)
    reciprocal_rank = 0.0
    if sample.expected_source:
        for rank, source in enumerate(top_sources, start=1):
            if source == sample.expected_source:
                reciprocal_rank = 1.0 / rank
                break
    citation_hit: bool | None = None
    if sample.expected_source:
        citation_hit = sample.expected_source in answer.citation_sources
    key_point_coverage = 0.0
    if sample.key_points:
        key_point_coverage = sum(
            1 for point in sample.key_points if _key_point_hit(point, answer.text)
        ) / len(sample.key_points)
    if sample.expected_source is None:
        refusal_correct = answer.refused
    else:
        refusal_correct = not answer.refused
    return CaseResult(
        sample_id=sample.sample_id,
        question=sample.question,
        expected_source=sample.expected_source,
        top_sources=top_sources,
        reciprocal_rank=reciprocal_rank,
        citation_hit=citation_hit,
        key_point_coverage=key_point_coverage,
        refused=answer.refused,
        refusal_correct=refusal_correct,
        latency_ms=answer.latency_ms,
        answer_chars=len(answer.text),
    )


def _compute_metrics(
    cases: Sequence[CaseResult], ks: Sequence[int]
) -> tuple[dict[str, float], dict[str, int]]:
    answerable = [case for case in cases if case.expected_source is not None]
    unanswerable = [case for case in cases if case.expected_source is None]
    metrics: dict[str, float] = {}
    for k in ks:
        hits = sum(1 for case in answerable if case.expected_source in case.top_sources[:k])
        metrics[f"recall@{k}"] = hits / len(answerable) if answerable else 0.0
    metrics["mrr"] = (
        sum(case.reciprocal_rank for case in answerable) / len(answerable) if answerable else 0.0
    )
    metrics["citation_hit_rate"] = (
        sum(1 for case in answerable if case.citation_hit) / len(answerable) if answerable else 0.0
    )
    metrics["refusal_accuracy"] = (
        sum(1 for case in cases if case.refusal_correct) / len(cases) if cases else 0.0
    )
    metrics["answerable_refusal_rate"] = (
        sum(1 for case in answerable if case.refused) / len(answerable) if answerable else 0.0
    )
    metrics["unanswerable_refusal_rate"] = (
        sum(1 for case in unanswerable if case.refused) / len(unanswerable) if unanswerable else 0.0
    )
    metrics["key_point_coverage"] = (
        sum(case.key_point_coverage for case in answerable) / len(answerable) if answerable else 0.0
    )
    metrics["avg_latency_ms"] = (
        sum(case.latency_ms for case in cases) / len(cases) if cases else 0.0
    )
    counts = {
        "total": len(cases),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "refusal_correct": sum(1 for case in cases if case.refusal_correct),
    }
    return metrics, counts


def evaluate(
    service: RAGService,
    gold_path: str | Path,
    *,
    ks: Iterable[int] = (1, 3, 5),
    report_path: str | Path | None = None,
) -> EvalReport:
    """执行评测，可选地把 Markdown 报告写到磁盘。"""

    normalized_ks = tuple(sorted({int(k) for k in ks if int(k) > 0}))
    if not normalized_ks:
        raise ValidationError("评测的 k 必须包含至少一个正整数")
    samples = load_gold_set(gold_path)
    cases: list[CaseResult] = []
    for sample in samples:
        answer = service.answer(sample.question)
        cases.append(_score_case(sample, answer))
    metrics, counts = _compute_metrics(cases, normalized_ks)
    report = EvalReport(
        gold_path=str(Path(gold_path)),
        generated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        ks=normalized_ks,
        metrics=metrics,
        counts=counts,
        cases=cases,
        config={
            "db_path": str(service.settings.db_path),
            "embedder": service.embedder.name,
            "embed_provider": service.settings.embed_provider,
            "generator": service.generator.name,
            "llm_provider": service.settings.llm_provider,
            "rerank_provider": service.settings.rerank_provider,
            "top_k": service.settings.top_k,
            "corpus": service.store.stats(),
        },
    )
    if report_path is not None:
        write_report(report, report_path)
    return report


def render_markdown(report: EvalReport) -> str:
    """渲染稳定、可提交到仓库的 Markdown 报告。"""

    corpus = report.config.get("corpus", {})
    if not isinstance(corpus, dict):
        corpus = {}
    lines = [
        "# 物流知识库 RAG 评测报告",
        "",
        "> 本报告由 `python -m app.cli eval` 自动生成，数字来自本地真实运行，未手工填写。",
        "",
        "## 运行环境",
        "",
        f"- 生成时间：`{report.generated_at}`",
        f"- Gold set：`{_display_path(report.gold_path)}`",
        f"- Embedder：`{report.config.get('embedder', '-')}`"
        f"（provider=`{report.config.get('embed_provider', '-')}`）",
        f"- Generator：`{report.config.get('generator', '-')}`"
        f"（provider=`{report.config.get('llm_provider', '-')}`）",
        f"- Rerank：`{report.config.get('rerank_provider', '-')}`；Top-k={report.config.get('top_k', '-')}",
        f"- 语料：{corpus.get('documents', 0)} 篇文档 / {corpus.get('chunks', 0)} 个分块，"
        f"数据库 `{_display_path(str(corpus.get('db_path', '-')))}`",
        f"- 样本：{report.counts['total']} 题（可回答 {report.counts['answerable']}，"
        f"不可回答 {report.counts['unanswerable']}）",
        "",
        "## 指标",
        "",
        "| 指标 | 结果 | 说明 |",
        "| --- | ---: | --- |",
    ]
    for name, value, description in report.metric_lines():
        lines.append(f"| {name} | {value} | {description} |")
    lines.extend(
        [
            "",
            "## 分题结果",
            "",
            "| ID | 问题 | 期望来源 | Top-1 | MRR | 引用命中 | 关键点覆盖 | 拒答 | 延迟 |",
            "| --- | --- | --- | --- | ---: | --- | ---: | --- | ---: |",
        ]
    )
    for case in report.cases:
        expected = case.expected_source or "（应拒答）"
        top1 = case.top_sources[0] if case.top_sources else "-"
        citation = "-" if case.citation_hit is None else ("是" if case.citation_hit else "否")
        refused = "是" if case.refused else "否"
        lines.append(
            f"| {case.sample_id} | {case.question} | {expected} | {top1} | "
            f"{case.reciprocal_rank:.3f} | {citation} | {case.key_point_coverage:.2f} | "
            f"{refused} | {case.latency_ms:.0f} ms |"
        )

    misses = [
        case
        for case in report.cases
        if (case.expected_source is not None and not case.citation_hit)
        or not case.refusal_correct
        or (case.expected_source is not None and case.key_point_coverage < 0.5)
        or (case.expected_source is not None and case.reciprocal_rank == 0)
    ]
    lines.extend(["", "## 未命中与拒答分析", ""])
    if not misses:
        lines.append("本轮所有样本的检索来源、引用和拒答判断均符合预期。")
    else:
        for case in misses:
            reasons: list[str] = []
            if case.expected_source and case.reciprocal_rank == 0:
                reasons.append("期望来源未进入 Top-k")
            if case.expected_source and case.citation_hit is False:
                reasons.append("回答未引用期望来源")
            if case.expected_source and case.key_point_coverage < 0.5:
                reasons.append(f"关键点覆盖偏低（{case.key_point_coverage:.2f}）")
            if not case.refusal_correct:
                reasons.append("拒答判断与预期不符")
            lines.append(f"- `{case.sample_id}` {case.question}：{'；'.join(reasons)}。")
    lines.extend(
        [
            "",
            "## 复现方式",
            "",
            "```bash",
            "python -m app.cli eval",
            "```",
            "",
            "评测默认读取 `data/gold_set.jsonl`。若数据库为空，CLI 会先入库 `data/sample_docs/`，"
            "整个过程不需要网络或 API Key。",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: EvalReport, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_markdown(report), encoding="utf-8")


def print_summary(report: EvalReport) -> None:
    """CLI 输出：一行一个指标，便于复制到 PR 描述。"""

    for name, value, _description in report.metric_lines():
        print(f"{name:<16} {value}")
    print(
        f"样本数           {report.counts['total']}（可回答 {report.counts['answerable']}，"
        f"不可回答 {report.counts['unanswerable']}）"
    )
