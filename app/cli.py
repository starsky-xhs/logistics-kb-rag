"""命令行入口。

用法::

    python -m app.cli demo
    python -m app.cli ingest data/sample_docs
    python -m app.cli search "退款多久到账"
    python -m app.cli ask "破损理赔需要哪些材料"
    python -m app.cli eval
    python -m app.cli stats
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import configure_utf8_stdio
from .config import Settings
from .errors import KbError, ValidationError
from .eval import evaluate, print_summary
from .models import Answer, SearchHit
from .service import RAGService

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLE_DIR = PROJECT_ROOT / "data" / "sample_docs"
DEFAULT_GOLD_PATH = PROJECT_ROOT / "data" / "gold_set.jsonl"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "docs" / "EVAL_REPORT.md"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="物流企业知识库智能问答服务（离线可运行）",
    )
    parser.add_argument("--version", action="version", version="logistics-kb-rag 1.0.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("demo", help="运行离线端到端演示（等价于 python -m app.demo）")

    ingest = subparsers.add_parser("ingest", help="增量入库一个目录")
    ingest.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_SAMPLE_DIR),
        help="待入库目录，默认 data/sample_docs",
    )

    search = subparsers.add_parser("search", help="只检索，不生成回答")
    search.add_argument("query", help="查询语句")
    search.add_argument("--top-k", type=int, default=None, help="返回条数")
    search.add_argument("--source", default=None, help="仅检索指定来源文件")
    search.add_argument("--json", action="store_true", help="输出 JSON")

    ask = subparsers.add_parser("ask", help="检索并生成带引用的回答")
    ask.add_argument("query", help="问题")
    ask.add_argument("--json", action="store_true", help="输出 JSON")
    ask.add_argument("--explain", action="store_true", help="额外打印检索得分构成")

    eval_parser = subparsers.add_parser("eval", help="运行 gold set 评测")
    eval_parser.add_argument("--gold", default=str(DEFAULT_GOLD_PATH), help="gold_set.jsonl 路径")
    eval_parser.add_argument(
        "--out", default=str(DEFAULT_REPORT_PATH), help="Markdown 报告输出路径"
    )
    eval_parser.add_argument("--k", default="1,3,5", help="Recall@k 的 k 列表，逗号分隔")

    stats = subparsers.add_parser("stats", help="查看语料与查询统计")
    stats.add_argument("--json", action="store_true", help="输出 JSON")
    return parser


def _settings() -> Settings:
    return Settings.from_env()


def _print_hits(hits: Sequence[SearchHit]) -> None:
    if not hits:
        print("未检索到相关片段。")
        return
    for rank, hit in enumerate(hits, start=1):
        snippet = hit.chunk.text.replace("\n", " ")
        if len(snippet) > 150:
            snippet = snippet[:150] + "…"
        print(f"{rank}. {hit.chunk.display_title}  score={hit.score:.4f}")
        print(f"   {snippet}")
        print(
            "   explain:"
            f" vector_rank={hit.vector_rank or '-'} vector={hit.vector_score:.3f}"
            f" bm25_rank={hit.bm25_rank or '-'} bm25={hit.bm25_score:.3f}"
            f" rrf={hit.rrf_score:.5f} rerank={hit.rerank_score:.3f}"
            f" heading_hit={hit.explain.get('heading_hit')}"
            f" coverage={hit.explain.get('content_term_coverage')}"
        )


def _answer_payload(answer: Answer) -> dict[str, Any]:
    return {
        "question": answer.question,
        "answer": answer.text,
        "refused": answer.refused,
        "citations": [asdict(citation) for citation in answer.citations],
        "latency_ms": round(answer.latency_ms, 2),
        "token_estimate": answer.token_estimate,
        "hits": [
            {
                "source": hit.chunk.source,
                "heading_path": list(hit.chunk.heading_path),
                "score": round(hit.score, 6),
                "explain": hit.explain,
            }
            for hit in answer.hits
        ],
    }


def _ensure_example_corpus(service: RAGService) -> None:
    """空库时自动入库示例文档，让 ``eval`` 真正可以一键复现。"""

    corpus = service.store.stats()
    if int(corpus.get("documents", 0)) > 0:
        return
    if not DEFAULT_SAMPLE_DIR.exists():
        raise ValidationError(
            f"数据库为空，且示例目录不存在：{DEFAULT_SAMPLE_DIR}；请先执行 ingest"
        )
    print(f"检测到空库，先入库示例文档：{DEFAULT_SAMPLE_DIR}", file=sys.stderr)
    report = service.ingest_dir(DEFAULT_SAMPLE_DIR)
    if report.failed:
        raise ValidationError(f"示例文档入库失败：{report.errors}")
    if int(service.store.stats().get("documents", 0)) == 0:
        raise ValidationError("示例文档入库后数据库仍为空，请检查数据目录")


def _run_ingest(args: argparse.Namespace, settings: Settings) -> int:
    service = RAGService(settings)
    try:
        report = service.ingest_dir(args.path)
        print(report.summary())
        for source, reason in report.warnings:
            print(f"[跳过] {source}: {reason}", file=sys.stderr)
        for source, reason in report.errors:
            print(f"[失败] {source}: {reason}", file=sys.stderr)
        return 0 if report.ok else 1
    finally:
        service.close()


def _run_search(args: argparse.Namespace, settings: Settings) -> int:
    service = RAGService(settings)
    try:
        hits = service.search(args.query, top_k=args.top_k, source=args.source)
        if args.json:
            payload = [
                {
                    "source": hit.chunk.source,
                    "heading_path": list(hit.chunk.heading_path),
                    "chunk_id": hit.chunk.chunk_id,
                    "text": hit.chunk.text,
                    "score": round(hit.score, 6),
                    "explain": hit.explain,
                }
                for hit in hits
            ]
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            _print_hits(hits)
        return 0
    finally:
        service.close()


def _run_ask(args: argparse.Namespace, settings: Settings) -> int:
    service = RAGService(settings)
    try:
        answer = service.answer(args.query)
        if args.json:
            print(json.dumps(_answer_payload(answer), ensure_ascii=False, indent=2))
            return 0
        print(answer.text)
        if args.explain:
            print("\n检索结果：")
            _print_hits(answer.hits)
        print(f"\n延迟 {answer.latency_ms:.1f} ms；token 估算 {answer.token_estimate}")
        return 0
    finally:
        service.close()


def _run_eval(args: argparse.Namespace, settings: Settings) -> int:
    try:
        ks = tuple(int(item.strip()) for item in args.k.split(",") if item.strip())
    except ValueError as exc:
        raise ValidationError("--k 必须是逗号分隔的整数，例如 1,3,5") from exc
    service = RAGService(settings)
    try:
        _ensure_example_corpus(service)
        report = evaluate(service, args.gold, ks=ks, report_path=args.out)
        print_summary(report)
        print(f"报告已写入：{Path(args.out).resolve()}")
        return 0
    finally:
        service.close()


def _run_stats(args: argparse.Namespace, settings: Settings) -> int:
    service = RAGService(settings)
    try:
        data = service.stats()
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
            return 0
        print(f"数据库：{data['db_path']}")
        print(
            f"文档：{data['documents']}  分块：{data['chunks']}  向量维度：{data['embedding_dim']}"
        )
        print(
            f"Embedder：{data['embedder']}  Generator：{data['generator']}  "
            f"语料版本：{data['revision']}"
        )
        sources = data.get("sources", [])
        if isinstance(sources, list) and sources:
            print("来源文件：")
            for item in sources:
                if isinstance(item, dict):
                    print(f"  - {item['source']}（{item['chunks']} 块）")
        recent = data.get("recent_queries", [])
        if isinstance(recent, list) and recent:
            print("最近查询：")
            for item in recent[-5:]:
                if isinstance(item, dict):
                    print(
                        f"  - [{item['kind']}] {item['query']}  "
                        f"{item['latency_ms']} ms  refused={item['refused']}"
                    )
        return 0
    finally:
        service.close()


def main(argv: Sequence[str] | None = None) -> int:
    configure_utf8_stdio()
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            from .demo import run_demo

            return run_demo()
        settings = _settings()
        if args.command == "ingest":
            return _run_ingest(args, settings)
        if args.command == "search":
            return _run_search(args, settings)
        if args.command == "ask":
            return _run_ask(args, settings)
        if args.command == "eval":
            return _run_eval(args, settings)
        if args.command == "stats":
            return _run_stats(args, settings)
        parser.error(f"未知命令：{args.command}")
        return 2
    except KbError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("已中断。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
