"""离线端到端演示。

不依赖网络、API Key 和第三方包，覆盖：首次入库、重复入库（幂等）、统计、
三个问答（含一个知识库无法回答的问题）。输出刻意保留检索 explain，便于观察
混合检索各通道的实际贡献。
"""

from __future__ import annotations

import time
from pathlib import Path

from . import configure_utf8_stdio
from .config import Settings
from .service import RAGService

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample_docs"
DEMO_DB = PROJECT_ROOT / "data" / "demo.sqlite3"

QUESTIONS = (
    "退货验收通过后，退款多久能到账？",
    "申请货物破损理赔需要准备哪些材料？",
    "国际海运整柜到鹿特丹的清关流程是什么？",
)


def _remove_demo_db(path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        candidate = path.with_name(path.name + suffix)
        if candidate.exists():
            candidate.unlink()


def _print_header(settings: Settings) -> None:
    print("=" * 72)
    print("物流企业知识库智能问答服务 · 离线演示")
    print("=" * 72)
    print(
        f"生成模式 : LLM_PROVIDER={settings.llm_provider} / EMBED_PROVIDER={settings.embed_provider}"
    )
    print(f"重排模式 : RERANK_PROVIDER={settings.rerank_provider}")
    print(f"演示数据库: {DEMO_DB}")
    print("说明     : 全程离线，使用确定性哈希向量与抽取式回答，不需要 API Key。")
    print()


def _print_report(report: object) -> None:
    print(f"  {report.summary()}")
    warnings = getattr(report, "warnings", [])
    errors = getattr(report, "errors", [])
    for source, reason in warnings:
        print(f"  [跳过] {source}: {reason}")
    for source, reason in errors:
        print(f"  [失败] {source}: {reason}")


def run_demo() -> int:
    """运行演示并返回进程退出码。"""

    configure_utf8_stdio()
    started = time.perf_counter()
    if not SAMPLE_DIR.exists():
        print(f"找不到示例文档目录：{SAMPLE_DIR}")
        return 2

    # 演示库独立于正式 KB_DB_PATH，避免污染用户数据；每次从空库开始保证输出可复现。
    DEMO_DB.parent.mkdir(parents=True, exist_ok=True)
    _remove_demo_db(DEMO_DB)
    settings = Settings.from_env(
        db_path=DEMO_DB,
        llm_provider="mock",
        embed_provider="hash",
        rerank_provider="heuristic",
    )

    service = RAGService(settings)
    try:
        _print_header(settings)

        print("[1/5] 首次增量入库")
        first = service.ingest_dir(SAMPLE_DIR)
        _print_report(first)
        print()

        print("[2/5] 重复入库（验证文件 hash 幂等）")
        second = service.ingest_dir(SAMPLE_DIR)
        _print_report(second)
        print()

        stats = service.stats()
        print("[3/5] 知识库统计")
        print(
            f"  文档 {stats['documents']} 篇 / 分块 {stats['chunks']} 个 / "
            f"向量维度 {stats['embedding_dim']} / 语料版本 {stats['revision']}"
        )
        sources = stats.get("sources", [])
        if isinstance(sources, list):
            for item in sources:
                if isinstance(item, dict):
                    print(f"  - {item['source']}（{item['chunks']} 块）")
        print()

        print("[4/5] 问答演示")
        for index, question in enumerate(QUESTIONS, start=1):
            print("-" * 72)
            print(f"问题 {index}：{question}")
            answer = service.answer(question)
            print("\n检索 explain（Top 3）：")
            if not answer.hits:
                print("  （无命中）")
            for rank, hit in enumerate(answer.hits[:3], start=1):
                explain = hit.explain
                print(
                    f"  #{rank} {hit.chunk.display_title}\n"
                    f"      final={hit.score:.4f}  "
                    f"vector_rank={explain.get('vector_rank')} "
                    f"cos={explain.get('vector_score')}  "
                    f"bm25_rank={explain.get('bm25_rank')} "
                    f"bm25={explain.get('bm25_score')}\n"
                    f"      rrf={explain.get('rrf_score')}  "
                    f"rerank={explain.get('rerank_score')}  "
                    f"heading_hit={explain.get('heading_hit')}  "
                    f"coverage={explain.get('content_term_coverage')}"
                )
            print("\n回答：")
            print(answer.text)
            print(
                f"\n[指标] refused={answer.refused}  "
                f"latency={answer.latency_ms:.1f} ms  "
                f"token_estimate={answer.token_estimate}"
            )
            print()

        print("[5/5] 完成")
        total_ms = (time.perf_counter() - started) * 1000.0
        print(f"演示总耗时 {total_ms:.0f} ms；日志为结构化 JSON，输出到 stderr。")
        return 0
    finally:
        service.close()


def main() -> int:
    return run_demo()


if __name__ == "__main__":
    raise SystemExit(main())
