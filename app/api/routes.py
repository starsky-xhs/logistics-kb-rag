"""HTTP 路由。

``/api/chat`` 使用 SSE，事件与 ``RAGService.stream_answer`` 一一对应：
``citation``、``token``、``done``、``error``。错误事件之后连接关闭，前端据此
结束 loading 状态，而不是把异常当成正常 token 渲染。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from .. import __version__
from ..errors import KbError, ValidationError
from ..models import ChatMessage
from ..service import RAGService
from .schemas import (
    ChatRequest,
    HealthResponse,
    HitOut,
    IngestRequest,
    IngestResponse,
    IngestWarnings,
    SearchResponse,
    StatsResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def get_service(request: Request) -> RAGService:
    """依赖注入入口；测试可以通过 ``app.dependency_overrides`` 替换。"""

    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="服务尚未完成初始化")
    return service


def _sse(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


@router.get("/healthz", response_model=HealthResponse, tags=["ops"])
def healthz(service: RAGService = Depends(get_service)) -> HealthResponse:
    stats = service.store.stats()
    return HealthResponse(
        status="ok",
        version=__version__,
        documents=int(stats["documents"]),
        chunks=int(stats["chunks"]),
        embedder=service.embedder.name,
        generator=service.generator.name,
    )


@router.post("/api/ingest", response_model=IngestResponse, tags=["knowledge"])
def ingest(
    payload: IngestRequest = Body(...),
    service: RAGService = Depends(get_service),
) -> IngestResponse:
    report = service.ingest_dir(payload.path)
    return IngestResponse(
        root=report.root,
        scanned=report.scanned,
        added=report.added,
        updated=report.updated,
        skipped=report.skipped,
        failed=report.failed,
        chunks=report.chunks,
        elapsed_ms=round(report.elapsed_ms, 2),
        errors=[IngestWarnings(file=name, reason=reason) for name, reason in report.errors],
        warnings=[IngestWarnings(file=name, reason=reason) for name, reason in report.warnings],
    )


@router.get("/api/search", response_model=SearchResponse, tags=["knowledge"])
def search(
    q: str = Query(min_length=1, max_length=2000, description="查询语句"),
    top_k: int = Query(default=5, ge=1, le=20),
    source: str | None = Query(default=None, description="仅检索指定来源文件"),
    service: RAGService = Depends(get_service),
) -> SearchResponse:
    hits = service.search(q, top_k=top_k, source=source)
    return SearchResponse(
        question=q,
        hits=[
            HitOut(
                source=hit.chunk.source,
                heading_path=list(hit.chunk.heading_path),
                chunk_id=hit.chunk.chunk_id,
                text=hit.chunk.text,
                score=round(hit.score, 6),
                explain=hit.explain,
            )
            for hit in hits
        ],
    )


@router.post("/api/chat", tags=["chat"])
def chat(
    payload: ChatRequest = Body(...),
    service: RAGService = Depends(get_service),
) -> StreamingResponse:
    history = [
        ChatMessage(role=message.role, content=message.content) for message in payload.history
    ]

    def event_stream() -> Iterator[str]:
        try:
            for event in service.stream_answer(payload.question, history):
                yield _sse(event.kind, event.data)
        except ValidationError as exc:
            yield _sse("error", {"code": "validation_error", "message": str(exc)})
        except KbError as exc:
            logger.exception("流式回答失败", extra={"question": payload.question})
            yield _sse("error", {"code": "generation_error", "message": str(exc)})
        except Exception:
            logger.exception("流式回答出现未预期错误", extra={"question": payload.question})
            yield _sse("error", {"code": "internal_error", "message": "服务内部错误，请稍后重试"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/api/stats", response_model=StatsResponse, tags=["ops"])
def stats(service: RAGService = Depends(get_service)) -> StatsResponse:
    data = service.stats()
    return StatsResponse(
        documents=int(data["documents"]),
        chunks=int(data["chunks"]),
        embedding_dim=int(data["embedding_dim"]),
        revision=int(data["revision"]),
        db_path=str(data["db_path"]),
        sources=data["sources"],  # type: ignore[arg-type]
        config=data["config"],  # type: ignore[arg-type]
        embedder=str(data["embedder"]),
        generator=str(data["generator"]),
        recent_queries=data["recent_queries"],  # type: ignore[arg-type]
    )
