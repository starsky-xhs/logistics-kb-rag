"""FastAPI 应用工厂。

核心依赖是可选安装。未安装时导入本模块不会抛异常，只有真正调用
``create_app`` 才会给出带安装命令的错误，方便核心测试在零依赖环境跑完。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from ..config import Settings

logger = logging.getLogger(__name__)

FASTAPI_AVAILABLE = False
FASTAPI_IMPORT_ERROR = ""
FastAPI = None  # type: ignore[assignment,misc]
CORSMiddleware = None  # type: ignore[assignment,misc]

try:  # pragma: no cover - 取决于运行环境是否安装可选依赖
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    from ..errors import GenerationError, KbError, ValidationError
    from ..service import RAGService
    from .routes import router

    FASTAPI_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - 同上
    FASTAPI_IMPORT_ERROR = str(exc)


@asynccontextmanager
async def _lifespan(app: object) -> AsyncIterator[None]:
    """启动时建库，关闭时释放 HTTP 客户端。"""

    state = app.state
    settings = getattr(state, "settings", None) or Settings.from_env()
    service = RAGService(settings)
    state.service = service
    logger.info(
        "HTTP 服务启动完成",
        extra={
            "db_path": str(settings.db_path),
            "embedder": service.embedder.name,
            "generator": service.generator.name,
        },
    )
    try:
        yield
    finally:
        service.close()
        state.service = None
        logger.info("HTTP 服务已停止")


def create_app(settings: Settings | None = None) -> Any:
    """创建 FastAPI 应用；供 uvicorn ``--factory`` 使用。"""

    if not FASTAPI_AVAILABLE:
        raise RuntimeError(
            "未安装 FastAPI 可选依赖，无法启动 HTTP 服务。"
            "请执行 `pip install -r requirements.txt` 后重试。"
            f"原始导入错误：{FASTAPI_IMPORT_ERROR}"
        )

    application = FastAPI(
        title="物流企业知识库智能问答服务",
        description="内部客服知识库 RAG 服务，支持混合检索、引用与流式回答。",
        version="1.0.0",
        lifespan=_lifespan,
    )
    application.state.settings = settings
    application.state.service = None
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    application.include_router(router)

    @application.exception_handler(ValidationError)
    async def handle_validation(_request: object, exc: ValidationError) -> Any:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @application.exception_handler(GenerationError)
    async def handle_generation(_request: object, exc: GenerationError) -> Any:
        logger.exception("生成服务异常")
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @application.exception_handler(KbError)
    async def handle_kb_error(_request: object, exc: KbError) -> Any:
        logger.exception("知识库服务异常")
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    return application
