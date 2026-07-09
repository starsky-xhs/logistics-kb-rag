"""物流知识库 RAG 服务。

核心包只依赖 Python 标准库；FastAPI、httpx 与文档解析依赖均为可选。
"""

from __future__ import annotations

import sys

__version__ = "1.0.0"


def configure_utf8_stdio() -> None:
    """在 Windows 控制台把标准输出切到 UTF-8。

    cmd/PowerShell 的默认代码页是 GBK；不处理时 ``python -m app.demo`` 的中文
    在重定向或 CI 日志里会变成乱码。终端本身仍可正常显示 UTF-8。
    """

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
