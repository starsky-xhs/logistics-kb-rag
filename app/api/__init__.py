"""FastAPI 适配层（可选依赖）。

核心业务全部在 ``app`` 包内，本目录只负责 HTTP 协议、schema 与 SSE 编码。
未安装 FastAPI 时导入本包不会崩溃，``create_app`` 会给出明确安装提示。
"""

from __future__ import annotations
