"""FastAPI 适配层测试。

未安装可选依赖时整类跳过，保证零依赖环境也能跑完核心测试。安装
``requirements.txt`` 后会自动执行真实 HTTP 路由与 SSE 测试。
"""

from __future__ import annotations

import unittest

try:
    from app.api.app import FASTAPI_AVAILABLE, create_app
except Exception:  # pragma: no cover - 可选依赖损坏时也不应影响核心测试
    FASTAPI_AVAILABLE = False
    create_app = None  # type: ignore[assignment]

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    TestClient = None  # type: ignore[assignment,misc]

from tests.support import (
    FREIGHT_DOC,
    REFUND_DOC,
    make_settings,
    temporary_workspace,
    write_doc,
)

_SKIP_REASON = "未安装 FastAPI/httpx 可选依赖，跳过 HTTP 测试"


@unittest.skipUnless(FASTAPI_AVAILABLE and TestClient is not None, _SKIP_REASON)
class TestFastAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_path = self.enterContext(temporary_workspace("api"))
        self.docs_dir = self.tmp_path / "docs"
        self.docs_dir.mkdir()
        write_doc(self.docs_dir, "售后.md", REFUND_DOC)
        write_doc(self.docs_dir, "运费.md", FREIGHT_DOC)
        self.settings = make_settings(self.tmp_path)
        assert create_app is not None
        self.app = create_app(self.settings)
        self._client_context = TestClient(self.app)
        self.client = self._client_context.__enter__()
        response = self.client.post("/api/ingest", json={"path": str(self.docs_dir)})
        self.assertEqual(response.status_code, 200, response.text)

    def tearDown(self) -> None:
        self._client_context.__exit__(None, None, None)

    def test_healthz(self) -> None:
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["documents"], 2)
        self.assertGreaterEqual(payload["chunks"], 2)

    def test_search_endpoint(self) -> None:
        response = self.client.get("/api/search", params={"q": "退款多久到账", "top_k": 2})
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["question"], "退款多久到账")
        self.assertTrue(payload["hits"])
        self.assertIn("explain", payload["hits"][0])
        self.assertEqual(payload["hits"][0]["source"], "售后.md")

    def test_chat_sse_events(self) -> None:
        with self.client.stream(
            "POST", "/api/chat", json={"question": "退款多久能到账？"}
        ) as response:
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
            body = "".join(response.iter_text())
        self.assertIn("event: citation", body)
        self.assertIn("event: token", body)
        self.assertIn("event: done", body)
        self.assertIn("1 至 3 个工作日", body)
        self.assertNotIn("event: error", body)

    def test_stats_endpoint(self) -> None:
        self.client.get("/api/search", params={"q": "退款到账"})
        response = self.client.get("/api/stats")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["documents"], 2)
        self.assertIn("config", payload)
        self.assertTrue(payload["recent_queries"])

    def test_validation_error(self) -> None:
        response = self.client.post("/api/chat", json={"question": ""})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
