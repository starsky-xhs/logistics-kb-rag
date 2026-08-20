"""配置与模型的基础测试。"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import Settings
from app.errors import ConfigError
from app.models import estimate_tokens
from tests.support import temporary_workspace


class TestSettings(unittest.TestCase):
    def test_from_env_overrides_are_applied(self) -> None:
        with temporary_workspace("config") as tmp:
            env = {
                "KB_DB_PATH": str(Path(tmp) / "x.sqlite3"),
                "RAG_TOP_K": "7",
                "RAG_CHUNK_SIZE": "500",
                "LLM_PROVIDER": "mock",
            }
            with patch.dict(os.environ, env, clear=False):
                settings = Settings.from_env()
            self.assertEqual(settings.top_k, 7)
            self.assertEqual(settings.chunk_size, 500)
            self.assertEqual(settings.db_path.name, "x.sqlite3")

    def test_invalid_values_raise_config_error(self) -> None:
        with patch.dict(os.environ, {"RAG_CHUNK_OVERLAP": "9999"}, clear=False):
            with self.assertRaises(ConfigError):
                Settings.from_env()
        with self.assertRaises(ConfigError):
            Settings(top_k=0)
        with self.assertRaises(ConfigError):
            Settings(embed_provider="unknown")
        with self.assertRaises(ConfigError):
            Settings(embed_provider="http", embed_api_key="")

    def test_describe_does_not_leak_key(self) -> None:
        settings = Settings(embed_provider="http", embed_api_key="secret")
        self.assertNotIn("secret", str(settings.describe()))

    def test_unknown_override_is_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            Settings.from_env(not_a_field=1)


class TestModels(unittest.TestCase):
    def test_estimate_tokens(self) -> None:
        self.assertEqual(estimate_tokens(""), 0)
        self.assertGreater(estimate_tokens("中文中文"), 0)
        self.assertGreater(estimate_tokens("hello world"), 0)


if __name__ == "__main__":
    unittest.main()
