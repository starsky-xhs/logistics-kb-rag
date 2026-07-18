"""哈希向量测试：确定性、归一化与维度。"""

from __future__ import annotations

import math
import unittest

from app.config import Settings
from app.embed import HashEmbedder, build_embedder
from app.errors import ConfigError


class TestHashEmbedder(unittest.TestCase):
    def test_same_text_is_deterministic_across_instances(self) -> None:
        left = HashEmbedder(dim=128, salt="s1")
        right = HashEmbedder(dim=128, salt="s1")
        self.assertEqual(
            list(left.embed_query("退款到账时效")), list(right.embed_query("退款到账时效"))
        )

    def test_different_salt_changes_vector(self) -> None:
        left = HashEmbedder(dim=128, salt="s1").embed_query("退款到账时效")
        right = HashEmbedder(dim=128, salt="s2").embed_query("退款到账时效")
        self.assertNotEqual(list(left), list(right))

    def test_vector_is_l2_normalized(self) -> None:
        vector = HashEmbedder(dim=64).embed_query("物流 运费 refund 2026")
        norm = math.sqrt(sum(value * value for value in vector))
        self.assertAlmostEqual(norm, 1.0, places=5)
        self.assertEqual(len(vector), 64)

    def test_batch_encoding_matches_single(self) -> None:
        embedder = HashEmbedder(dim=64)
        batch = embedder.embed_documents(["退款", "运费"])
        self.assertEqual(len(batch), 2)
        self.assertEqual(list(batch[0]), list(embedder.embed_query("退款")))
        self.assertEqual(embedder.embed_documents([]), [])

    def test_empty_and_dimension_validation(self) -> None:
        vector = HashEmbedder(dim=32).embed_query("")
        self.assertEqual(len(vector), 32)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in vector)), 0.0)
        with self.assertRaises(ConfigError):
            HashEmbedder(dim=4)

    def test_factory(self) -> None:
        embedder = build_embedder(Settings(embed_provider="hash", embed_dim=64))
        self.assertEqual(embedder.dim, 64)
        with self.assertRaises(ConfigError):
            build_embedder(Settings(embed_provider="http", embed_api_key=""))


if __name__ == "__main__":
    unittest.main()
