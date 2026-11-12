"""Embedding 后端测试：OpenAI 兼容接口、缓存与降级。"""
from __future__ import annotations

import json
import unittest

import httpx

from issue_agent.config import Settings
from issue_agent.errors import ConfigError
from issue_agent.rag.embed import OpenAICompatibleEmbedder, build_embedder
from issue_agent.rag.index import HashingEmbedder


def _settings(**overrides) -> Settings:
    values = dict(embedding_provider="openai", embedding_api_key="sk-test-embedding",
                  embedding_model="test-embedding", embedding_base_url="https://api.openai.com/v1",
                  embedding_dim=64)
    values.update(overrides)
    return Settings(**values)


class TestEmbedding(unittest.TestCase):
    def test_hashing_is_default(self):
        embedder = build_embedder(Settings())
        self.assertIsInstance(embedder, HashingEmbedder)
        self.assertEqual(len(embedder.embed("hello")), Settings().embedding_dim)

    def test_openai_embedding_and_cache(self):
        calls: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(json.loads(request.content))
            return httpx.Response(200, json={"data": [{"embedding": [0.1] * 8}]})

        settings = _settings()
        client = httpx.Client(base_url=settings.embedding_base_url,
                              transport=httpx.MockTransport(handler))
        embedder = OpenAICompatibleEmbedder(settings, client=client)
        first = embedder.embed("same text")
        second = embedder.embed("same text")
        self.assertEqual(len(first), 8)
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)             # 第二次命中内存缓存
        self.assertEqual(calls[0]["model"], "test-embedding")
        self.assertEqual(calls[0]["input"], "same text")

    def test_falls_back_to_hashing_on_first_failure(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "down"})

        settings = _settings()
        client = httpx.Client(base_url=settings.embedding_base_url,
                              transport=httpx.MockTransport(handler))
        embedder = OpenAICompatibleEmbedder(settings, client=client)
        vector = embedder.embed("hello")
        self.assertEqual(len(vector), settings.embedding_dim)
        self.assertTrue(embedder._degraded)

    def test_openai_provider_requires_key(self):
        with self.assertRaises(ConfigError):
            Settings(embedding_provider="openai", embedding_api_key="").validate()

    def test_unknown_provider_rejected(self):
        with self.assertRaises(ConfigError):
            build_embedder(Settings(embedding_provider="nope"))
