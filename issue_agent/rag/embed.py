"""向量化后端：本地哈希向量（默认）与 OpenAI 兼容 Embedding API。

生产环境用真实 Embedding（OpenAI、bge、SiliconFlow 等）通常能把同义改写召回拉上来；
离线测试与没有 Embedding Key 的部署继续用确定性哈希向量，保证可复现、零依赖。
"""
from __future__ import annotations

from typing import Any

import httpx

from ..config import Settings
from ..errors import ConfigError
from ..obs.logs import get_logger
from ..security.guard import redact
from .index import HashingEmbedder

log = get_logger("issue_agent.embed")


class OpenAICompatibleEmbedder:
    """调用 OpenAI 兼容的 ``/embeddings`` 接口。

    第一次调用失败时降级为哈希向量（整次运行保持一致）；若已经拿到部分向量后失败，
    为避免维度混用导致余弦相似度失真，直接抛错由上层降级报告处理。
    """

    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        if not settings.embedding_api_key and client is None:
            raise ConfigError("EMBEDDING_PROVIDER=openai 需要 EMBEDDING_API_KEY 或 OPENAI_API_KEY")
        self.settings = settings
        self.model = settings.embedding_model
        self._cache: dict[str, list[float]] = {}
        self._fallback = HashingEmbedder(settings.embedding_dim)
        self._degraded = False
        self.dim: int | None = None
        self._client = client or httpx.Client(
            base_url=settings.embedding_base_url,
            timeout=httpx.Timeout(settings.embedding_timeout),
            headers={"Authorization": f"Bearer {settings.embedding_api_key}"},
        )

    def embed(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        if self._degraded:
            return self._cache.setdefault(text, self._fallback.embed(text))
        try:
            vector = self._request(text)
        except Exception as exc:  # noqa: BLE001
            if self._cache:
                # 已有远程向量：再混入 512 维哈希向量会让余弦相似度完全失真。
                raise
            log.warning("Embedding API 不可用，本次运行降级为哈希向量：%s",
                        redact(str(exc))[:200])
            self._degraded = True
            return self._cache.setdefault(text, self._fallback.embed(text))
        self.dim = len(vector)
        self._cache[text] = vector
        return vector

    def _request(self, text: str) -> list[float]:
        response = self._client.post("/embeddings", json={"model": self.model, "input": text})
        response.raise_for_status()
        data = response.json()
        try:
            vector = data["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Embedding API 返回结构异常") from exc
        if not isinstance(vector, list) or not vector:
            raise ValueError("Embedding API 返回空向量")
        return [float(value) for value in vector]


def build_embedder(settings: Settings, client: Any | None = None):
    """按配置返回嵌入器；``hashing`` 为默认，离线可复现。"""
    provider = (settings.embedding_provider or "hashing").lower()
    if provider == "hashing":
        return HashingEmbedder(settings.embedding_dim)
    if provider == "openai":
        return OpenAICompatibleEmbedder(settings, client=client)
    raise ConfigError(f"未知的 EMBEDDING_PROVIDER: {settings.embedding_provider!r}")

