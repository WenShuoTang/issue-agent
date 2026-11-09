"""混合检索：BM25（稀疏） + 哈希向量（稠密） + RRF 融合 + 轻量重排。

关于「向量」的诚实说明：
生产环境应该用真实 Embedding 模型（bge / text-embedding-3 等）。本项目默认用
**确定性哈希向量**，原因有两个：① 离线可复现、无外部依赖、零成本；② 简历项目里
「假装有语义检索」比「说清楚边界」更减分。哈希向量能捕捉词形/子词相似，但抓不住
真正的同义改写——这一点在 docs/EVALUATION.md 里用实验对比说明。

检索链路：chunk → BM25 排名 + 余弦排名 → RRF 融合 → 加权重排 → 文档级去重。
每个结果都带分项得分，便于解释「为什么这条证据排在前面」。
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .chunk import Chunk, chunk_document

_TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """英文按词，中文按「字 + 相邻二字」切分（无需分词器，召回更稳）。"""
    lowered = (text or "").lower()
    tokens = _TOKEN_RE.findall(lowered)
    dense: list[str] = []
    cjk_run: list[str] = []
    for token in tokens:
        if len(token) == 1 and "\u4e00" <= token <= "\u9fff":
            cjk_run.append(token)
            continue
        if cjk_run:
            dense.extend(cjk_run)
            dense.extend(a + b for a, b in zip(cjk_run, cjk_run[1:]))
            cjk_run = []
        dense.append(token)
        # 标识符拆分：NPE、IndexError、get_issue 这类词对检索很关键。
        for part in re.split(r"[_\-.]", token):
            if len(part) > 2 and part != token:
                dense.append(part)
    if cjk_run:
        dense.extend(cjk_run)
        dense.extend(a + b for a, b in zip(cjk_run, cjk_run[1:]))
    return dense


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._tokens: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._df: Counter[str] = Counter()
        self._avg_len = 1.0

    def fit(self, docs: list[str]) -> "BM25":
        self._tokens = [Counter(tokenize(doc)) for doc in docs]
        self._lengths = [sum(t.values()) for t in self._tokens]
        self._df = Counter()
        for tokens in self._tokens:
            self._df.update(tokens.keys())
        self._avg_len = max(1.0, sum(self._lengths) / max(1, len(self._lengths)))
        return self

    def scores(self, query: str) -> list[float]:
        q_terms = set(tokenize(query))
        total = len(self._tokens)
        result: list[float] = []
        for index, tokens in enumerate(self._tokens):
            score = 0.0
            length = self._lengths[index]
            for term in q_terms:
                tf = tokens.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (total - self._df[term] + 0.5) / (self._df[term] + 0.5))
                score += idf * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * length / self._avg_len))
            result.append(score)
        return result


class HashingEmbedder:
    """签名哈希向量：token 与字符三元组哈希到固定维度，L2 归一化。"""

    def __init__(self, dim: int = 512):
        self.dim = max(64, dim)
        self._cache: dict[str, list[float]] = {}

    def _hash(self, token: str) -> int:
        return int.from_bytes(hashlib.md5(token.encode("utf-8")).digest()[:8], "big") % self.dim

    def embed(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        vector = [0.0] * self.dim
        counts = Counter(tokenize(text))
        for token, count in counts.items():
            weight = 1.0 + math.log(count)
            vector[self._hash(token)] += weight
            # 字符三元组：缓解拼写差异与驼峰拆词问题。
            for i in range(len(token) - 2):
                vector[self._hash(token[i:i + 3])] += 0.35 * weight
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        vector = [v / norm for v in vector]
        if len(self._cache) < 4096:
            self._cache[text] = vector
        return vector

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    bm25: float = 0.0
    dense: float = 0.0
    rrf: float = 0.0
    title_boost: float = 0.0

    def to_dict(self) -> dict:
        payload = self.chunk.to_dict()
        payload.update({
            "score": round(self.score, 4),
            "bm25": round(self.bm25, 4),
            "dense": round(self.dense, 4),
            "rrf": round(self.rrf, 4),
            "title_boost": round(self.title_boost, 4),
        })
        return payload


@dataclass
class HybridIndex:
    """一次运行内的证据索引；语料规模通常 < 100 个 chunk，纯内存足够。"""

    embedder: HashingEmbedder = field(default_factory=lambda: HashingEmbedder(512))
    k1: float = 1.5
    b: float = 0.75
    rrf_k: int = 60
    chunks: list[Chunk] = field(default_factory=list)
    _bm25: BM25 = field(default_factory=BM25)
    _vectors: list[list[float]] = field(default_factory=list)
    _by_doc: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))

    def add_document(self, doc_id: str, title: str, text: str, url: str, kind: str,
                     **kwargs) -> list[Chunk]:
        new_chunks = chunk_document(doc_id, title, text, url, kind, **kwargs)
        for chunk in new_chunks:
            self.add_chunk(chunk)
        return new_chunks

    def add_chunk(self, chunk: Chunk) -> None:
        self.chunks.append(chunk)
        self._by_doc[chunk.doc_id].append(len(self.chunks) - 1)
        self._rebuild()

    def add_chunks(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self.chunks.append(chunk)
            self._by_doc[chunk.doc_id].append(len(self.chunks) - 1)
        self._rebuild()

    def _rebuild(self) -> None:
        corpus = [f"{chunk.title}\n{chunk.title}\n{chunk.text}" for chunk in self.chunks]
        self._bm25.fit(corpus)
        self._vectors = [self.embedder.embed(text) for text in corpus]

    @property
    def size(self) -> int:
        return len(self.chunks)

    def search(self, query: str, *, k: int = 8, per_doc: bool = True) -> list[ScoredChunk]:
        if not self.chunks or not query.strip():
            return []
        bm25_scores = self._bm25.scores(query)
        query_vector = self.embedder.embed(query)
        dense_scores = [self.embedder.cosine(query_vector, vector) for vector in self._vectors]

        bm25_rank = _ranks(bm25_scores)
        dense_rank = _ranks(dense_scores)
        bm25_max = max(bm25_scores) or 1.0
        dense_max = max(dense_scores) or 1.0
        query_tokens = set(tokenize(query))

        scored: list[ScoredChunk] = []
        for index, chunk in enumerate(self.chunks):
            rrf = (1.0 / (self.rrf_k + bm25_rank[index])
                   + 1.0 / (self.rrf_k + dense_rank[index]))
            title_tokens = set(tokenize(chunk.title))
            title_boost = 0.08 if query_tokens & title_tokens else 0.0
            # 融合权重来自离线评测调参（见 docs/EVALUATION.md），不是拍脑袋。
            score = (0.50 * (rrf / 0.033)
                     + 0.25 * (bm25_scores[index] / bm25_max)
                     + 0.25 * (dense_scores[index] / dense_max)
                     + title_boost)
            scored.append(ScoredChunk(chunk=chunk, score=score,
                                      bm25=bm25_scores[index], dense=dense_scores[index],
                                      rrf=rrf, title_boost=title_boost))
        scored.sort(key=lambda item: item.score, reverse=True)
        if not per_doc:
            return scored[:k]
        return _dedupe_by_doc(scored, k)


def _ranks(scores: list[float]) -> list[int]:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranks = [0] * len(scores)
    for position, index in enumerate(order, start=1):
        ranks[index] = position
    return ranks


def _dedupe_by_doc(scored: list[ScoredChunk], k: int) -> list[ScoredChunk]:
    """同一文档的多个 chunk 只保留最高分那条，避免上下文被同一份 Issue 塞满。"""
    best: dict[str, ScoredChunk] = {}
    for item in scored:
        doc_id = item.chunk.doc_id
        if doc_id not in best or item.score > best[doc_id].score:
            best[doc_id] = item
    ranked = sorted(best.values(), key=lambda item: item.score, reverse=True)
    return ranked[:k]
