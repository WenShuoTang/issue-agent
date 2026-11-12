"""检索对照实验：BM25-only vs 混合检索 vs 加标题加权。

用法：
    python -m issue_agent.eval.retrieval_bench
    python -m issue_agent.eval.retrieval_bench --embedding openai

目的：给 ``HybridIndex`` 的默认权重一个可复现的对照，而不是只写"调参过"。
注意：离线 fixture 只有十几条文档，结果只能说明相对趋势，不能当作线上召回率。
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from ..config import Settings, load_settings
from ..rag.embed import build_embedder
from ..rag.index import HashingEmbedder, HybridIndex
from .dataset import EvalCase, offline_cases
from .metrics import mrr, recall_at_k

MODES: dict[str, dict] = {
    "bm25": dict(rrf_weight=0.0, bm25_weight=1.0, dense_weight=0.0, title_boost_weight=0.0),
    "hybrid": dict(rrf_weight=0.50, bm25_weight=0.25, dense_weight=0.25, title_boost_weight=0.0),
    "hybrid+title": dict(rrf_weight=0.50, bm25_weight=0.25, dense_weight=0.25,
                         title_boost_weight=0.08),
}


def build_corpus(cases: list[EvalCase]) -> list[dict]:
    """把所有 fixture 中的 Issue / PR 合成一个候选池，制造真实一点的干扰项。"""
    seen: set[int] = set()
    docs: list[dict] = []
    for case in cases:
        store = case.store
        items = [store.get("issue", {})]
        for bucket in ("search", "pull_requests"):
            value = store.get(bucket) or {}
            if isinstance(value, dict):
                for group in value.values():
                    items.extend(group or [])
            elif isinstance(value, list):
                items.extend(value)
        for item in items:
            number = item.get("number")
            if not number or number in seen:
                continue
            seen.add(int(number))
            docs.append({"number": int(number), "title": item.get("title", ""),
                         "body": item.get("body", "")})
    return docs


def run_mode(case: EvalCase, corpus: list[dict], mode: dict, embedder, k: int = 5) -> dict:
    index = HybridIndex(embedder=embedder, **mode)
    for doc in corpus:
        index.add_document(str(doc["number"]), doc["title"], doc["body"], "", "issue")
    query = f"{case.store.get('issue', {}).get('title', '')} " \
            f"{case.store.get('issue', {}).get('body', '')[:400]}"
    results = index.search(query, k=k, per_doc=True)
    ranked = [int(item.chunk.doc_id) for item in results]
    return {"ranked": ranked,
            "recall_at_5": recall_at_k(ranked, case.relevant_issues, k),
            "mrr": mrr(ranked, case.relevant_issues)}


def run(*, settings: Settings | None = None, embedding: str | None = None,
        emit=print) -> dict:
    settings = settings or load_settings()
    if embedding:
        settings = settings.with_overrides(embedding_provider=embedding)
    embedder = build_embedder(settings)
    cases = [case for case in offline_cases() if case.relevant_issues]
    corpus = build_corpus(offline_cases())

    results: dict[str, dict] = {}
    for name, weights in MODES.items():
        mode_embedder = (HashingEmbedder(settings.embedding_dim)
                         if isinstance(embedder, HashingEmbedder) else embedder)
        per_case = []
        for case in cases:
            started = time.monotonic()
            scored = run_mode(case, corpus, weights, mode_embedder)
            scored["case_id"] = case.case_id
            scored["duration_ms"] = int((time.monotonic() - started) * 1000)
            per_case.append(scored)
        recalls = [item["recall_at_5"] for item in per_case if item["recall_at_5"] is not None]
        mrrs = [item["mrr"] for item in per_case if item["mrr"] is not None]
        results[name] = {
            "cases": per_case,
            "recall_at_5_mean": round(sum(recalls) / len(recalls), 4) if recalls else None,
            "mrr_mean": round(sum(mrrs) / len(mrrs), 4) if mrrs else None,
            "avg_ms": round(sum(item["duration_ms"] for item in per_case) / len(per_case), 2),
        }
        emit(f"  {name:<14} Recall@5={results[name]['recall_at_5_mean']} "
             f"MRR={results[name]['mrr_mean']} avg={results[name]['avg_ms']}ms")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embedding": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "corpus_docs": len(corpus),
        "cases": len(cases),
        "note": "离线 fixture 规模很小，仅用于相对趋势；不能当作线上召回率。",
        "modes": results,
    }
    folder = settings.data_dir / "eval"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "retrieval-bench.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检索模式对照实验")
    parser.add_argument("--embedding", choices=["hashing", "openai"], default=None)
    args = parser.parse_args(argv)
    print("== 检索对照（BM25 vs hybrid） ==")
    report = run(embedding=args.embedding)
    print(f"\n文档数 {report['corpus_docs']}｜样本 {report['cases']}｜"
          f"embedding={report['embedding']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
