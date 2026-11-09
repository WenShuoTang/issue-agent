"""评测执行器：``python -m issue_agent.eval.runner --offline``

两种模式：
- ``--offline``：离线、确定性、无网络。验证的是**流水线正确性**（检索召回、引用有效、
  预算生效、护栏触发），不是模型效果。
- ``--live``：真实 GitHub + 真实 LLM。用于「模型 + Prompt + 检索」的端到端效果，
  必须人工标注 ``golden_live.jsonl`` 后才能算召回。

输出：控制台表格 + ``data/eval/latest.json``（前端可读）+ 时间戳存档。
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from ..agent.runner import AgentRunner
from ..config import Settings, load_settings
from ..llm.fake import HeuristicLLM
from ..testing import FakeGitHub, make_settings
from .dataset import clone_store, offline_cases
from .metrics import aggregate, evaluate_case


def run_offline(*, settings: Settings | None = None, limit: int | None = None,
                min_recall: float = 0.5, min_citation: float = 0.9,
                emit=print) -> dict:
    settings = settings or load_settings()
    cases = offline_cases()
    if limit:
        cases = cases[:limit]
    results: list[dict] = []
    for case in cases:
        with tempfile.TemporaryDirectory(prefix="eval-") as folder:
            case_settings = make_settings(folder, max_llm_calls=14, max_rounds=3,
                                          max_steps=8, wall_clock_budget_s=30.0,
                                          trace_dir=Path(folder) / "traces")
            runner = AgentRunner(case_settings, llm=HeuristicLLM(),
                                 github=FakeGitHub(clone_store(case)))
            try:
                result = runner.run(case.repo, case.issue_number, tenant="eval")
            finally:
                runner.close()
        record = evaluate_case(case, result)
        results.append(record)
        emit(f"  {record['case_id']:<14} status={record['status']:<9} "
             f"recall@5={_fmt(record['recall_at_5'])} "
             f"cited={record['cited_count']:<2} "
             f"ms={record['duration_ms']:<5} ranked={record['ranked']}")

    summary = aggregate(results, min_recall=min_recall, min_citation=min_citation)
    report = {
        "mode": "offline",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "HeuristicLLM（确定性规则，不代表真实模型效果）",
        "dataset": {"name": "synthetic-offline", "cases": len(results),
                    "note": "合成数据，仅验证流水线；真实效果请用 --live"},
        "cases": results,
        "aggregate": summary,
        "passed": summary["gate"]["passed"],
    }
    _write_report(settings, report)
    return report


def run_live(*, settings: Settings | None = None, dataset_path: Path | None = None,
             emit=print) -> dict:
    settings = settings or load_settings()
    if not settings.llm_api_key:
        raise SystemExit("live 评测需要 DEEPSEEK_API_KEY；离线验证请用 --offline")
    path = dataset_path or (Path(__file__).parent / "golden_live.jsonl")
    if not path.exists():
        raise SystemExit(f"未找到标注集 {path}；请参考 docs/EVALUATION.md 的标注流程创建")
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    results: list[dict] = []
    runner = AgentRunner(settings)
    try:
        for case in cases:
            started = time.monotonic()
            result = runner.run(case["repo"], int(case["issue_number"]),
                                user_context=case.get("user_context", ""), tenant="eval-live")
            record = {
                "case_id": case.get("case_id") or f"{case['repo']}#{case['issue_number']}",
                "repo": case["repo"],
                "issue_number": int(case["issue_number"]),
                "status": result["status"],
                "relevant": case.get("relevant_issue_numbers", []),
                "ranked": [item.get("ranked") for item in result.get("retrieval", [])],
                "citations": [c["id"] for c in result.get("citations", [])],
                "duration_ms": int((time.monotonic() - started) * 1000),
                "tokens": result.get("metrics", {}).get("tokens_total", 0),
                "warnings": result.get("warnings", []),
            }
            if record["relevant"]:
                from .metrics import recall_at_k

                flat = [number for batch in record["ranked"] for number in (batch or [])]
                record["recall_at_5"] = recall_at_k(flat, record["relevant"], 5)
            results.append(record)
            emit(f"  {record['case_id']:<24} status={record['status']:<9} "
                 f"citations={len(record['citations'])} ms={record['duration_ms']}")
    finally:
        runner.close()
    report = {
        "mode": "live",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": settings.llm_model,
        "cases": results,
        "aggregate": {"cases": len(results),
                      "labeled": sum(1 for case in results if case.get("relevant")),
                      "avg_tokens": round(sum(case["tokens"] for case in results) / max(1, len(results)), 1)},
        "passed": all(case["status"] != "failed" for case in results),
    }
    _write_report(settings, report)
    return report


def _fmt(value) -> str:
    return "—" if value is None else f"{value:.2f}"


def _write_report(settings: Settings, report: dict) -> Path:
    folder = settings.data_dir / "eval"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive = folder / f"eval-{report['mode']}-{stamp}.json"
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    archive.write_text(payload, encoding="utf-8")
    (folder / "latest.json").write_text(payload, encoding="utf-8")
    return archive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Issue Agent 评测")
    parser.add_argument("--offline", action="store_true", help="离线确定性评测（推荐先跑）")
    parser.add_argument("--live", action="store_true", help="真实 API 端到端评测")
    parser.add_argument("--dataset", type=Path, default=None, help="live 模式标注集路径")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-recall", type=float, default=0.5)
    parser.add_argument("--min-citation", type=float, default=0.9)
    parser.add_argument("--json", action="store_true", help="输出完整 JSON")
    args = parser.parse_args(argv)
    if not args.offline and not args.live:
        args.offline = True
    print("== Issue Agent 评测 ==")
    if args.offline:
        print("[offline] 合成数据 · 确定性 LLM · 验证流水线")
        report = run_offline(limit=args.limit, min_recall=args.min_recall,
                             min_citation=args.min_citation)
    else:
        print("[live] 真实 GitHub + 真实模型")
        report = run_live(dataset_path=args.dataset)
    summary = report["aggregate"]
    print("\n== 汇总 ==")
    print(f"  样本 {summary['cases']}｜失败 {summary.get('failed', 0)}｜通过率 {summary.get('pass_rate', '—')}")
    print(f"  Recall@5 {summary.get('recall_at_5_mean')}｜MRR {summary.get('mrr_mean')}")
    print(f"  引用有效率 {summary.get('citation_validity_mean')}｜"
          f"引用覆盖率 {summary.get('citation_coverage_mean')}")
    print(f"  P50 {summary.get('latency_p50_ms')}ms｜P95 {summary.get('latency_p95_ms')}ms")
    print(f"  门禁：{'通过' if report['passed'] else '未通过'}")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
