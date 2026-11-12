"""真实 Issue 批量评测：成功率、降级率、Token、延迟与召回。

前置：``issue_agent/eval/golden_live.jsonl`` 里的样本需要人工标注
``relevant_issue_numbers``（可用 ``--dry-run`` 先检查完成度）。

用法：
    python -m issue_agent.eval.live_batch --dry-run
    python -m issue_agent.eval.live_batch --limit 10
    python -m issue_agent.eval.live_batch --limit 30 --dataset issue_agent/eval/golden_live.jsonl

注意：每条都会真实调用 GitHub 与 LLM，会产生费用；建议先用 --limit 小批量试跑。
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from ..agent.runner import AgentRunner
from ..config import Settings, load_settings
from .metrics import percentile, recall_at_k

DEFAULT_DATASET = Path(__file__).with_name("golden_live.jsonl")


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def run_batch(cases: list[dict], *, settings: Settings | None = None, limit: int | None = None,
              emit=print) -> dict:
    settings = settings or load_settings()
    selected = cases[:limit] if limit else cases
    if not selected:
        raise SystemExit("没有可运行的样本")
    runner = AgentRunner(settings)
    records: list[dict] = []
    try:
        for position, case in enumerate(selected, 1):
            started = time.monotonic()
            result = runner.run(case["repo"], int(case["issue_number"]),
                                tenant="eval-live")
            metrics = result.get("metrics") or {}
            ranked = [issue_number for batch in (result.get("retrieval") or [])
                      for issue_number in (batch.get("ranked") or [])]
            relevant = case.get("relevant_issue_numbers") or []
            records.append({
                "case_id": case.get("case_id", f"{case['repo']}#{case['issue_number']}"),
                "repo": case["repo"],
                "issue_number": case["issue_number"],
                "status": result["status"],
                "duration_ms": int((time.monotonic() - started) * 1000),
                "tokens": metrics.get("tokens_total", 0),
                "llm_calls": metrics.get("llm_calls")
                if metrics.get("llm_calls") is not None
                else (metrics.get("llm") or {}).get("calls", 0)
                if isinstance(metrics.get("llm"), dict) else 0,
                "rounds": metrics.get("rounds", 0),
                "citations": len(result.get("citations") or []),
                "warnings": len(result.get("warnings") or []),
                "recall_at_5": recall_at_k(ranked, relevant, 5) if relevant else None,
            })
            emit(f"  [{position}/{len(selected)}] {records[-1]['case_id']:<40} "
                 f"status={result['status']:<9} {records[-1]['duration_ms']}ms "
                 f"{records[-1]['tokens']} tokens")
    finally:
        runner.close()

    recalls = [record["recall_at_5"] for record in records
               if record["recall_at_5"] is not None]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": settings.llm_model,
        "provider": settings.llm_provider,
        "cases": len(records),
        "labeled": len(recalls),
        "success_rate": round(sum(1 for r in records if r["status"] == "succeeded") / len(records), 4)
        if records else 0.0,
        "degrade_rate": round(sum(1 for r in records if r["status"] == "partial") / len(records), 4)
        if records else 0.0,
        "failure_rate": round(sum(1 for r in records if r["status"] == "failed") / len(records), 4)
        if records else 0.0,
        "avg_tokens": round(mean([r["tokens"] for r in records]), 1) if records else 0,
        "p50_latency_ms": percentile([r["duration_ms"] for r in records], 0.5),
        "p95_latency_ms": percentile([r["duration_ms"] for r in records], 0.95),
        "recall_at_5_mean": round(mean(recalls), 4) if recalls else None,
        "records": records,
    }
    folder = settings.data_dir / "eval"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (folder / f"live-batch-{stamp}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "live-batch-latest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="真实 Issue 批量评测")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="只检查数据集，不调用 API")
    args = parser.parse_args(argv)

    if not args.dataset.exists():
        print(f"数据集不存在：{args.dataset}")
        return 2
    cases = load_cases(args.dataset)
    labeled = sum(1 for case in cases if case.get("relevant_issue_numbers"))
    print(f"样本 {len(cases)}｜已标注 {labeled}｜未标注 {len(cases) - labeled}")
    if args.dry_run:
        for case in cases:
            flag = "labeled" if case.get("relevant_issue_numbers") else "TODO"
            print(f"  {case['repo']}#{case['issue_number']}  {flag}  {case.get('title', '')[:60]}")
        return 0
    report = run_batch(cases, limit=args.limit)
    print("\n== 汇总 ==")
    print(f"  成功率 {report['success_rate']:.1%}｜降级率 {report['degrade_rate']:.1%}"
          f"｜失败率 {report['failure_rate']:.1%}")
    print(f"  平均 Token {report['avg_tokens']}｜P50 {report['p50_latency_ms']}ms"
          f"｜P95 {report['p95_latency_ms']}ms")
    print(f"  Recall@5 {report['recall_at_5_mean']}（已标注 {report['labeled']} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
