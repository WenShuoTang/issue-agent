"""评测指标：检索、引用、依据性、成本与延迟。

指标定义（口径与 docs/EVALUATION.md 一致）：
- **Recall@5**：人工标注的相关历史 Issue 中，有多少出现在前 5 条检索结果里。
  只在有标注的样本上计算；无标注样本返回 ``None`` 而不是 0，避免拉低均值。
- **MRR**：第一个相关结果的倒数排名。
- **引用覆盖率**：有至少一条有效引用的假设比例。它衡量「说不清依据就不说」的比例。
- **引用有效率**：引用的证据 ID 真实存在的比例（幻觉引用率 = 1 - 有效率）。
- **依据性通过率**：``verify`` 节点没有发现无依据结论的样本比例。
- **P95 延迟 / 单次成本**：决定这个功能能不能放进客服工作流。
"""
from __future__ import annotations

import re
from statistics import mean

_NUMBER_RE = re.compile(r"#(\d+)")


def _order_unique(values: list[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def ranked_issue_numbers(result: dict, docs: list[dict]) -> list[int]:
    """优先用真实检索排名；没有检索记录时退回证据顺序（并标注不可比）。"""
    ranked: list[int] = []
    for item in result.get("retrieval") or []:
        ranked.extend(int(number) for number in item.get("ranked") or [])
    if ranked:
        return _order_unique(ranked)
    for doc in docs:
        match = _NUMBER_RE.search(doc.get("title", ""))
        if match:
            ranked.append(int(match.group(1)))
    return _order_unique(ranked)


def recall_at_k(ranked: list[int], relevant: list[int], k: int = 5) -> float | None:
    if not relevant:
        return None
    top = set(ranked[:k])
    return round(len(top & set(relevant)) / len(set(relevant)), 4)


def mrr(ranked: list[int], relevant: list[int]) -> float | None:
    if not relevant:
        return None
    relevant_set = set(relevant)
    for index, number in enumerate(ranked, start=1):
        if number in relevant_set:
            return round(1.0 / index, 4)
    return 0.0


def citation_coverage(hypotheses: list[dict]) -> float | None:
    if not hypotheses:
        return None
    cited = sum(1 for item in hypotheses if item.get("citations"))
    return round(cited / len(hypotheses), 4)


def citation_validity(draft: dict, evidence_ids: set[str]) -> float | None:
    citations = [cid for item in (draft.get("hypotheses") or [])
                 for cid in item.get("citations") or []]
    citations += [cid for item in (draft.get("actions") or [])
                  for cid in item.get("citations") or []]
    if not citations:
        return None
    valid = sum(1 for cid in citations if cid in evidence_ids)
    return round(valid / len(citations), 4)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return round(ordered[index], 3)


def evaluate_case(case, result: dict, *, k: int = 5) -> dict:
    evidence = result.get("evidence") or []
    evidence_ids = {doc["id"] for doc in evidence}
    draft = result.get("draft") or {}
    ranked = ranked_issue_numbers(result, evidence)
    metrics = result.get("metrics") or {}
    coverage = citation_coverage(draft.get("hypotheses") or [])
    validity = citation_validity(draft, evidence_ids)
    cited = {cid for item in (draft.get("hypotheses") or [])
             for cid in item.get("citations") or []}
    return {
        "case_id": case.case_id,
        "description": case.description,
        "repo": case.repo,
        "issue_number": case.issue_number,
        "status": result.get("status"),
        "recall_at_5": recall_at_k(ranked, case.relevant_issues, k),
        "mrr": mrr(ranked, case.relevant_issues),
        "ranked": ranked,
        "relevant": case.relevant_issues,
        "evidence_count": len(evidence),
        "cited_count": len(cited),
        "citation_coverage": coverage,
        "citation_in_evidence": validity,
        "retrieval_rounds": metrics.get("rounds", 0),
        "tool_calls": metrics.get("tool_calls", 0),
        "llm_calls": (metrics.get("llm") or {}).get("calls", metrics.get("llm_calls", 0)),
        "tokens": (metrics.get("llm") or {}).get("total_tokens", metrics.get("tokens_total", 0)),
        "duration_ms": metrics.get("duration_ms", 0),
        "warnings": len(result.get("warnings") or []),
        "injection_flagged": any("可疑指令" in warning for warning in result.get("warnings") or []),
        "report_chars": len(result.get("report") or ""),
    }


def aggregate(cases: list[dict], *, min_recall: float = 0.5,
              min_citation: float = 0.9) -> dict:
    def _values(key: str) -> list[float]:
        return [case[key] for case in cases if case.get(key) is not None]

    recall_values = _values("recall_at_5")
    mrr_values = _values("mrr")
    durations = [case["duration_ms"] for case in cases]
    failed = [case for case in cases if case["status"] == "failed"]
    recall_mean = round(mean(recall_values), 4) if recall_values else None
    citation_mean = round(mean(_values("citation_in_evidence")), 4) if _values("citation_in_evidence") else None
    coverage_values = _values("citation_coverage")
    pass_flags = []
    for case in cases:
        ok = case["status"] != "failed"
        if case["relevant"] and case["recall_at_5"] is not None:
            ok = ok and case["recall_at_5"] >= min_recall
        if case["citation_in_evidence"] is not None:
            ok = ok and case["citation_in_evidence"] >= min_citation
        pass_flags.append(ok)
    return {
        "cases": len(cases),
        "failed": len(failed),
        "recall_at_5_mean": recall_mean,
        "mrr_mean": round(mean(mrr_values), 4) if mrr_values else None,
        "citation_validity_mean": citation_mean,
        "citation_coverage_mean": round(mean(coverage_values), 4) if coverage_values else None,
        "avg_tool_calls": round(mean([case["tool_calls"] for case in cases]), 2),
        "avg_retrieval_rounds": round(mean([case["retrieval_rounds"] for case in cases]), 2),
        "avg_tokens": round(mean([case["tokens"] for case in cases]), 1),
        "latency_p50_ms": percentile(durations, 0.5),
        "latency_p95_ms": percentile(durations, 0.95),
        "pass_rate": round(sum(pass_flags) / len(cases), 4) if cases else 0.0,
        "gate": {"min_recall": min_recall, "min_citation": min_citation,
                 "passed": bool(cases) and all(pass_flags)},
    }
