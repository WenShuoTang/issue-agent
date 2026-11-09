"""报告渲染：把结构化草稿变成可直接贴进工单/IM 的 Markdown。

为什么结构化与渲染分离：
- 校验（引用白名单、越界表述）作用在结构化数据上，比正则扫 Markdown 可靠；
- 同一份结构化结果可以渲染成 Markdown、JSON、Slack 卡片，前端不用重新解析文本；
- 渲染是确定性的，可单测，不消耗 token。
"""
from __future__ import annotations

from ..security.guard import scrub_unsafe_claims
from .state import EvidenceDoc

_CONFIDENCE_LABEL = {"high": "高", "medium": "中", "low": "低"}


def _md_text(value: object, limit: int = 700) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ")
    text = text.replace("[", "［").replace("]", "］").replace("`", "ʼ")
    return text[:limit].strip()


def _link(citation_id: str, documents: list[EvidenceDoc]) -> str | None:
    for doc in documents:
        if doc.get("id") == citation_id and doc.get("url"):
            return f"[{citation_id}]({doc['url']})"
    return None


def _links(citation_ids: list[str], documents: list[EvidenceDoc]) -> str:
    links = [link for cid in citation_ids if (link := _link(cid, documents))]
    return " ".join(dict.fromkeys(links)) if links else "（无直接证据）"


def used_citations(draft: dict, documents: list[EvidenceDoc]) -> list[dict]:
    """收集草稿真正引用到的证据，输出可直接给前端渲染的列表。"""
    allowed = {doc["id"] for doc in documents}
    used: list[str] = []
    for item in (draft.get("hypotheses") or []):
        used.extend(cid for cid in item.get("citations", []) if cid in allowed)
    for item in (draft.get("actions") or []):
        used.extend(cid for cid in item.get("citations", []) if cid in allowed)
    ordered = [doc["id"] for doc in documents if doc["id"] in set(used)]
    return [{"id": doc_id,
             "title": next(doc["title"] for doc in documents if doc["id"] == doc_id),
             "url": next(doc["url"] for doc in documents if doc["id"] == doc_id),
             "kind": next(doc.get("kind", "") for doc in documents if doc["id"] == doc_id)}
            for doc_id in ordered]


def render_markdown(*, repo: str, issue_number: int, issue_url: str, draft: dict,
                    documents: list[EvidenceDoc], warnings: list[str], gaps: list[str],
                    budget: dict, status: str, trace_summary: dict | None = None,
                    error: str = "") -> str:
    summary = _md_text(draft.get("summary"), 1200) or "现有公开证据不足以形成可靠结论。"
    lines: list[str] = [
        f"# GitHub Issue 排障报告：{repo}#{issue_number}",
        "",
        f"> 目标 Issue：{issue_url or f'https://github.com/{repo}/issues/{issue_number}'}",
        "",
        "## 结论摘要",
        "",
        summary,
        "",
        "## 可能原因（待验证假设，不是已确认根因）",
        "",
    ]

    hypotheses = draft.get("hypotheses") or []
    if not hypotheses:
        lines.append("现有证据不足，未能形成可引用的假设。")
    for index, item in enumerate(hypotheses, start=1):
        confidence = _CONFIDENCE_LABEL.get(str(item.get("confidence", "low")), "低")
        lines.append(f"{index}. **[{confidence}置信度]** {_md_text(item.get('claim'), 600)}"
                     f"  证据：{_links(item.get('citations') or [], documents)}")

    lines += ["", "## 建议排查步骤（Agent 未执行，需人工确认）", ""]
    actions = draft.get("actions") or []
    if not actions:
        lines.append("建议先补全最小复现步骤、完整错误栈和目标仓库版本。")
    for index, item in enumerate(actions, start=1):
        lines.append(f"{index}. {_md_text(item.get('step'), 700)}"
                     f"  依据：{_links(item.get('citations') or [], documents)}")

    lines += ["", "## 如何验证", ""]
    verification = draft.get("verification") or []
    lines += [f"- {_md_text(item, 400)}" for item in verification] or ["- 按上述步骤复现，确认错误是否消失。"]

    lines += ["", "## 不确定性与缺少的信息", ""]
    combined_gaps = [*gaps, *(draft.get("gaps") or [])]
    seen: set[str] = set()
    for gap in combined_gaps:
        text = _md_text(gap, 400)
        if text and text not in seen:
            seen.add(text)
            lines.append(f"- {text}")
    if not seen:
        lines.append("- 未记录到显著缺口。")

    if warnings:
        lines += ["", "## 运行提示", ""]
        for warning in dict.fromkeys(warnings):
            lines.append(f"- {_md_text(warning, 300)}")

    risk_notes = draft.get("risk_notes") or []
    if risk_notes:
        lines += ["", "## 风险提示", ""] + [f"- {_md_text(item, 300)}" for item in risk_notes]

    lines += ["", "## 证据清单（可点击原始链接）", ""]
    for doc in documents:
        risk = f"，含可疑指令（风险 {doc['risk']}）" if doc.get("risk", 0) >= 0.5 else ""
        lines.append(f"- [{doc['id']}] [{_md_text(doc['title'], 160)}]({doc['url']})"
                     f"（{doc.get('kind', 'unknown')}{risk}）")

    lines += ["", "## 运行指标", ""]
    lines.append(
        f"- 检索轮次 {budget.get('rounds', 0)}｜工具调用 {budget.get('tool_calls', 0)}"
        f"｜LLM 调用 {budget.get('llm_calls', 0)}｜Token {budget.get('tokens_total', 0)}"
        f"｜耗时 {budget.get('elapsed_seconds', 0)}s｜成本 ${budget.get('cost_usd', 0)}"
    )
    if budget.get("exhausted"):
        lines.append(f"- 触发预算降级：{', '.join(budget['exhausted'])}")
    if trace_summary:
        lines.append(f"- 执行轨迹：{trace_summary.get('span_count', 0)} 个 span，"
                     f"{trace_summary.get('error_spans', 0)} 个错误 span")
    if status != "succeeded":
        lines.append(f"- 本次运行状态：{status}（{_md_text(error, 300) or '部分能力被降级'}）")

    lines += ["", "---", "",
              "> 免责声明：本报告由 Agent 基于公开 GitHub 资料生成，属于**待验证假设**；"
              "未运行目标仓库代码，未修改任何代码或 Issue。最终结论需由工程师复现验证。"]
    return scrub_unsafe_claims("\n".join(lines))


def render_degraded_report(*, repo: str, issue_number: int, error: str, error_code: str,
                           documents: list[EvidenceDoc], warnings: list[str],
                           gaps: list[str], budget: dict) -> str:
    """预算耗尽/中途失败时的**确定性降级报告**。

    真实业务里这比"失败"重要得多：用户已经等了 40 秒、烧了 token，哪怕模型没来得及写作，
    也要把已抓到的证据、可能方向、缺什么信息交出去。所有内容都不经过模型，因此没有幻觉。
    """
    lines = [
        f"# 排障分析（降级结果）：{repo}#{issue_number}",
        "",
        "> 本次运行在生成完整结论前被中止，**没有**得出可引用根因。以下为已收集证据与人工可继续的方向。",
        "",
        "## 为什么被中止",
        "",
        f"- {_md_text(error, 500)}",
        f"- 错误码：`{error_code or 'unknown'}`｜检索轮次 {budget.get('rounds', 0)}"
        f"｜工具调用 {budget.get('tool_calls', 0)}｜LLM 调用 {budget.get('llm_calls', 0)}"
        f"｜Token {budget.get('tokens_total', 0)}｜耗时 {budget.get('elapsed_seconds', 0)}s",
        "",
        "## 已收集到的证据（可直接点开核对）",
        "",
    ]
    if documents:
        for doc in documents[:8]:
            risk = "（含可疑指令，仅作线索）" if doc.get("risk", 0) >= 0.5 else ""
            excerpt = _md_text(doc.get("body", ""), 220)
            lines.append(f"- [{doc['id']}] [{_md_text(doc['title'], 120)}]({doc['url']}){risk}")
            if excerpt:
                lines.append(f"  - 摘录：{excerpt}")
    else:
        lines.append("- 无（在读取阶段就中止了）")

    lines += ["", "## 建议的人工排查方向", "",
              "1. 先确认目标 Issue 的复现步骤与版本，和历史 Issue 中提到的版本差异逐条对照。",
              "2. 检查评论中是否出现 workaround、配置项改名或已合并 PR 的引用。",
              "3. 若仍需 Agent 结论，可按下方建议调整预算后重试（已抓证据会从检查点复用）。"]

    if gaps:
        lines += ["", "## 已识别的信息缺口", ""] + [f"- {_md_text(gap, 300)}" for gap in gaps[:6]]

    if warnings:
        lines += ["", "## 数据源提示", ""] + [f"- {_md_text(item, 300)}" for item in warnings[:6]]

    if error_code == "budget_exceeded":
        advice = ("预算类中止不是故障：调大 `MAX_LLM_CALLS` / `TOKEN_BUDGET` / `WALL_CLOCK_BUDGET_S`，"
                  "或用 `POST /api/v1/sessions/{id}/resume` 从检查点续跑。")
    elif error_code in {"github_error", "tool_error"}:
        advice = "数据源类中止：检查网络/代理与 `GITHUB_TOKEN` 额度，确认仓库与 Issue 可公开访问。"
    else:
        advice = "先运行 `python -m issue_agent.cli doctor` 检查依赖与配置，再重试。"
    lines += ["", "## 接下来怎么做", "", f"- {advice}", ""]
    lines += ["---", "",
              "> 本降级报告由确定性规则生成，未经模型改写；未运行目标仓库代码，未修改任何文件。"]
    return scrub_unsafe_claims("\n".join(lines))


def render_failure_report(*, repo: str, issue_number: int, error: str,
                          documents: list[EvidenceDoc] | None = None) -> str:
    """失败也要给出可读、可追踪的结果，而不是一个 500。"""
    lines = [
        f"# 排障分析未完成：{repo}#{issue_number}",
        "",
        "## 原因",
        "",
        f"- {_md_text(error, 500)}",
        "",
        "## 已收集到的证据",
        "",
    ]
    for doc in documents or []:
        lines.append(f"- [{doc['id']}] [{_md_text(doc['title'], 160)}]({doc['url']})")
    if not documents:
        lines.append("- 无（在读取阶段就失败了）")
    lines += ["", "## 建议", "",
              "- 检查网络/代理、`GITHUB_TOKEN` 额度与 `DEEPSEEK_API_KEY` 配置后重试。",
              "- 若是目标 Issue 不存在，请确认仓库公开且编号正确。"]
    return "\n".join(lines)
