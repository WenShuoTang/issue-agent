"""命令行入口：analyze / serve / eval / doctor。

    python -m issue_agent.cli analyze --repo idosal/git-mcp --issue 214 --stream
    python -m issue_agent.cli serve --port 8000
    python -m issue_agent.cli eval --offline
    python -m issue_agent.cli doctor
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_settings
from .errors import AgentError


def _cmd_analyze(args: argparse.Namespace) -> int:
    from .agent.runner import AgentRunner

    settings = load_settings()
    runner = AgentRunner(settings)
    try:
        if args.stream:
            final: dict = {}
            for event in runner.stream(args.repo, args.issue, user_context=args.context,
                                       tenant=args.tenant):
                if event["event"] == "span":
                    print(f"  [{event['kind']:<6}] {event['name']:<28} "
                          f"{event['duration_ms']:>5}ms {event['status']}")
                elif event["event"] == "error":
                    print(f"  错误：{event['error']}", file=sys.stderr)
                elif event["event"] == "final":
                    final = event.get("result") or {}
            result = final
        else:
            result = runner.run(args.repo, args.issue, user_context=args.context,
                                tenant=args.tenant)
    except AgentError as exc:
        print(f"失败（{exc.code}）：{exc.public_message}", file=sys.stderr)
        return 2
    finally:
        runner.close()

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result.get("report", ""))
        print("\n--- metrics ---")
        print(json.dumps({k: result["metrics"].get(k) for k in
                          ("rounds", "tool_calls", "tokens_total",
                           "cost_usd", "elapsed_seconds")}, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(result.get("report", ""), encoding="utf-8")
        print(f"\n报告已写入 {Path(args.out).resolve()}", file=sys.stderr)
    return 0 if result.get("status") != "failed" else 1


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api.app import create_app

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .eval.runner import main as eval_main

    argv = ["--live"] if args.live else ["--offline"]
    if args.limit:
        argv += ["--limit", str(args.limit)]
    return eval_main(argv)


def _cmd_doctor(args: argparse.Namespace) -> int:
    import platform

    from .memory.store import MemoryStore
    from .providers import get_preset

    settings = load_settings()
    checks: list[tuple[str, bool, str]] = []
    try:
        import fastapi  # noqa: F401
        import jsonschema  # noqa: F401
        import langgraph  # noqa: F401
        import openai  # noqa: F401

        checks.append(("依赖", True, "langgraph/fastapi/openai/jsonschema 可用"))
    except ImportError as exc:
        checks.append(("依赖", False, f"缺少依赖：{exc}（pip install -r requirements.txt）"))
    checks.append(("Python", sys.version_info >= (3, 10),
                   f"{platform.python_version()}（需要 >= 3.10）"))
    provider = get_preset(settings.llm_provider)
    key_ok = bool(settings.llm_api_key) or not settings.llm_requires_key
    if key_ok:
        key_detail = ("已配置" if settings.llm_api_key else "本地服务无需 Key")
    else:
        key_detail = (f"未配置 {settings.llm_key_env}（申请步骤见 docs/PROVIDERS.md）")
    checks.append(("LLM", key_ok,
                   f"{provider.label}｜{settings.llm_model}｜{settings.llm_base_url}｜"
                   f"{settings.llm_kind}｜{key_detail}"))
    checks.append(("GitHub Token", bool(settings.github_token),
                   "已配置（5,000 次/小时）" if settings.github_token
                   else "未配置（匿名 60 次/小时；申请步骤见 docs/PROVIDERS.md）"))
    try:
        store = MemoryStore(settings.db_path)
        stats = store.stats()
        checks.append(("数据库", True, f"{stats['sessions']} 个会话，{stats['episodes']} 条记忆"))
    except Exception as exc:  # noqa: BLE001
        checks.append(("数据库", False, str(exc)[:120]))
    print("== 环境自检 ==")
    for name, ok, detail in checks:
        print(f"  [{'OK ' if ok else 'FAIL'}] {name:<12} {detail}")
    print("\n== 生效配置（脱敏）==")
    print(json.dumps(settings.describe(), ensure_ascii=False, indent=2))
    return 0 if all(ok for _, ok, _ in checks) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="issue_agent", description="GitHub Issue 排障 Agent")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="分析一个 GitHub Issue")
    analyze.add_argument("--repo", required=True, help="owner/repo")
    analyze.add_argument("--issue", required=True, type=int)
    analyze.add_argument("--context", default="", help="本机环境/版本等补充信息")
    analyze.add_argument("--tenant", default="local")
    analyze.add_argument("--out", type=Path, default=None, help="报告写入文件")
    analyze.add_argument("--json", action="store_true", help="输出完整 JSON")
    analyze.add_argument("--stream", action="store_true", help="打印执行轨迹")

    serve = sub.add_parser("serve", help="启动 Web 服务")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--log-level", default="info")

    evaluate = sub.add_parser("eval", help="运行评测")
    evaluate.add_argument("--offline", action="store_true")
    evaluate.add_argument("--live", action="store_true")
    evaluate.add_argument("--limit", type=int, default=None)

    sub.add_parser("doctor", help="环境自检")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {"analyze": _cmd_analyze, "serve": _cmd_serve, "eval": _cmd_eval,
                "doctor": _cmd_doctor}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
