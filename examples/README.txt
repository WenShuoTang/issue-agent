可用于人工验证的公开案例：

Repo: idosal/git-mcp
Issue: 214
场景：MCP 工具能调用但历史文档检索为空。

说明：
- 这是公开 Issue 引用，不保证任何时刻都能读取（仓库可能改名、Issue 可能关闭、网络可能不通）。
- 未配置 DEEPSEEK_API_KEY / GITHUB_TOKEN 时，可先跑离线测试与离线评测：

    python -m unittest discover -s tests -t .
    python -m issue_agent.eval.runner --offline

- 想复现「真实效果」的评测，请按 docs/EVALUATION.md 的流程自建标注集，
  使用 `python -m issue_agent.eval.runner --live --dataset <你的标注集>`。

sample-report.md 是用**合成数据**（FakeGitHub + HeuristicLLM）生成的报告示例，
只用于展示交付形态，不包含任何真实用户、真实仓库或真实 Issue 内容。
