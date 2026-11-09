# 贡献指南

感谢你愿意改进这个项目。提交前请先读完下面几条，能省掉大部分来回沟通。

## 开发环境

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # 填入自己的 DEEPSEEK_API_KEY / GITHUB_TOKEN
python -m issue_agent.cli doctor
```

Python >= 3.10。项目不依赖 Docker / Redis / 向量数据库。

## 提交前必须跑的命令

```bash
python scripts/preflight.py                    # 隐私与密钥扫描（必须通过）
python -m compileall -q issue_agent
python -m unittest discover -s tests -t .      # 全部离线，不需要 Key/网络
python -m issue_agent.eval.runner --offline    # 评测门禁
```

CI 会执行同样的步骤（见 `.github/workflows/ci.yml`）。

## 代码约定

- **新增能力必须带离线测试**：用 `issue_agent/testing.py` 的 `FakeGitHub` 和
  `issue_agent/llm/fake.py` 的 `HeuristicLLM` / `ScriptedLLM`，不要依赖真实 API。
- **错误用 `AgentError` 子类**：带稳定 `code` 和 `http_status`，不要直接抛裸 `Exception`。
- **工具必须只读**：新增 GitHub 工具只能是 GET；写操作请在 `security/policy.py` 里显式拒绝。
- **Prompt 改动会影响结果缓存**：`prompts.py` 里的指纹会自动更新，无需手动处理，
  但请同步更新 `docs/` 里对应的 Prompt 说明。
- **不要把本地路径、真实 Key、真实用户数据写进代码或测试**。

## 提交 Pull Request

1. Fork 并新建分支：`feat/short-name` 或 `fix/short-name`。
2. 一个 PR 只做一件事；说明**问题 → 方案 → 如何验证**。
3. 如果改了行为，请同步更新 README / docs 中受影响的段落。
4. `preflight.py`、单测与离线评测均需通过。

## 报告问题

- 功能/Bug：用 Issue 模板，附最小复现步骤与 `python -m issue_agent.cli doctor` 输出（**先脱敏**）。
- 安全问题：不要开公开 Issue，请走 [SECURITY.md](SECURITY.md) 的私下报告渠道。
- 隐私问题：见 [SECURITY.md](SECURITY.md) 与 README 的「隐私与数据」一节。

## 许可证

提交贡献即表示你同意以 [MIT License](LICENSE) 授权你的贡献。
