## 这个 PR 做了什么

<!-- 一句话说明；关联 Issue：Closes #123 -->

## 为什么这么做

<!-- 问题背景、取舍 -->

## 如何验证

- [ ] `python scripts/preflight.py` 通过
- [ ] `python -m unittest discover -s tests -t .` 通过
- [ ] `python -m issue_agent.eval.runner --offline` 通过
- [ ] 手动验证（如有）：命令/截图/请求样例

## 检查项

- [ ] 未提交 `.env`、`data/`、trace、真实密钥或他人隐私数据
- [ ] 新增能力有离线测试（FakeGitHub / HeuristicLLM）
- [ ] 新增工具为只读，并在 `security/policy.py` 中受限
- [ ] 如改动 Prompt，已同步更新文档
- [ ] 如改动行为，已更新 README / docs
