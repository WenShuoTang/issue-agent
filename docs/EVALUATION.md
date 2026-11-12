# 评测方法与结果

评测的目标不是"证明 Agent 很聪明"，而是回答四个可被追问的问题：

1. **检索**：相关历史 Issue 能不能被找到？（Recall@k / MRR）
2. **依据**：结论有没有证据，引用是不是真的存在？（引用覆盖率 / 引用有效率）
3. **成本**：一次分析要多少钱、多少时间？（Token / 延迟 / 工具调用数）
4. **退化**：改动 Prompt 或检索后，是不是变差了？（回归门禁）

---

## 1. 两种评测，必须分清

| 模式 | 命令 | 数据 | 验证什么 | 不能证明什么 |
|---|---|---|---|---|
| 离线 | `--offline` | 合成 fixture + 规则型 LLM 替身 | 图路由、检索、引用校验、预算、护栏 | 模型效果、真实召回率 |
| 真实 | `--live` | 真实 GitHub + 真实模型 + 人工标注 | 端到端效果 | 统计显著性（样本小） |

**离线模式的价值**：确定性、零成本、可在 CI 每次提交都跑。它抓的是"代码改坏了"，
而不是"模型变笨了"。
**离线模式的陷阱**：如果拿它的数字当产品指标，会高估实际效果。README 与评测文档均已标注。

---

## 2. 指标定义

| 指标 | 定义 | 计算位置 |
|---|---|---|
| Recall@5 | 标注的相关 Issue 中，出现在前 5 条检索结果的比例 | `eval/metrics.py::recall_at_k` |
| MRR | 第一个相关结果的排名倒数（第 1 名 = 1.0） | `mrr` |
| 引用有效率 | 引用 ID 真实存在于证据集合的比例（幻觉引用率 = 1 - 该值） | `citation_validity` |
| 引用覆盖率 | 至少带 1 条引用的假设 / 全部假设 | `citation_coverage` |
| 依据性通过率 | `verify` 未发现无依据结论的样本比例 | 由 `verify` + 测试覆盖 |
| 工具效率 | 每次分析的工具调用数、检索轮数 | `evaluate_case` |
| P50 / P95 延迟 | 端到端耗时分布 | `percentile` |
| 单次 Token | 输入 + 输出 token | `LLMClient.usage` |

**为什么不用"准确率"**：排障没有唯一正确答案，只有"有依据的假设 + 可执行的验证步骤"。
用 Recall 衡量"材料找没找到"，用引用指标衡量"话有没有依据"，比自造一个 accuracy 指标更可核对。

---

## 3. 数据集

### 3.1 离线集（4 条，合成）

| case_id | 场景 | 标注的相关 Issue | 陷阱 |
|---|---|---|---|
| `timeout-001` | 升级后搜索超时 | #13, #14 | #13 是"索引过期"，与超时无关 |
| `auth-002` | Token 轮换后 401 | #31 | #32 是无关的 OAuth 讨论 |
| `config-003` | 配置键改名导致解析失败 | #41, #42 | #42 是无关的 YAML 缩进问题 |
| `no-answer-004` | 无任何相关历史 Issue | （空） | 验证"搜不到时不编造" |

数据源：`issue_agent/eval/dataset.py`（store 由 `issue_agent/testing.py` 生成，
包含一条**恶意评论**用于验证注入标记）。

### 3.2 真实标注集（需要自己建，30~50 条）

推荐流程（这也是文档里唯一能宣称"真实效果"的前提）：

1. **选样本**：从 3~5 个活跃仓库各取 8~10 个 issue，覆盖 bug / 配置 / 安装 / 性能四类；
   排除纯讨论、feature request、已关闭无复现的。
2. **标相关 Issue**：只标"能解释当前故障或提供 workaround"的历史 Issue，最多 3 条。
   不标"提到同一个词但无关"的。
3. **标根因（可选）**：对应修复 PR 或配置项；用于人工核对报告是否命中。
4. **双人标注 + 分歧仲裁**：至少 20% 样本双标，记录 Cohen's kappa；低于 0.6 说明标注定义要重写。
5. **冻结版本**：标注集入库（`golden_live.jsonl`），每行：
   ```json
   {"case_id":"git-mcp-214","repo":"idosal/git-mcp","issue_number":214,
    "relevant_issue_numbers":[190,201],
    "expected_root_cause_keywords":["index","embedding"],
    "notes":"需要在 0.2.x 版本复现"}
   ```
6. **不要用被测系统生成标注**（会用幻觉污染标签）。

---

## 4. 离线评测结果（可复现）

```bash
python -m issue_agent.eval.runner --offline
```

当前结果：

| case | status | Recall@5 | 引用数 | 检索到的编号 | 耗时 |
|---|---|---|---|---|---|
| timeout-001 | succeeded | 1.00 | 3 | 13, 14 | ~1.1s |
| auth-002 | succeeded | 1.00 | 3 | 31 | ~1.1s |
| config-003 | succeeded | 0.50 | 3 | 42 | ~1.1s |
| no-answer-004 | succeeded | —（无标注） | 2 | 99 | ~0.9s |
| **汇总** | 4/4 非失败 | **0.833** | 有效率 **1.000** | 覆盖率 1.000 | P50 ≈1.0s / P95 ≈1.2s |

> 耗时包含 SQLite 检查点写入等本地开销，不含真实模型推理；接真实模型后主要耗时在 LLM。

**怎么读这组数字**：

- `config-003` 只有 0.5，说明"配置键改名"这类查询对**关键词检索**不友好：规则型 LLM 抽出的
  检索词是 `ConfigError invalid YAML upgrade`，命中了无关的 #42 而漏掉 #41。
  这正是"BM25 抓不住同义改写"的实证——真实模型会抽到 `search.timeout` 这类精确词，
  但离线模式故意不掩盖这个弱点。
- `1.000` 的引用有效率是**结构性保证**（校验器会丢弃未知 ID），不是模型变好了。

---

## 5. 真实端到端评测

```bash
python -m issue_agent.eval.runner --live --dataset issue_agent/eval/golden_live.jsonl
```

成本提示：单条约 15k token / 40~60s / 8~12 次 LLM 调用。50 条 ≈ 75 万 token。

### 已完成的真实运行（不是基准，仅作冒烟证据）

`idosal/git-mcp#214`，真实 GitHub + DeepSeek：

| 次 | 预算 | 结果 | 耗时 | Token | 观察 |
|---|---|---|---|---|---|
| 1 | LLM 8 次 | 预算耗尽 → 降级 | 45.6s | 14,641 | 默认预算过紧 |
| 2 | LLM 14 / 轮次 3 | 轮次耗尽 → 降级 | 24.9s | 17,287 | 第 4 次检索被预算打断 |
| 3 | 修复后 | **成功**（含一次依据性重写） | 53.1s | 29,395 | 14 次 LLM、8 次工具、2 轮检索、37 span |

交付形态示例（**合成数据**，不含真实用户信息）见 `examples/sample-report.md`。

这三次运行直接产生了三个代码改进（见 `docs/PRODUCTION.md` 第 0 章）：

1. 默认 LLM 预算 8 → 16（实测 14 次刚好打满）；
2. `act` 在轮次用尽时直接收尾，并且"评估节点要求的补检索"不再额外消耗一次 LLM 决策；
3. 无草稿时输出**确定性降级报告**（证据摘录 + 人工方向），而不是一句"内部错误"。

---

## 6. 检索实验：怎么做对比

想回答"混合检索比 BM25 好多少"，用同一批标注样本跑三组配置：

| 组 | 配置 | 代码开关 |
|---|---|---|
| A. BM25 only | 仅稀疏检索 | `HybridIndex` 里把 RRF 权重设为 0（或临时改 `search`） |
| B. BM25 + 向量（RRF） | 当前默认 | `HybridIndex` |
| C. B + 标题/文档级重排 | 当前默认（含 title boost） | `HybridIndex.search` |

每组记录：Recall@5、MRR、平均检索耗时、平均证据数。**必须固定**同一个数据集与同一模型
（或同一离线替身），否则比较无效。

关于"真 Embedding"：当前默认是哈希向量（离线可复现）。不改代码，配置即可切换：

```ini
EMBEDDING_PROVIDER=openai
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=sk-...
```

然后重跑 `python -m issue_agent.eval.retrieval_bench` 做 BM25 vs hybrid 对照。

预期收益：同义改写（"超时" ↔ "timeout"、"认证失败" ↔ "401"）的召回会明显提升；
代价是每次检索一次网络调用（延迟 + 成本），需要缓存向量。这个对比**目前没有做过**，
文档不编造数字。

---

## 7. 回归门禁

`python -m issue_agent.eval.runner --offline --min-recall 0.5 --min-citation 0.9`
在门禁不通过时返回退出码 1，可直接放进 CI：

```yaml
- run: python -m pip install -r requirements.txt
- run: python -m unittest discover -s tests -t .
- run: python -m issue_agent.eval.runner --offline
```

门禁原则：**只看下限，不看均值**。均值会被好样本拉高，下限才能防住"某个场景彻底坏掉"。
报告写入 `data/eval/latest.json`，前端 `/api/v1/eval/latest` 可读。

---

## 8. 指标陷阱清单（自我审查用）

- [ ] 是否把合成数据的结果写成了产品指标？
- [ ] 是否只在 4 条样本上调 Prompt，然后宣称"优化了 X%"？
- [ ] 引用有效率 100% 是否只是"引用存在"，而不是"引用支持结论"？
- [ ] 是否漏掉了失败样本（只统计 succeeded 的 mean 会偏高）？
- [ ] 是否比较了不同模型/不同数据集的数字？
- [ ] 延迟是否包含冷启动、GitHub 限流重试？
- [ ] 成本单价是否用了真实账单，而不是猜的？
- [ ] 标注是否由被测系统自己生成（污染）？
- [ ] 是否统计了"空结果/降级"的比例，而不只是成功案例？

---

## 9. 检索模式对照（离线）

```bash
python -m issue_agent.eval.retrieval_bench
```

在 9 条合成候选文档、3 个有标注样本上跑三种配置：

| 模式 | Recall@5 | MRR | 平均耗时 |
|---|---|---|---|
| BM25-only | 1.000 | 0.833 | ~10ms |
| Hybrid（BM25 + 哈希向量 + RRF） | 1.000 | 0.833 | ~5ms |
| Hybrid + 标题加权 | 1.000 | 0.833 | ~5ms |

结论：fixture 太小，三种模式没有差异；这组数字只能验证脚本可复现，**不能**证明混合检索更优，
也不能用来解释默认权重。要得到有意义的权重依据，需要真实标注集 + 真 Embedding：

```bash
EMBEDDING_PROVIDER=openai python -m issue_agent.eval.retrieval_bench
```

代码中的默认权重（0.50/0.25/0.25 + 0.08 标题加成）因此标注为**经验值**，不是调参结论。

## 10. 真实评测集与批量跑批

`issue_agent/eval/golden_live.jsonl` 收录了 30 个公开 Issue（来自 8 个仓库），用于真实
端到端评测。当前状态：**30 条均待人工标注** `relevant_issue_numbers`（字段
`needs_labeling: true`）。

```bash
python -m issue_agent.eval.live_batch --dry-run        # 检查数据集，不调用 API
python -m issue_agent.eval.live_batch --limit 10       # 小批量真实跑
```

跑批输出每个样本的状态、耗时、Token、引用数与 Recall@5，并汇总成功率、降级率、失败率、
均值 Token、P50/P95 延迟。真实跑批会产生 API 费用，仓库不预置跑批结果；简历中的
成本/稳定性数字必须来自你自己的实际跑批。

## 11. Badcase 记录

| # | Badcase | 根因 | 处理 | 回归 |
|---|---|---|---|---|
| 1 | 真实 Issue `idosal/git-mcp#214`：8 次 LLM 预算耗尽，输出降级报告 | 默认预算过紧 + 轮次耗尽后仍尝试检索 | 默认预算 8→16；轮次用尽直接收尾；无草稿时输出确定性降级报告 | `test_budget_exceeded_returns_partial`、`test_resume_after_budget_failure_completes` |
| 2 | `config-003` 离线 Recall@5=0.5（配置键改名类查询漏召回） | 哈希向量抓不住同义改写 + 规则型替身抽词差 | 保留为已知限制；提供 `retrieval_bench` 与 `EMBEDDING_PROVIDER=openai` 做对照 | `test_retrieval_benchmark` |

