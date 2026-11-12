# 架构设计

内容包括系统运行方式、关键设计取舍与已知妥协。
面向接手代码的开发者与部署者。

---

## 1. 设计目标与硬约束

| 目标 | 约束 | 结果 |
|---|---|---|
| 可离线验证 | 无 Key、无网络也能跑测试与评测 | `FakeGitHub` + `HeuristicLLM` + 合成数据 |
| 零外部基础设施 | 单机、无 Docker/Redis/向量库 | SQLite 承担缓存、记忆、检查点 |
| 只读、可审计 | 不产生任何写操作 | 8 个只读工具 + 策略层拒绝写工具 |
| 成本可控 | 一次分析有明确上限 | 六维预算 + 超限降级 |
| 结论可信 | 不许编造、不许邀功 | 引用白名单 + 越界表述改写 + 免责声明 |
| 失败可用 | 任何异常都要有可读输出 | 单一 `degrade` 出口 + 三种降级报告 |

---

## 2. 信任模型与数据流

系统的核心假设：**GitHub 上的任何文本都是不可信数据**。Issue 正文、评论、README、
历史 Issue、PR 描述都可能包含针对 LLM 的指令（提示注入），甚至伪造的"系统消息"。

处理链路：

```
GitHub API ──► sanitize_untrusted() ──► 去掉控制字符/零宽/HTML 注释
                    │
                    ├─► scan_injection() 打分（0~1）+ 标签，写入证据 risk
                    │
                    ▼
             evidence doc（E1..EN，含 URL、kind、risk）
                    │
                    ▼
   render_evidence()：每条证据用本次运行随机 nonce 包裹
        <<<UNTRUSTED:8f3a1c:E3>>> ... <<<END_UNTRUSTED:8f3a1c:E3>>>
                    │
                    ▼
               LLM（system prompt 声明：边界内一律是数据）
                    │
                    ▼
   Pydantic 校验 + 引用白名单（E999 直接丢弃）+ 越界表述改写
```

**为什么不用"删掉注入句子"的做法**：Issue 里可能正在讨论提示注入本身（本项目的仓库就是），
删句子会破坏排障语境。正确做法是保留原文、标记风险、降权、并用**结构化校验**兜底——
让"模型是否服从注入"不再决定安全性。

---

## 3. 状态机（LangGraph）

```
START → ingest → plan → act ──┐
                       ▲      │ finished / 步数用尽
                       └──────┴→ assess ──┬─ force_query → act
                                          └─ 否则 → compose ──► verify ──┬─ rewrite_pending → compose
                                                                          └─ 否则 → finalize → END
任何节点 status ∈ {failed, partial} ──────────────────────────────────────► degrade → END
```

| 节点 | 职责 | 关键输出 | 失败行为 |
|---|---|---|---|
| `ingest` | 读目标 Issue/评论/README，召回记忆，建索引 | `documents`、`warnings` | 目标 Issue 读不到→failed |
| `plan` | 生成 1~3 条检索词与待验证判断 | `plan`、`queries` | 失败→failed |
| `act` | 单步工具决策 + 执行 + 观测回填 | `documents`、`observations`、`rounds` | 工具失败只记 warning |
| `assess` | 证据覆盖度判断，是否再检索 | `force_query`、`gaps` | 失败→failed |
| `compose` | 写结构化草稿（只用可引用证据） | `draft`、`citable_ids` | 失败→partial + 降级报告 |
| `verify` | 引用/表述/依据性校验，决定是否重写 | `rewrite_pending`、`critique` | 失败→failed |
| `finalize` | 渲染 Markdown、写回记忆、汇总指标 | `report`、`citations`、`metrics` | — |
| `degrade` | 确定性降级报告 | `report` | — |

设计细节：

1. **节点名不能与 state key 同名**。LangGraph 0.2 会直接抛
   `'plan' is already being used as a state key`，所以 `plan` 节点注册为 `planning`。
2. **一次只调一个工具**。串行决策让"哪一步引入幻觉"可归因，也避免多工具并行时模型乱选。
3. **状态显式整体替换**，不用 `Annotated[list, add]` 这类 reducer：证据被谁覆盖了一眼能看出来。
4. **每个节点都写检查点**（纳秒时间戳为序号，避免节点间互相覆盖），失败节点也写，
   因此 `resume` 能从失败处继续，而不是从头再烧一遍钱。

---

## 4. 工具协议

### 4.1 注册与校验

每个工具声明 `ToolSpec`：名称、描述、JSON Schema、结果上限、超时、是否只读。
`ToolRegistry.execute` 是唯一的执行入口，顺序固定：

```
权限策略 → 注册检查 → 未知参数拦截 → jsonschema 校验
        → 同参去重（命中即返回 duplicate）
        → 预算检查（工具次数）
        → 线程池执行 + 超时
        → 结果脱敏 + 截断 + 打点
```

要点：

- **`additionalProperties: false`**：模型被注入后想塞 `repo: other/repo` 之类额外字段？
  直接拒绝。即使参数合法，`act` 节点还会强制把 `repo` 覆写为当前仓库。
- **同参去重**：重复调用返回上次结果并标 `duplicate`，从机制上让"再搜一次"变成无效动作。
- **错误不出栈**：工具失败返回 `ok=False` 的观测，节点继续跑；只有读取目标 Issue 失败才终止。
- **超时用线程池**：Python 线程无法强杀，超时只是"不再等待"。真正的超时保障是 httpx 的
  connect/read timeout（12s），线程池超时是最后一道闸。

### 4.2 两种工具协议

| 协议 | 触发 | 适用 | 取舍 |
|---|---|---|---|
| `json`（默认） | 模型输出 `{"action":...,"args":...}` | OpenAI 兼容的 JSON mode（DeepSeek 默认） | 兼容性最好，但多一层解析 |
| `native` | OpenAI `tools` + `tool_calls` | 支持 function calling 的模型 | 更原生，部分兼容端点不稳定 |

两种协议共用同一份 Schema、同一个注册表、同一套权限校验——协议只是传输层差异。

### 4.3 工具渐进披露

第一轮 `act` 只暴露检索类工具；拿到至少一轮证据（或 `assess` 明确要求补检索）后，
才把 `get_file` 加入可选工具。作用与 Skill 的按需加载一致：减少模型在信息不足时
猜测文件路径，也缩小单轮可调用的工具面。实现见 `nodes.py::_act_tools`。

### 4.4 工具清单

| 工具 | 说明 | 结果上限 |
|---|---|---|
| `get_issue` | 目标 Issue 标题/正文/标签 | 9 KB |
| `list_issue_comments` | 分页评论，长线程首尾抽样 | 9 KB |
| `get_readme` | 仓库 README | 7 KB |
| `search_issues` | 仓库内关键词检索（禁限定词） | 10 KB |
| `list_related_pull_requests` | 可能修复该问题的 PR | 6 KB |
| `get_file` | 指定文本文件（配置样例等） | 7 KB |
| `search_knowledge_base` | 本次证据上的混合检索 | 8 KB |
| `get_document` | 按 ID 取完整证据 | 8 KB |

---

## 5. 检索（RAG）

### 5.1 切块

按 Markdown 标题分段，代码块不切开；超长段落按行切并保留 100 字符重叠。
块 ID 形如 `E3#c2`，与证据 ID 绑定，保证引用可回溯到原文位置。

### 5.2 混合检索

```
BM25（k1=1.5, b=0.75，标题加权）
  + 余弦相似度（512 维哈希向量：词 + 字符三元组）
  → RRF 融合：1/(60+rank_bm25) + 1/(60+rank_dense)
  → 加权重排：0.50*RRF + 0.25*BM25_norm + 0.25*cos + 0.08*标题命中
  → 文档级去重（同一 Issue 只保留最高分块）
```

为什么默认用哈希向量：**离线可复现、零成本、无外部依赖**。它能抓词形与子词相似，
抓不住真正的同义改写——这是明确记录的局限。需要真实语义检索时设置
`EMBEDDING_PROVIDER=openai`（配合 `EMBEDDING_BASE_URL / MODEL / API_KEY`），
实现见 `rag/embed.py::OpenAICompatibleEmbedder`，无需改代码。

### 5.3 为什么还要 BM25

Issue 排障的关键词高度特异：错误码、异常类名、配置项（`SEARCH_TIMEOUT`）、版本号。
这类 token 恰恰是向量模型容易"平均掉"的。混合检索对这类查询更稳。

---

## 6. Context Engineering

Prompt 里的每条证据都必须是 `[E<id>] 标题 / URL / 类型 / 正文`，因为：
**模型要能引用，校验才能执行**。没有稳定 ID，引用校验就只能靠正则猜。

预算策略：

1. 目标 Issue 永远完整保留（最多 4200 字符），它是问题定义。
2. 其余证据按加入顺序，每条最多 1400 字符，总额受 `CONTEXT_CHAR_BUDGET` 限制。
3. 超出预算的证据不会进入写作上下文，也就**不允许被引用**（`citable_ids`），
   避免"引用了一条模型没看过的证据"。
4. 观测记录倒序拼接，最近的工具结果优先，防止长会话把关键结果挤出上下文。

已知简化：按**字符**而非 token 计预算。中文约 1.6 字符/token，英文约 4 字符/token，
混合文本会低估 20%~40%。生产应换成模型 tokenizer（见 PRODUCTION 文档）。

---

## 7. 记忆分层

| 层 | 存储 | 生命周期 | 用途 |
|---|---|---|---|
| 工作记忆 | LangGraph state | 单次运行 | 证据、观测、草稿 |
| 检查点 | `checkpoints` 表 | 40 个/会话 | 失败续跑、审计回放 |
| 情景记忆 | `episodes` 表 | 180 天（可配） | 同仓库相似问题先召回，省钱 |
| 语义事实 | `facts` 表 | 长期（带置信度） | 稳定知识（环境要求、配置含义） |
| 反馈 | `feedback` 表 | 长期 | 评测集与 Prompt 迭代 |

召回时 episode 会作为 `kind=memory` 的证据加入，并在正文里写明"N 天前、置信度 X"，
报告里也会提示"可能是过期经验"。**记忆不是事实，是线索**。

所有 SQL 强制带 `tenant`，租户隔离在存储层实现，避免"路由层忘了过滤"导致越权读。

---

### 5.4 向量化后端

默认使用确定性哈希向量（`EMBEDDING_PROVIDER=hashing`）：离线可复现、零成本。
设置 `EMBEDDING_PROVIDER=openai` 后改用任意 OpenAI 兼容 `/embeddings` 端点，
由 `rag/embed.py::OpenAICompatibleEmbedder` 负责批内缓存与失败降级（首次失败整次
运行退回哈希，避免维度混用）。权重对照脚本：`python -m issue_agent.eval.retrieval_bench`。

### 5.5 召回兜底

GitHub 搜索对整句自然语言召回很差，对错误码/异常名命中率高。因此当 `search_issues`
返回空时，节点会用 `build_fallback_query()` 从标题/正文抽取 `ReadTimeout`、`SEARCH_TIMEOUT`、
`401` 这类 token **再搜一次**（不消耗 LLM 调用），第一次的检索词与回退原因都写进
`queries` 与 `warnings`，报告里可追溯。

仍为空则如实记录 `retrieval_empty_total`，由 `assess` 决定换词重检索（≤ `MAX_ROUNDS`）
或进入写作并在 `gaps` 里声明"未找到相关历史 Issue"。**绝不因为搜不到就编一个原因。**

---

## 8. 预算与降级

预算维度：LLM 调用数、Token、金额、工具调用数、检索轮次、墙钟时间。

```
LLMClient / HeuristicLLM ──记账──► Budget ──超限──► BudgetExceeded
                                          │
                   节点包装器捕获 ──────────┘
                                          ▼
                             status=partial + degrade 节点
                                          ▼
                    有草稿 → 渲染 Markdown（标注 partial）
                    无草稿有证据 → 确定性降级报告（证据摘录 + 人工方向）
                    无证据 → 失败报告（原因 + 建议）
```

降级报告**不经过模型**，所以不会在这时候产生幻觉；同时把已经花掉的 token 变成可用信息。

### 8.1 结果缓存与并发合并

公网上同一问题常被反复提问。服务层做两级复用：

```
请求 ──► analysis_key(tenant, repo, issue, context_hash, model, prompt_fingerprint)
          │
          ├─ 命中且未过期 → 直接返回（带 cache.age_seconds / hits）
          └─ 未命中 → SingleFlight（同 key 只执行一次，其余等待共享结果）
                        │
                        └─ status == succeeded → 写入结果缓存（TTL 900s）
```

设计要点：
- **只缓存成功结果**：`partial/failed` 不缓存，避免把一次预算耗尽固化成"标准答案"。
- **Prompt 指纹失效**：`prompts.py` 对全部 Prompt 取哈希，改文案自动让旧缓存失效。
- **租户隔离**：key 含 tenant，宁可降低命中率也不跨租户复用。
- **有界 TTL**：默认 15 分钟；Issue 更新频率高的仓库可调短。

---

## 9. 可观测性

- **Span 树**：`agent.run` → `node.*` → `llm.chat_json` / `tool.*` / `retrieval`，
  每个 span 有 trace_id、parent_id、耗时、状态、属性。落盘 `data/traces/trace-*.jsonl`。
- **指标**（`/metrics`）：
  `agent_runs_total{status}`、`agent_run_seconds`、`agent_run_tokens`、
  `llm_calls_total`、`llm_latency_ms`、`llm_output_invalid_total`、`llm_truncated_total`、
  `tool_calls_total{tool}`、`tool_errors_total`、`tool_denied_total`、`tool_timeout_total`、
  `github_cache_hits_total`、`github_ratelimit_remaining`、
  `prompt_injection_detected_total{kind}`、`agent_budget_exceeded_total{kind}`、
  `api_run_duration_ms`。
- **日志**：JSON 行，含 request_id / trace_id / tenant，消息与异常自动脱敏。

---

## 10. 并发与异步模型

- FastAPI 同步端点跑在线程池；分析作业用独立的 `ThreadPoolExecutor`，避免阻塞 HTTP 线程。
- 一次运行 = 一个图实例（编译开销 ~毫秒），把 budget / tracer / index 绑定到本次运行。
- SSE 通过作业事件缓冲区实现，支持 `Last-Event-ID` 断线重放；不使用长轮询。
- SQLite 用 WAL + `timeout=5` + 每次操作显式关闭连接（`with sqlite3.Connection` 只提交不关闭，
  Windows 下会漏句柄）。
- 已知边界：限流、作业表、事件缓冲都是**进程内内存**。多副本部署必须换 Redis
  （令牌桶 + Stream/List 事件队列），否则限流与续传语义都会失效。

---

## 11. 关键取舍

| 决策 | 选择 | 替代方案 | 为什么 |
|---|---|---|---|
| 检索 | BM25 + 哈希向量 | 真 Embedding + 向量库 | 离线可跑、零成本；替换点单一 |
| 工具协议 | JSON 动作为主 | 纯 native function calling | JSON mode 跨供应商最稳 |
| 循环粒度 | 单工具/步 | 并行多工具 | 可归因、可控成本 |
| 状态合并 | 显式整体替换 | reducer 累加 | 调试友好 |
| HITL | API 层审批/反馈 | LangGraph `interrupt` | 审批发生在服务边界，更贴近业务 |
| 记忆 | SQLite + BM25 召回 | 向量记忆库 | 规模小、可解释、可审计 |
| 评测 | 合成离线 + 真 label 才 live | 只做人工体验 | 避免"用评测数字骗自己" |

---

## 12. 代码地图

```
agent/graph.py     图装配与条件路由（先看这里）
agent/nodes.py     8 个节点 + 预算/异常边界 + 检查点
agent/runner.py    依赖装配、结果规范化、流式、续跑
agent/context.py   证据渲染与预算
agent/report.py    确定性渲染 + 降级报告
llm/client.py      结构化调用、修复重试、记账
tools/base.py      注册表与执行管线
security/guard.py  注入检测、脱敏、引用校验、越界表述
eval/runner.py     离线/真实评测与门禁
```
