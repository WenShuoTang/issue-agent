# 大模型与 GitHub Token 配置指南

本项目不绑定任何一家模型厂商。切换供应商只需要改 `.env` 里的 `LLM_PROVIDER`，
代码、Prompt、评测、记忆、缓存全部无需改动。

- [1. 支持的供应商](#1-支持的供应商)
- [2. 切换供应商（3 步）](#2-切换供应商3-步)
- [3. 各厂商 Key 申请步骤](#3-各厂商-key-申请步骤)
- [4. GitHub Token 申请步骤（重要）](#4-github-token-申请步骤重要)
- [5. 兼容性矩阵](#5-兼容性矩阵)
- [6. 常见错误排查](#6-常见错误排查)
- [7. 成本核算](#7-成本核算)

---

## 1. 支持的供应商

| `LLM_PROVIDER` | 厂商 | 协议 | 示例模型 | Base URL | Key 环境变量 |
|---|---|---|---|---|---|
| `deepseek` | DeepSeek（默认） | OpenAI 兼容 | `deepseek-flash` | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` |
| `openai` | OpenAI | OpenAI | `gpt-4o-mini` | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `anthropic` | Anthropic Claude | Messages API | `claude-sonnet-4-5` | `https://api.anthropic.com` | `ANTHROPIC_API_KEY` |
| `qwen` | 通义千问（阿里云百炼） | OpenAI 兼容 | `qwen-plus` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY` |
| `moonshot` | Moonshot Kimi | OpenAI 兼容 | `moonshot-v1-8k` | `https://api.moonshot.cn/v1` | `MOONSHOT_API_KEY` |
| `zhipu` | 智谱 GLM | OpenAI 兼容 | `glm-4-flash` | `https://open.bigmodel.cn/api/paas/v4` | `ZHIPU_API_KEY` |
| `siliconflow` | SiliconFlow 硅基流动 | OpenAI 兼容 | `Qwen/Qwen2.5-7B-Instruct` | `https://api.siliconflow.cn/v1` | `SILICONFLOW_API_KEY` |
| `openrouter` | OpenRouter（聚合） | OpenAI 兼容 | `deepseek/deepseek-chat` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| `gemini` | Google Gemini | OpenAI 兼容端点 | `gemini-2.0-flash` | `https://generativelanguage.googleapis.com/v1beta/openai/` | `GEMINI_API_KEY` |
| `ollama` | 本地 Ollama | OpenAI 兼容 | `qwen2.5:7b` | `http://127.0.0.1:11434/v1` | 无需 Key |
| `vllm` | 本地 vLLM / 自建服务 | OpenAI 兼容 | `Qwen/Qwen2.5-7B-Instruct` | `http://127.0.0.1:8000/v1` | 无需 Key |
| `custom` | 任意 OpenAI 兼容网关 | OpenAI 兼容 | 自填 | 必须自填 | `LLM_API_KEY` |

> 默认模型名只是"能用"的起点，厂商会持续改名/下线。请以各自控制台为准，
> 用 `LLM_MODEL` 覆盖。预设定义在 `issue_agent/providers.py`，加一行就能支持新厂商。

---

## 2. 切换供应商（3 步）

1. 在 `.env` 里设置 `LLM_PROVIDER`；
2. 填对应厂商的 Key 环境变量（只需一个，其余留空）；
3. 运行自检：

```bash
python -m issue_agent.cli doctor
```

例如切到 OpenAI：

```ini
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```

切到本地 Ollama（无需 Key）：

```ini
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b
# 先启动：ollama serve && ollama pull qwen2.5:7b
```

切到公司内网网关：

```ini
LLM_PROVIDER=custom
LLM_BASE_URL=http://10.0.0.5:8000/v1
LLM_MODEL=internal-qwen
LLM_API_KEY=internal-key
```

### 2.1 前端填写凭据（模型 + GitHub）

不需要改服务器配置，打开页面即可：

1. 打开 <http://127.0.0.1:8000>；
2. **大模型供应商** 选择厂商（列表由 `/api/v1/meta` 动态下发）；
3. **模型 API Key** 填入该厂商 Key（密码输入框）；
4. **GitHub Token**（可选）填入 PAT：匿名 60 次/小时，填了 5,000 次/小时，并可读私有仓库；
5. 可选：展开「高级」填模型名；
6. 开始分析。报告指标会显示 `provider / model` 以及 GitHub 认证方式。

对应 API：

```http
POST /api/v1/analyze
Content-Type: application/json

{
  "repo": "idosal/git-mcp",
  "issue_number": 214,
  "llm": {"provider": "qwen", "model": "qwen-plus", "api_key": "sk-..."},
  "github": {"token": "github_pat_..."}
}
```

行为与安全约束：

| 规则 | 说明 |
|---|---|
| 模型 Key 生命周期 | 只存在于该次请求的 Settings 副本，运行结束即丢弃 |
| GitHub Token 生命周期 | 只用于该次运行的一次性 GitHub 客户端，结束即关闭连接 |
| 不落盘 | 两者都不写数据库、结果缓存、trace JSONL、日志 |
| 不回显 | 响应只返回 provider/model 与 `github_auth`，不含凭据 |
| 缓存隔离 | 结果缓存与 GitHub HTTP 缓存都按 Token 作用域隔离，不同权限不共享 |
| 不能指定 base_url | 防止服务端被当成 SSRF 跳板；自建端点必须服务端配置 |
| 缺模型 Key | 返回 **400**，不会静默回退到服务器 Key |
| 与服务器默认同厂商 | 不填模型 Key 时复用服务器 Key（方便自部署） |
| 限流/预算 | 仍按服务端配置执行，BYOK 不能绕过 |

> 公网多租户部署时，建议按 API Key 划分 tenant。结果缓存按
> `tenant + provider + model + Prompt 指纹 + GitHub 凭据范围` 复用；
> 同一 tenant、同一凭据的重复请求会命中缓存，不同 Token 不会互相复用。

---

## 3. 各厂商 Key 申请步骤

### DeepSeek（默认）

1. 打开 <https://platform.deepseek.com/> 注册并登录；
2. 左侧 **API keys** → **Create new API key**；
3. 复制 `sk-` 开头的 Key，填入 `DEEPSEEK_API_KEY`；
4. 模型名以控制台「模型 & 价格」为准，填到 `LLM_MODEL`（默认预设 `deepseek-flash`）。
5. 需要充值后才有额度；余额与用量在控制台可查。

### OpenAI

1. <https://platform.openai.com/> 登录；
2. **API keys** → **Create new secret key**；
3. 复制 `sk-...` 填入 `OPENAI_API_KEY`；
4. 账号需有可用额度（Billing 里充值）；
5. 模型名按账号权限选择，如 `gpt-4o-mini`、`gpt-4o`。

### Anthropic Claude

1. <https://console.anthropic.com/> 注册登录；
2. **Settings → API Keys → Create Key**；
3. 填入 `ANTHROPIC_API_KEY`；
4. 默认模型 `claude-sonnet-4-5`，如不可用请在控制台确认后改 `LLM_MODEL`；
5. 注意：Claude 没有 OpenAI 的 `response_format`，本项目靠 Prompt + 解析兜底，
   结构化输出失败率略高于 JSON mode，预算可适当放宽。

### 通义千问（阿里云百炼 / DashScope）

1. <https://bailian.console.aliyun.com/> 开通百炼；
2. 右上角 **API-KEY** → 创建；
3. 填入 `DASHSCOPE_API_KEY`；
4. 本项目使用 **OpenAI 兼容模式** 端点，不要填 DashScope 原生接口地址；
5. 模型名如 `qwen-plus`、`qwen-max`、`qwen-turbo`。

### Moonshot Kimi

1. <https://platform.moonshot.cn/> 注册；
2. **API Key 管理 → 新建**；
3. 填入 `MOONSHOT_API_KEY`；
4. 模型如 `moonshot-v1-8k` / `moonshot-v1-32k`。

### 智谱 GLM

1. <https://bigmodel.cn/> 注册；
2. **API Keys → 创建**；
3. 填入 `ZHIPU_API_KEY`；
4. 模型如 `glm-4-flash`、`glm-4-plus`；不同型号对工具调用支持不同，建议先保持 `LLM_TOOL_PROTOCOL=json`。

### SiliconFlow（硅基流动）

1. <https://cloud.siliconflow.cn/> 注册；
2. **API 密钥 → 新建**；
3. 填入 `SILICONFLOW_API_KEY`；
4. 模型名带组织前缀，如 `Qwen/Qwen2.5-7B-Instruct`。

### OpenRouter

1. <https://openrouter.ai/> 注册；
2. **Keys → Create Key**；
3. 填入 `OPENROUTER_API_KEY`；
4. 模型名形如 `deepseek/deepseek-chat`、`anthropic/claude-3.5-sonnet`（以站点列表为准）。

### Google Gemini

1. <https://aistudio.google.com/app/apikey> 创建 API Key；
2. 填入 `GEMINI_API_KEY`；
3. 使用 Google 的 **OpenAI 兼容端点**（预设已配好），模型如 `gemini-2.0-flash`。

### 本地 Ollama

```bash
ollama serve
ollama pull qwen2.5:7b
```

```ini
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b
```

小模型对 JSON 与工具调用的稳定性较弱，建议调大 `MAX_LLM_CALLS`/`MAX_REWRITES`。

### 本地 vLLM / 自建 OpenAI 兼容服务

```bash
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --port 8000
```

```ini
LLM_PROVIDER=vllm
LLM_BASE_URL=http://127.0.0.1:8000/v1
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
```

---

## 4. GitHub Token 申请步骤（重要）

> 只读公开数据可以完全不用 Token（匿名 60 次/小时），但一次分析大约要 5~10 次请求，
> 很容易触发限流。**强烈建议配置 Token**：认证后 5,000 次/小时。

### 方案 A：Fine-grained token（推荐，权限最小）

1. 登录 GitHub，点右上角头像 → **Settings**；
2. 左侧最下方 **Developer settings**；
3. **Personal access tokens → Fine-grained tokens → Generate new token**；
4. 填写：
   - **Token name**：如 `issue-agent-readonly`
   - **Expiration**：90 days（到期前记得轮换）
   - **Resource owner**：你的账号
5. **Repository access**：
   - 只分析公开仓库：选 **Public repositories (read-only)**；
   - 只分析自己的若干仓库：选 **Only select repositories** 并勾选。
6. **Permissions → Repository permissions**：
   - **Issues**：`Read-only`（必需）
   - **Contents**：`Read-only`（读 README）
   - **Metadata**：`Read-only`（默认必选）
   - **Pull requests**：`Read-only`（可选，用于查关联 PR）
   - 其余全部保持 `No access`
7. 点 **Generate token**，复制 `github_pat_...`（**只显示一次**）：优先粘贴到页面的 **GitHub Token** 输入框；需要给服务端设默认值时才写入 `.env` 的 `GITHUB_TOKEN`。

### 方案 B：Classic token（更简单，权限更粗）

1. **Settings → Developer settings → Personal access tokens → Tokens (classic)**；
2. **Generate new token (classic)**；
3. Scopes：
   - 公开仓库：勾选 `public_repo` 即可；
   - 需要读私有仓库：勾选 `repo`（权限较大，谨慎）。
4. 生成后复制 `ghp_...`，粘贴到页面 **GitHub Token** 输入框（或写入服务端 `.env`）。

### 验证 Token 是否生效

```bash
# Windows PowerShell
curl.exe -s -H "Authorization: Bearer $env:GITHUB_TOKEN" https://api.github.com/rate_limit
# macOS / Linux
curl -s -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/rate_limit
```

返回 JSON 里 `"limit": 5000` 即生效；`60` 说明没带上 Token。

或直接跑：

```bash
python -m issue_agent.cli doctor
```

### 安全提醒

- **永远不要提交 Token**：`.env` 已在 `.gitignore` 中；CI 里用 GitHub Secrets；
- 用最小权限（fine-grained + 只读），不要图省事勾 `repo` 全权限；
- 设置过期时间并定期轮换；泄露后立即在 Settings 里 **Revoke**；
- 不要把 Token 贴到 Issue、截图、录屏或聊天记录里。

---

## 5. 兼容性矩阵

| 能力 | OpenAI 兼容供应商 | Anthropic |
|---|---|---|
| 结构化 JSON | 优先用 `response_format=json_object`，不支持时自动去掉重试 | 无 JSON mode，靠 Prompt + 解析 + 校验修复 |
| 工具调用（`native`） | 用 `tools` / `tool_calls` | 自动转换为 `tools` / `tool_use` 块 |
| 默认工具协议 | `json`（兼容性最好） | `json` |
| 思考模式开关 | DeepSeek 支持，默认关闭；其他厂商自动忽略/回退 | 不适用 |
| `max_tokens` | 统一使用 | 必需字段，已自动带上 |
| 截断检测 | `finish_reason=length` | `stop_reason=max_tokens` 映射为 length |

> 切换模型后建议先跑一次 `python -m issue_agent.eval.runner --offline`（离线，不花钱）
> 再跑一条真实 Issue，观察报告里的 `llm_output_invalid_total` 与重写次数。

---

## 6. 常见错误排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `401` / `invalid api key` | Key 与 `LLM_PROVIDER` 不匹配，或 Key 未生效 | 确认填的是对应厂商的 Key 变量；重新生成 |
| `404 model not found` | `LLM_MODEL` 名字不对 | 以厂商控制台模型列表为准覆盖 |
| `404` 且请求地址错误 | Base URL 少了/多了 `/v1` | OpenAI 兼容端点通常要 `/v1`；Anthropic 不能加 `/v1` |
| `400 response_format is not supported` | 该端点不支持 JSON mode | 本项目会自动去掉该参数重试；若仍失败，把 `LLM_TOOL_PROTOCOL` 保持 `json` 并检查模型能力 |
| `429` / `insufficient_quota` | 额度不足或触发限流 | 充值/换 Key；项目内置退避重试，仍失败会降级 |
| `context_length_exceeded` | 上下文过长 | 调小 `CONTEXT_CHAR_BUDGET`、`MAX_EVIDENCE_DOCS` |
| 输出经常不是 JSON | 小模型/无 JSON mode | 换更强模型；或提高 `MAX_REWRITES`、降低单次输出复杂度 |
| Ollama `model not found` | 模型未拉取 | `ollama pull <model>` |
| 本地服务连不上 | 端口/防火墙 | 确认 `LLM_BASE_URL` 可达，vLLM 默认 8000、Ollama 默认 11434 |

---

## 7. 成本核算

项目默认单价为 0（不显示假数字）。要知道真实成本，请在 `.env` 填写你所用模型的单价
（美元/百万 token，按厂商账单）：

```ini
LLM_PRICE_INPUT_PER_M=0.27
LLM_PRICE_OUTPUT_PER_M=1.10
```

之后报告的「运行指标」与 `/metrics` 里会出现 `cost_usd`。不同供应商、不同型号价格差异很大，
评测对比时务必用同一份单价配置。
