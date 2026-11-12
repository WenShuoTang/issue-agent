<p align="center"><b>English</b> · <a href="README.zh-CN.md">简体中文</a></p>

<p align="center">
  <img src="docs/assets/logo-256.png" width="96" alt="Issue Triage Agent logo">
</p>

<h1 align="center">GitHub Issue Triage Agent (end-to-end Agentic RAG)</h1>

<p align="center">
  <img src="docs/assets/social-preview.png" width="880" alt="Issue Triage Agent">
</p>

> Give it an `owner/repo` and an issue number. The agent reads the issue, plans retrieval,
> calls read-only tools to collect evidence, judges whether the evidence is sufficient, and
> writes triage hypotheses with original source links. A critic pass removes unsupported
> claims before the report is published.
> **Read-only**: it does not run the target repository, does not modify issues or PRs, and
> never claims a fix.

Stack: **Python 3.10+ / FastAPI / LangGraph / multi-provider LLM (DeepSeek, OpenAI, Claude,
Qwen, Moonshot, Zhipu, Gemini, local Ollama/vLLM, and any OpenAI-compatible endpoint) /
GitHub REST API / SQLite (cache, memory, checkpoints) / plain HTML + SSE**.
A single machine does not need Docker, Redis, a vector database, or a GPU; the Docker setup
is only an optional deployment path.

---

## 1. Problem and closed loop

When engineers debug a production error, the real workflow is: read the issue, scan comments
for workarounds, search for similar historical issues, compare versions, form a hypothesis,
then verify it. That information is scattered across issue bodies, comments, READMEs,
historical issues, and merged PRs. Doing it by hand usually takes 20-60 minutes.

This project turns that workflow into a bounded agent:

```
Target issue ──┬─ comments (head/tail sample) ─┐
               ├─ README ──────────────────────┤
               └─ historical issues / PRs ─────┘
                        │
               hybrid retrieval (BM25 + vector + RRF)
                        │
  plan → tool calls → evidence assessment ─┬─ not enough → retrieve again (bounded)
                                            └─ enough → write → groundedness review
                                                          ├─ fail → rewrite once
                                                          └─ pass → cited report
```

Output: a summary, ranked hypotheses with confidence, **a source link for every claim**,
suggested investigation steps, verification steps, a list of missing information, the full
execution trace, and cost metrics.

---

## 2. Architecture

![Architecture](docs/assets/architecture.svg)

<details>
<summary>ASCII version (for plain-text environments)</summary>

```
┌────────────────────────────── Web UI (index.html, SSE) ──────────────────────────────┐
│ form → POST /api/v1/jobs → GET /api/v1/jobs/{id}/events (live trace) → report + feedback
└───────────────────────────────────────┬──────────────────────────────────────────────┘
                                        │
┌─────────────────────────────── FastAPI layer ────────────────────────────────────────┐
│ API-key auth / tenant isolation / sliding-window rate limit / request ID / error codes
└───────────────────────────────────────┬──────────────────────────────────────────────┘
                                        │
┌────────────────────────────── Agent orchestration (LangGraph) ────────────────────────┐
│ ingest → plan → act ⇄ assess → compose ⇄ verify → finalize                            │
│                     ↘ budget exhausted / node error → degrade (still returns a report)│
│ budget guards · loop guards · citation whitelist · unsafe-claim filter · checkpoints  │
└──────┬──────────────────┬──────────────────┬──────────────────┬──────────────────────┘
       │                  │                  │                  │
┌──────▼──────┐  ┌────────▼────────┐  ┌──────▼───────┐  ┌───────▼────────┐
│ Tool registry│  │ Hybrid RAG      │  │ Long-term    │  │ Observability  │
│ JSON schema  │  │ chunk/BM25/vec  │  │ memory       │  │ spans/metrics  │
│ timeout/dedup│  │ RRF / rerank    │  │ checkpoint   │  │ structured logs│
└──────┬──────┘  └─────────────────┘  └──────────────┘  └────────────────┘
       │
┌──────▼──────────────────────────────────────────────────────────────────────────────┐
│ GitHub read-only client: ETag requests · TTL cache · pagination · backoff · egress   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

</details>

---

## 3. Technology map (code-level)

| Capability | Implementation | Where |
|---|---|---|
| Agent loop / ReAct | one tool per step, observations fed back, bounded loop | `issue_agent/agent/nodes.py` |
| Graph engineering | LangGraph `StateGraph`, conditional routing, single failure exit | `issue_agent/agent/graph.py` |
| Structured output | Pydantic contracts, JSON parsing, repair retry with validation errors | `issue_agent/llm/client.py`, `llm/schemas.py` |
| Tool use / function calling | tool registry, JSON Schema validation, `json`/`native` protocols | `issue_agent/tools/` |
| Agentic RAG | model decides when, how often, and what to retrieve | `agent/nodes.py`, `tools/kb_tools.py` |
| Hybrid retrieval | heading-aware chunking, BM25, hashed vectors, RRF, doc-level dedupe | `issue_agent/rag/` |
| Memory | working, episodic, semantic facts, checkpoints, feedback | `issue_agent/memory/store.py` |
| Context engineering | evidence IDs/URLs, character budget, nonce-isolated untrusted text | `issue_agent/agent/context.py` |
| Guardrails | injection detection, secret redaction, citation whitelist, read-only policy | `issue_agent/security/` |
| Budgets & degradation | six-dimensional budget, deterministic degraded report on exhaustion | `issue_agent/budget.py` |
| Observability | span tree, Prometheus metrics, JSON logs, trace JSONL | `issue_agent/obs/` |
| Evaluation | offline dataset, Recall@k/MRR/citation validity, regression gate | `issue_agent/eval/` |
| Serving | FastAPI, async jobs, SSE replay, auth and rate limiting | `issue_agent/api/` |
| Resilience | conditional-request cache, exponential backoff, timeouts, dedupe, resume | `tools/github_tools.py`, `cache.py` |

---

## 4. Quick start

### Requirements (Python >= 3.10)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m issue_agent.cli serve --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>, enter a **model API key** and a **GitHub token** in the page,
and run an analysis. No source edits or `.env` are required.
Example: `idosal/git-mcp` / `214` (public issue; content may change over time).

> `cp .env.example .env` is optional. It only sets server-side defaults (for shared
> deployments, or when visitors should not need their own keys).
> Conda also works: `conda create -n issue-agent python=3.11 && conda activate issue-agent`.

### 4.1 Choose a model (12 presets in the UI)

Supported: DeepSeek, OpenAI, Anthropic Claude, Qwen (DashScope), Moonshot, Zhipu GLM,
SiliconFlow, OpenRouter, Gemini, local Ollama/vLLM, and any OpenAI-compatible endpoint.

Pick a provider in the page and paste its key; leave it empty to use the server default.

| Provider | Protocol | Example model | Key variable |
|---|---|---|---|
| `deepseek` (default) | OpenAI-compatible | `deepseek-flash` | `DEEPSEEK_API_KEY` |
| `openai` | OpenAI | `gpt-4o-mini` | `OPENAI_API_KEY` |
| `anthropic` | Messages API | `claude-sonnet-4-5` | `ANTHROPIC_API_KEY` |
| `qwen` | OpenAI-compatible | `qwen-plus` | `DASHSCOPE_API_KEY` |
| `moonshot` | OpenAI-compatible | `moonshot-v1-8k` | `MOONSHOT_API_KEY` |
| `ollama` | OpenAI-compatible | `qwen2.5:7b` | not required |

Optional server defaults (`.env`):

```ini
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```

Key acquisition steps, the compatibility matrix (JSON mode / tool calling / thinking), and
troubleshooting are in **[docs/PROVIDERS.md](docs/PROVIDERS.md)** (Chinese).

### 4.2 GitHub token (optional, entered in the UI)

Without a token you get **60 requests/hour**. With a token you get **5,000 requests/hour**
and can read private repositories you have access to.

1. GitHub → avatar → **Settings** → **Developer settings**.
2. **Personal access tokens → Fine-grained tokens → Generate new token**.
3. Repository access: **Public repositories (read-only)**, or select specific repos.
4. Permissions: **Issues: Read-only**, **Contents: Read-only**, **Metadata: Read-only**
   (optionally **Pull requests: Read-only**).
5. Paste `github_pat_...` into the **GitHub Token** field in the page.

The report metrics show whether the run used *your token*, the *server token*, or *anonymous*
access.

> The token is shown only once. Do not commit it. Detailed steps (including classic tokens
> and security notes) are in
> [docs/PROVIDERS.md#4-github-token-申请步骤重要](docs/PROVIDERS.md#4-github-token-申请步骤重要) (Chinese).

### 4.3 Bring your own credentials (BYOK)

![UI preview](docs/assets/ui-preview.png)

<sub>UI mock: provider selection plus model key and GitHub token on the left, live trace in
the middle, cited report on the right.</sub>

Both credentials are used only for the current analysis. They are **not persisted, not
logged, and not written to traces**:

1. **Provider**: choose a vendor (empty = server default).
2. **Model API key**: the key for that vendor.
3. **GitHub token**: optional; raises limits or grants access to private repos.
4. **Advanced → model name**: optional; empty uses the provider default.

Equivalent API request:

```json
POST /api/v1/analyze
{
  "repo": "idosal/git-mcp",
  "issue_number": 214,
  "llm": {"provider": "openai", "model": "gpt-4o-mini", "api_key": "sk-..."},
  "github": {"token": "github_pat_..."}
}
```

Security constraints:

- **The client cannot set `base_url`** (prevents SSRF); self-hosted gateways must be configured
  server-side via `LLM_BASE_URL`.
- Both credentials live only in memory for that run; they are never written to the database,
  the result cache, or the trace.
- Result-cache keys include the model and the GitHub-credential scope, so different tokens
  never share results.
- GitHub HTTP cache entries are scoped by token as well, so permission A's cached responses
  cannot leak to permission B.
- Selecting another provider without a key returns **400**, not 500.
- Server-side rate limits still apply; BYOK cannot be used to bypass them.

### Docker (optional)

```bash
docker compose up --build      # binds 127.0.0.1:8000 only
```

### CLI

```bash
python -m issue_agent.cli analyze --repo idosal/git-mcp --issue 214 --stream --out report.md
python -m issue_agent.cli eval --offline          # offline eval (no key or network needed)
python -m issue_agent.cli doctor                  # environment self-check
```

---

## 5. HTTP API

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/analyze` | synchronous analysis, full result |
| POST | `/api/v1/jobs` | create an async job, returns `job_id` |
| GET | `/api/v1/jobs/{id}` | job status and result |
| GET | `/api/v1/jobs/{id}/events` | SSE stream (`Last-Event-ID` replay supported) |
| GET | `/api/v1/sessions`, `/api/v1/sessions/{id}` | session history (tenant-scoped) |
| POST | `/api/v1/sessions/{id}/resume` | resume from a checkpoint |
| POST | `/api/v1/feedback` | 👍/👎 plus a comment, feeds the eval candidate pool |
| GET | `/api/v1/meta`, `/api/v1/stats`, `/api/v1/eval/latest` | config, stats, latest eval |
| GET | `/healthz`, `/readyz`, `/metrics` | probes and Prometheus metrics |

Authentication: with `API_KEYS="key:tenant,key2:tenant2"`, every `/api/v1/*` call must send
`X-API-Key`. Without keys, only loopback clients are allowed.
Errors use `{"error":{"code","message","retryable"},"request_id"}`.

`/api/v1/analyze` and `/api/v1/jobs` accept optional `llm` and `github` fields for user-provided
credentials (see 4.3); omitting them uses server defaults.

**Request reuse**: identical `(tenant, repo, issue, user context, model, prompt version,
GitHub credential scope)` requests return the cached result within `RESULT_CACHE_TTL_S`
(default 900s) and include `cache.age_seconds`. Concurrent identical requests before the cache
is written are collapsed by single-flight; running jobs are deduplicated by the same key.

---

## 6. Evaluation (reproducible numbers)

```bash
python -m issue_agent.eval.runner --offline
```

Current offline results (4 **synthetic** cases with a deterministic LLM stand-in; this
validates the pipeline, not model quality):

| Metric | Value | Meaning |
|---|---|---|
| Recall@5 | 0.833 | share of labeled related issues in the top 5 |
| MRR | 1.000 | reciprocal rank of the first relevant result |
| Citation validity | 1.000 | cited IDs must exist (zero hallucinated citations) |
| Citation coverage | 1.000 | share of hypotheses with at least one citation |
| P50 / P95 latency | ~1.0s / 1.2s | includes local SQLite overhead, excludes real model inference |
| Tests | 103 offline cases | `python -m unittest discover -s tests -t .` |

Live smoke test (`idosal/git-mcp#214`, real GitHub + DeepSeek): one successful run used
**53s / 29.4k tokens / 14 LLM calls / 8 tool calls / 2 retrieval rounds** and triggered one
groundedness rewrite. See [examples/sample-report.md](examples/sample-report.md).

> These numbers must **not** be presented as product quality. Real quality requires `--live`
> runs on public issues; the labeling workflow and gates are in
> [docs/EVALUATION.md](docs/EVALUATION.md) (Chinese).

---

## 7. Production problems (summary)

The full list (50+ items with root cause, fix, code location, and residual risk) is in
**[docs/PRODUCTION.md](docs/PRODUCTION.md)** (Chinese). Highlights:

- **GitHub rate limits and flaky networks** → conditional requests + ETag 304 + TTL cache +
  jittered backoff + limit detection (no pointless retries).
- **Huge comment threads** → pagination caps + head/tail sampling (repro steps at the start,
  conclusions at the end) with an explicit "middle skipped" warning.
- **Unstable model JSON** → `response_format=json_object`, truncation detection, repair retry
  with validation errors, and a hard failure instead of silent success.
- **Hallucinated citations** → evidence-ID whitelist; unknown IDs are dropped and counted;
  evidence URLs are restricted to GitHub domains.
- **Prompt injection** ("ignore previous instructions" inside an issue) → nonce delimiters
  around untrusted text, injection scoring, down-ranking, read-only tool whitelist, and
  forced same-repo scope.
- **Runaway agent loops** → six-dimensional budget (steps, rounds, tools, LLM calls, tokens,
  cost, wall clock) + same-argument dedupe + stop on repeated retrieval.
- **Context bloat** → target issue first, per-evidence truncation, character budget
  (known simplification: not tokenizer-based).
- **Overconfidence** → confidence levels, "hypothesis, not confirmed root cause" wording,
  unsafe-claim detection and rewriting, and a disclaimer.
- **Bad failure UX** → every node failure falls through to `degrade`, returning a readable
  report instead of a 500.
- **Invisible cost** → every run records tokens, cost, tool calls, and latency; `/metrics`
  is scrapeable.
- **Public traffic** → result cache + single-flight + job dedupe + rate limits + six budgets;
  capacity math is in section 8 of [docs/PRODUCTION.md](docs/PRODUCTION.md) (Chinese).
- **Duplicate questions** → cache keys include tenant/model/prompt fingerprint; only successful
  runs are cached; hits expose `cache.age_seconds`.
- **RAG recall failures** → when GitHub search returns nothing, fall back to error-code /
  exception-name search (no extra LLM call); if still empty, retry with new terms or state
  "insufficient evidence" — never invent a cause.
- **Multiple replicas** → rate limiting, jobs, and event buffers are single-process; production
  needs Redis. This is documented, not hand-waved.

---

## 8. Security boundaries

1. **Read-only**: the whitelist contains 8 read-only tools; write operations (comment, close,
   create PR) are rejected in the policy layer.
2. **Fixed egress**: the HTTP client is pinned to `https://api.github.com`; no arbitrary URL
   fetching (SSRF protection).
3. **Data is data**: repository content is wrapped in per-run random nonce delimiters and the
   model is told to ignore instructions inside; structured citation checks are the backstop.
4. **Redaction**: keys, tokens, private keys, JWTs, and emails are scrubbed from logs, traces,
   and error responses (`security/guard.py`).
5. **Tenant isolation**: every memory query enforces `tenant` at the SQL layer.
6. **Production fail-fast**: `ISSUE_AGENT_ENV=prod` refuses to start without API keys, CORS
   origins, and rate limiting.

---

## 9. Layout

```
issue_agent/
  config.py budget.py cache.py errors.py testing.py
  security/     guard.py policy.py            # injection detection, redaction, citations, read-only policy
  llm/          client.py schemas.py fake.py transport.py providers.py
  tools/        base.py github_tools.py kb_tools.py
  rag/          chunk.py index.py             # chunking, BM25, hashed vectors, RRF, rerank
  memory/       store.py                      # sessions, checkpoints, episodes, facts, feedback
  agent/        state.py prompts.py context.py nodes.py graph.py report.py runner.py
  api/          app.py security.py jobs.py schemas.py
  eval/         dataset.py metrics.py runner.py
  obs/          tracing.py metrics.py logs.py
  cli.py index.html
tests/          test_foundation.py test_agent.py test_api.py test_eval.py test_providers.py
docs/           ARCHITECTURE.md PRODUCTION.md EVALUATION.md PROVIDERS.md SECURITY.md IMAGES.md
```

---

## 10. Known limitations

1. The retrieval corpus is small (fewer than 30 documents per run); it is not a repository-wide
   long-term index. Cross-issue knowledge relies on episodic memory, not full-corpus embeddings.
2. The default vectorizer is a **deterministic hashed vector**; it cannot capture true paraphrase.
   Production should use bge/text-embedding models.
3. The context budget is character-based, not tokenizer-based; long Chinese text underestimates
   tokens by roughly 30%.
4. Comments are sampled from the head and tail; information in the middle of long threads may be
   skipped (the report says so).
5. No cross-repository retrieval; related issues with very different wording may not be found.
6. Rate limiting, the job queue, and the event buffer are single-process; multi-replica deployments
   need Redis.
7. The agent does not execute code, read the full repository, or parse PR diffs; "is it fixed?"
   can only be a lead, not a conclusion.
8. Offline evaluation uses synthetic data and a rule-based LLM stand-in; it validates the pipeline,
   not model quality.

---

## 11. Documentation

Detailed docs are currently written in Chinese (the code and this README are English).

| Document | Content |
|---|---|
| [docs/PROVIDERS.md](docs/PROVIDERS.md) | 12 providers, key acquisition, GitHub tokens, compatibility, troubleshooting |
| [docs/IMAGES.md](docs/IMAGES.md) | image assets, design tokens, regeneration prompts |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | state machine, tool protocol, retrieval/memory/context design, tradeoffs |
| [docs/PRODUCTION.md](docs/PRODUCTION.md) | production problem list, capacity planning, SLOs, launch checklist |
| [docs/EVALUATION.md](docs/EVALUATION.md) | metric definitions, labeling workflow, experiments, regression gate |
| [docs/SECURITY.md](docs/SECURITY.md) | threat model, mitigations, residual risks |

---

## 12. Privacy and data

- **No telemetry**: the project reports nothing; it only calls the GitHub API (public repos, or
  private repos your token can access) and the LLM endpoint you configure.
- **Keys stay local**: `.env` is gitignored; `.env.example` contains placeholders only. Run
  `python scripts/preflight.py` before committing to scan for keys, local paths, and emails.
- **Runtime artifacts contain real content**: `data/` (SQLite sessions, result cache, traces,
  reports) records issue text and your context. **Clean it before sharing traces, databases, or
  screenshots**, and never commit it.
- **Third-party data flow**: analysis content (public GitHub data plus your sanitized context) is
  sent to the LLM provider. Check their data terms before processing sensitive information, or
  point `LLM_BASE_URL` at a self-hosted endpoint.
- **The sample report is synthetic**: `examples/sample-report.md` was generated with FakeGitHub
  and a rule-based LLM; it contains no real users, repositories, or issues.

---

## 13. License and contributing

- Code is released under the [MIT License](LICENSE); third-party dependencies keep their own licenses.
- Contribution guide: [CONTRIBUTING.md](CONTRIBUTING.md); code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
- Report security issues privately via [SECURITY.md](SECURITY.md); do not open a public issue.
- Before submitting:

  ```bash
  python scripts/preflight.py
  python -m unittest discover -s tests -t .
  python -m issue_agent.eval.runner --offline
  ```

---

## 14. References

- [LangGraph StateGraph](https://reference.langchain.com/python/langgraph/graph/state/StateGraph)
- [GitHub REST Issues API](https://docs.github.com/en/rest/issues/issues)
- [DeepSeek API / JSON mode](https://api-docs.deepseek.com/guides/json_mode/)
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
