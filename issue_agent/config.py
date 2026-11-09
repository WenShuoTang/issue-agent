"""配置集中管理：环境变量 → 不可变 Settings。

原则：
1. **默认值必须能离线跑通**（不配 Key 也能执行单元测试与离线评测）。
2. **生产模式 fail-fast**：``ISSUE_AGENT_ENV=prod`` 时缺少 API Key、缺少显式允许的
   CORS 来源、缺少租户密钥都会直接拒绝启动，而不是带着不安全的默认值上线。
3. ``describe()`` 只输出脱敏后的视图，可安全暴露给前端 / 日志。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from .errors import ConfigError
from .providers import get_preset

_REPO_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    """仅用标准库读取根目录 .env；已存在的环境变量优先（12-factor）。"""
    target = path or (_REPO_ROOT / ".env")
    if not target.exists():
        return
    for line in target.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key.isidentifier() and value:
            os.environ.setdefault(key, value)


def _env_str(name: str, default: str = "") -> str:
    # 空字符串视为「未配置」：避免 .env 里留空的项把路径/模型名覆盖成空值。
    return os.getenv(name, "").strip() or default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"环境变量 {name} 必须是整数，当前为 {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"环境变量 {name} 必须是数字，当前为 {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _split(raw: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in re.split(r"[,\s]+", raw) if x.strip())


@dataclass(frozen=True)
class Settings:
    # ---- LLM ----
    llm_provider: str = "deepseek"
    llm_kind: str = "openai"          # openai | anthropic（由预设决定）
    llm_api_key: str = ""
    llm_key_env: str = "DEEPSEEK_API_KEY"
    llm_requires_key: bool = True
    llm_model: str = "deepseek-flash"
    llm_base_url: str = "https://api.deepseek.com"
    llm_timeout: float = 35.0
    llm_max_retries: int = 2
    llm_temperature: float = 0.1
    llm_tool_protocol: str = "json"  # json | native
    llm_thinking_disable: bool = True  # 仅 DeepSeek 等支持 thinking 的供应商生效
    llm_cache_enabled: bool = False
    result_cache_enabled: bool = True
    result_cache_ttl_s: int = 900
    llm_price_input_per_m: float = 0.0
    llm_price_output_per_m: float = 0.0

    # ---- GitHub ----
    github_token: str = ""
    github_timeout: float = 12.0
    github_max_pages: int = 3
    github_http_ttl: int = 1800

    # ---- 路径 ----
    data_dir: Path = field(default_factory=lambda: _REPO_ROOT / "data")
    db_path: Path = field(default_factory=lambda: _REPO_ROOT / "data" / "agent.db")
    trace_dir: Path = field(default_factory=lambda: _REPO_ROOT / "data" / "traces")

    # ---- 预算 ----
    max_rounds: int = 3
    max_steps: int = 12
    max_llm_calls: int = 16
    max_tool_calls: int = 20
    max_rewrites: int = 1
    enable_llm_critique: bool = True
    token_budget: int = 80_000
    cost_budget_usd: float = 0.10
    wall_clock_budget_s: float = 120.0

    # ---- 上下文 / 检索 ----
    context_char_budget: int = 12_000
    max_evidence_docs: int = 24
    retrieval_top_k: int = 8
    rerank_top_k: int = 6
    embedding_dim: int = 512
    memory_recall_k: int = 3

    # ---- API / 安全 ----
    env: str = "dev"  # dev | prod
    api_keys: dict[str, str] = field(default_factory=dict)  # key -> tenant
    dev_tenant: str = "local"
    rate_limit_per_min: int = 20
    cors_origins: tuple[str, ...] = ()
    max_user_context_chars: int = 600
    log_level: str = "INFO"
    request_timeout_s: float = 120.0
    job_workers: int = 2
    job_ttl_s: int = 3600

    # ------------------------------------------------------------------
    @property
    def is_prod(self) -> bool:
        return self.env.lower() in {"prod", "production"}

    def validate(self) -> "Settings":
        if self.llm_tool_protocol not in {"json", "native"}:
            raise ConfigError("LLM_TOOL_PROTOCOL 只能是 json 或 native")
        if not self.llm_base_url.startswith(("http://", "https://")):
            raise ConfigError(f"LLM_BASE_URL 必须是 http(s) 地址，当前为 {self.llm_base_url!r}")
        if self.llm_kind not in {"openai", "anthropic"}:
            raise ConfigError(f"不支持的 LLM 协议类型：{self.llm_kind}")
        if self.max_rounds < 1 or self.max_steps < 1:
            raise ConfigError("MAX_ROUNDS / MAX_STEPS 必须 >= 1")
        if self.is_prod:
            if not self.api_keys:
                raise ConfigError("生产模式必须配置 API_KEYS（格式 key:tenant,key2:tenant2），禁止匿名访问")
            if not self.cors_origins:
                raise ConfigError("生产模式必须显式配置 CORS_ORIGINS")
            if self.rate_limit_per_min <= 0:
                raise ConfigError("生产模式必须启用限流（RATE_LIMIT_PER_MIN > 0）")
        return self

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def describe(self) -> dict:
        """脱敏配置视图（可返回给前端做能力探测）。"""
        return {
            "env": self.env,
            "llm_model": self.llm_model,
            "llm_provider": {
                "name": self.llm_provider,
                "kind": self.llm_kind,
                "api_key_env": self.llm_key_env,
                "requires_key": self.llm_requires_key,
            },
            "llm_configured": bool(self.llm_api_key) or not self.llm_requires_key,
            "github_token_configured": bool(self.github_token),
            "tool_protocol": self.llm_tool_protocol,
            "budgets": {
                "max_rounds": self.max_rounds,
                "max_steps": self.max_steps,
                "max_llm_calls": self.max_llm_calls,
                "max_tool_calls": self.max_tool_calls,
                "token_budget": self.token_budget,
                "cost_budget_usd": self.cost_budget_usd,
                "wall_clock_budget_s": self.wall_clock_budget_s,
            },
            "retrieval": {"top_k": self.retrieval_top_k, "rerank_top_k": self.rerank_top_k,
                          "embedding_dim": self.embedding_dim},
            "auth_required": bool(self.api_keys),
            "result_cache": {"enabled": self.result_cache_enabled,
                             "ttl_seconds": self.result_cache_ttl_s},
            "rate_limit_per_min": self.rate_limit_per_min,
        }

    def with_overrides(self, **kwargs) -> "Settings":
        return replace(self, **kwargs)

    def for_llm(self, *, provider: str | None = None, model: str | None = None,
                api_key: str | None = None) -> "Settings":
        """按单次请求的供应商/模型/Key 派生一份 Settings（不修改全局配置）。

        安全约束：
        - **不允许**客户端指定 base_url，避免服务端被当成 SSRF 跳板；
        - Key 只存在于本次运行的 Settings 对象里，不落盘、不进结果、不进 trace；
        - 未显式给 Key 且供应商与服务器默认一致时，复用服务器 Key（方便自部署）。
        """
        if not provider and not model and api_key is None:
            return self
        name = (provider or self.llm_provider).strip().lower()
        try:
            preset = get_preset(name)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        same = preset.name == self.llm_provider
        key = api_key if api_key is not None else (self.llm_api_key if same else "")
        chosen_model = (model or "").strip()[:120] or (
            self.llm_model if same else preset.default_model)
        return replace(
            self,
            llm_provider=preset.name,
            llm_kind=preset.kind,
            llm_api_key=key,
            llm_key_env=preset.api_key_env,
            llm_requires_key=preset.requires_key,
            llm_model=chosen_model,
            # 同供应商保留服务器配置（可能是自建/代理地址），跨供应商用预设地址。
            llm_base_url=self.llm_base_url if same else preset.base_url,
            llm_thinking_disable=(self.llm_thinking_disable if same
                                  else preset.name == "deepseek"),
        )


def _parse_api_keys(raw: str) -> dict[str, str]:
    keys: dict[str, str] = {}
    for pair in _split(raw):
        if ":" not in pair:
            raise ConfigError("API_KEYS 格式应为 'key:tenant'，多个用逗号分隔")
        key, tenant = pair.split(":", 1)
        key, tenant = key.strip(), tenant.strip()
        if len(key) < 8:
            raise ConfigError("API Key 长度至少 8，避免弱口令")
        keys[key] = tenant or "default"
    return keys


def load_settings(env_path: Path | None = None, **overrides) -> Settings:
    load_dotenv(env_path)
    data_dir = Path(_env_str("ISSUE_AGENT_DATA_DIR", str(_REPO_ROOT / "data")))

    # 供应商预设 → 允许环境变量覆盖，未覆盖时用预设默认值。
    provider_name = _env_str("LLM_PROVIDER", "deepseek").lower()
    try:
        preset = get_preset(provider_name)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    api_key = _env_str(preset.api_key_env) or _env_str("LLM_API_KEY")
    model = _env_str("LLM_MODEL") or preset.default_model
    base_url = (_env_str("LLM_BASE_URL") or preset.base_url).rstrip("/")
    thinking_disable = _env_bool("LLM_THINKING_DISABLE", preset.name == "deepseek")

    settings = Settings(
        llm_provider=preset.name,
        llm_kind=preset.kind,
        llm_api_key=api_key,
        llm_key_env=preset.api_key_env,
        llm_requires_key=preset.requires_key,
        llm_model=model,
        llm_base_url=base_url,
        llm_timeout=_env_float("LLM_TIMEOUT", 35.0),
        llm_max_retries=_env_int("LLM_MAX_RETRIES", 2),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.1),
        llm_tool_protocol=_env_str("LLM_TOOL_PROTOCOL", preset.default_tool_protocol).lower(),
        llm_thinking_disable=thinking_disable,
        llm_cache_enabled=_env_bool("LLM_CACHE_ENABLED", False),
        result_cache_enabled=_env_bool("RESULT_CACHE_ENABLED", True),
        result_cache_ttl_s=_env_int("RESULT_CACHE_TTL_S", 900),
        llm_price_input_per_m=_env_float("LLM_PRICE_INPUT_PER_M", 0.0),
        llm_price_output_per_m=_env_float("LLM_PRICE_OUTPUT_PER_M", 0.0),
        github_token=_env_str("GITHUB_TOKEN"),
        github_timeout=_env_float("GITHUB_TIMEOUT", 12.0),
        github_max_pages=_env_int("GITHUB_MAX_PAGES", 3),
        github_http_ttl=_env_int("GITHUB_HTTP_TTL", 1800),
        data_dir=data_dir,
        db_path=Path(_env_str("ISSUE_AGENT_DB", str(data_dir / "agent.db"))),
        trace_dir=Path(_env_str("ISSUE_AGENT_TRACE_DIR", str(data_dir / "traces"))),
        max_rounds=_env_int("MAX_ROUNDS", 3),
        max_steps=_env_int("MAX_STEPS", 12),
        max_llm_calls=_env_int("MAX_LLM_CALLS", 16),
        max_tool_calls=_env_int("MAX_TOOL_CALLS", 20),
        max_rewrites=_env_int("MAX_REWRITES", 1),
        enable_llm_critique=_env_bool("ENABLE_LLM_CRITIQUE", True),
        token_budget=_env_int("TOKEN_BUDGET", 80_000),
        cost_budget_usd=_env_float("COST_BUDGET_USD", 0.10),
        wall_clock_budget_s=_env_float("WALL_CLOCK_BUDGET_S", 120.0),
        context_char_budget=_env_int("CONTEXT_CHAR_BUDGET", 12_000),
        max_evidence_docs=_env_int("MAX_EVIDENCE_DOCS", 24),
        retrieval_top_k=_env_int("RETRIEVAL_TOP_K", 8),
        rerank_top_k=_env_int("RERANK_TOP_K", 6),
        embedding_dim=_env_int("EMBEDDING_DIM", 512),
        memory_recall_k=_env_int("MEMORY_RECALL_K", 3),
        env=_env_str("ISSUE_AGENT_ENV", "dev").lower(),
        api_keys=_parse_api_keys(_env_str("API_KEYS")),
        dev_tenant=_env_str("DEV_TENANT", "local"),
        rate_limit_per_min=_env_int("RATE_LIMIT_PER_MIN", 20),
        cors_origins=_split(_env_str("CORS_ORIGINS")),
        max_user_context_chars=_env_int("MAX_USER_CONTEXT_CHARS", 600),
        log_level=_env_str("LOG_LEVEL", "INFO").upper(),
        request_timeout_s=_env_float("REQUEST_TIMEOUT_S", 120.0),
        job_workers=_env_int("JOB_WORKERS", 2),
        job_ttl_s=_env_int("JOB_TTL_S", 3600),
    )
    if overrides:
        settings = replace(settings, **overrides)
    settings.ensure_dirs()
    return settings.validate()
