"""FastAPI 应用：鉴权、限流、同步/异步分析、SSE 流式、反馈、指标。

设计原则：
- **路由层薄**：业务都在 runner / memory，路由只做协议转换与错误映射。
- **错误可分类**：AgentError 自带 code + HTTP 状态码，前端可据此决定是否重试。
- **不泄露内部细节**：500 只返回 request_id，堆栈进日志。
"""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse

from ..config import ConfigError, Settings, load_settings
from ..agent.prompts import PROMPT_FINGERPRINT
from ..errors import AgentError, NotFoundError, ValidationError
from ..obs.logs import configure_logging, get_logger
from ..obs.metrics import METRICS
from ..providers import list_presets
from ..results import analysis_key
from ..tools.github_tools import validate_repo
from .jobs import JobManager, sse_format
from .schemas import AnalyzeRequest, FeedbackRequest, JobCreated
from .security import ApiKeyAuthenticator, RateLimiter, http_exception_for, request_context_middleware

log = get_logger("issue_agent.api")
TERMINAL = {"succeeded", "partial", "failed"}


def _llm_override(req: AnalyzeRequest) -> dict | None:
    """把请求里的 LLMOverride 转成 runner 可用的字典（Key 只存在于内存）。

    注意：不把 ``SecretStr`` 直接 model_dump，避免把密钥对象带进其他结构或日志。
    """
    if req.llm is None:
        return None
    override: dict = {}
    if req.llm.provider:
        override["provider"] = req.llm.provider
    if req.llm.model:
        override["model"] = req.llm.model
    if req.llm.api_key:
        override["api_key"] = req.llm.api_key.get_secret_value()
    return override or None


def _resolve_llm(runner, req: AnalyzeRequest) -> tuple[dict | None, Settings]:
    """校验并解析本次请求的 LLM 配置；非法/缺 Key 时返回 400 而不是 502。"""
    override = _llm_override(req)
    try:
        effective = runner.settings.for_llm(**override) if override else runner.settings
    except ConfigError as exc:
        raise ValidationError(str(exc)) from exc
    # 只有真实客户端（非测试注入）才做 Key 预检；否则离线评测/单测会被挡住。
    if not runner.llm_is_injected and effective.llm_requires_key and not effective.llm_api_key:
        raise ValidationError(
            f"供应商 {effective.llm_provider} 需要 API Key：请在页面填写，"
            "或在服务端配置默认 Key（docs/PROVIDERS.md）")
    return override, effective


def create_app(settings: Settings | None = None, *, runner=None,
               job_manager: JobManager | None = None) -> FastAPI:
    settings = settings or load_settings()
    configure_logging(settings.log_level)
    if runner is None:
        from ..agent.runner import AgentRunner

        runner = AgentRunner(settings)
    manager = job_manager or JobManager(workers=settings.job_workers,
                                        ttl_seconds=settings.job_ttl_s)
    auth = ApiKeyAuthenticator(settings)
    limiter = RateLimiter(settings.rate_limit_per_min)
    require_tenant = auth.dependency()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        log.info("API 启动 env=%s model=%s", settings.env, settings.llm_model)
        try:
            yield
        finally:
            manager.shutdown()
            runner.close()

    app = FastAPI(title="GitHub Issue 排障 Agent",
                  version="1.0.0",
                  description="只读 Agentic RAG：Issue → 规划 → 工具检索 → 证据校验 → 带引用报告",
                  lifespan=lifespan)
    app.state.settings = settings
    app.state.runner = runner
    app.state.jobs = manager
    app.state.limiter = limiter
    app.middleware("http")(request_context_middleware)
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                           allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-API-Key"])

    # ---------------------------------------------------------------- 异常映射
    @app.exception_handler(AgentError)
    async def _agent_error(request: Request, exc: AgentError):
        payload = {"error": exc.to_dict(),
                   "request_id": getattr(request.state, "request_id", "")}
        return JSONResponse(status_code=exc.http_status, content=payload)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception("未处理异常 path=%s", request.url.path)
        return JSONResponse(status_code=500,
                            content={"error": {"code": "internal_error",
                                               "message": "内部执行失败，请稍后重试"},
                                     "request_id": getattr(request.state, "request_id", "")})

    # ---------------------------------------------------------------- UI / 探针
    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return (Path(__file__).resolve().parent.parent / "index.html").read_text(encoding="utf-8")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "time": time.time()}

    @app.get("/readyz")
    def readyz() -> JSONResponse:
        checks = {"db": False, "llm": bool(settings.llm_api_key), "github": True}
        try:
            runner.memory.stats(tenant=settings.dev_tenant)
            checks["db"] = True
        except Exception:
            checks["db"] = False
        status_code = 200 if checks["db"] else 503
        return JSONResponse(status_code=status_code,
                            content={"status": "ready" if status_code == 200 else "degraded",
                                     "checks": checks})

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        return METRICS.render()

    @app.get("/api/v1/meta")
    def meta() -> dict:
        return {
            **settings.describe(),
            "default_provider": settings.llm_provider,
            "providers": [{"name": preset.name, "label": preset.label,
                           "kind": preset.kind, "default_model": preset.default_model,
                           "requires_key": preset.requires_key,
                           "docs_url": preset.docs_url} for preset in list_presets()],
            "version": app.version,
        }

    # ---------------------------------------------------------------- 分析
    @app.post("/api/v1/analyze")
    def analyze(req: AnalyzeRequest, request: Request,
                tenant: str = Depends(require_tenant)) -> dict:
        limiter.check(tenant)
        override, _ = _resolve_llm(runner, req)
        try:
            return runner.run(req.repo, req.issue_number, user_context=req.user_context,
                              tenant=tenant, llm_override=override)
        except AgentError as exc:
            raise http_exception_for(exc, getattr(request.state, "request_id", "")) from exc

    @app.post("/api/v1/jobs", response_model=JobCreated)
    def create_job(req: AnalyzeRequest, tenant: str = Depends(require_tenant)) -> JobCreated:
        limiter.check(tenant)
        override, effective = _resolve_llm(runner, req)

        def work(emit) -> dict:
            final: dict = {}
            for event in runner.stream(req.repo, req.issue_number,
                                       user_context=req.user_context, tenant=tenant,
                                       llm_override=override):
                emit(event)
                if event.get("event") == "final":
                    final = event.get("result") or {}
            return final

        try:
            normalized = validate_repo(req.repo)
        except Exception:
            normalized = req.repo
        # 去重键只包含供应商/模型，不包含 Key（同一模型的不同 Key 可共享结果）。
        job = manager.submit(tenant, work, dedupe_key=analysis_key(
            tenant=tenant, repo=normalized, issue_number=req.issue_number,
            user_context=req.user_context, provider=effective.llm_provider,
            model=effective.llm_model, prompt_fingerprint=PROMPT_FINGERPRINT))
        return JobCreated(job_id=job.job_id, status=job.status,
                          events_url=f"/api/v1/jobs/{job.job_id}/events")

    def _get_job(job_id: str, tenant: str):
        job = manager.get(job_id, tenant)
        if job is None:
            raise NotFoundError("作业不存在或不属于当前租户")
        return job

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: str, tenant: str = Depends(require_tenant)) -> dict:
        return _get_job(job_id, tenant).public()

    @app.get("/api/v1/jobs/{job_id}/events")
    async def job_events(job_id: str, request: Request,
                         last_event_id: int | None = Header(default=None, alias="Last-Event-ID"),
                         tenant: str = Depends(require_tenant)):
        job = _get_job(job_id, tenant)
        cursor = max(0, int(last_event_id or 0))
        deadline = time.monotonic() + settings.request_timeout_s + 30

        async def stream() -> AsyncIterator[str]:
            nonlocal cursor
            idle = 0.0
            while time.monotonic() < deadline:
                events = job.snapshot(cursor)
                for event in events:
                    yield sse_format(event, cursor)
                    cursor += 1
                if job.status in TERMINAL and cursor >= len(job.events):
                    yield sse_format({"event": "end", "status": job.status, "job_id": job.job_id})
                    return
                if await request.is_disconnected():
                    return
                await asyncio.sleep(0.15)
                idle += 0.15
                if idle >= 10:
                    idle = 0.0
                    yield ": keep-alive\n\n"
            yield sse_format({"event": "end", "status": "timeout", "job_id": job.job_id})

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "Connection": "keep-alive",
                                          "X-Accel-Buffering": "no"})

    # ---------------------------------------------------------------- 会话 / 记忆
    @app.get("/api/v1/sessions")
    def list_sessions(tenant: str = Depends(require_tenant)) -> dict:
        return {"sessions": runner.memory.list_sessions(tenant=tenant, limit=20)}

    @app.get("/api/v1/sessions/{session_id}")
    def get_session(session_id: str, tenant: str = Depends(require_tenant)) -> dict:
        session = runner.memory.load_session(session_id, tenant=tenant)
        if session is None:
            raise NotFoundError("会话不存在")
        return session

    @app.post("/api/v1/sessions/{session_id}/resume")
    def resume_session(session_id: str, tenant: str = Depends(require_tenant)) -> dict:
        limiter.check(tenant)
        try:
            return runner.resume(session_id, tenant=tenant)
        except AgentError as exc:
            raise http_exception_for(exc, "") from exc

    @app.post("/api/v1/feedback")
    def feedback(req: FeedbackRequest, tenant: str = Depends(require_tenant)) -> dict:
        runner.memory.record_feedback(run_id=req.run_id, tenant=tenant,
                                      rating=req.rating, comment=req.comment)
        return {"ok": True, "feedback": runner.memory.feedback_stats(tenant=tenant)}

    @app.get("/api/v1/stats")
    def stats(tenant: str = Depends(require_tenant)) -> dict:
        return {
            "memory": runner.memory.stats(tenant=tenant),
            "feedback": runner.memory.feedback_stats(tenant=tenant),
            "jobs": manager.stats(),
            "results": runner.results.stats(tenant=tenant),
            "rate_limit": {"limit_per_min": settings.rate_limit_per_min,
                           "current": limiter.snapshot().get(tenant, 0)},
            "metrics": METRICS.snapshot(),
        }

    @app.get("/api/v1/eval/latest")
    def eval_latest(tenant: str = Depends(require_tenant)) -> dict:
        path = settings.data_dir / "eval" / "latest.json"
        if not path.exists():
            raise NotFoundError("尚未运行评测：python -m issue_agent.eval.runner --offline")
        return json.loads(path.read_text(encoding="utf-8"))

    return app


# uvicorn issue_agent.api.app:app 的默认入口（生产建议用工厂 + 显式配置）
def __getattr__(name: str):
    if name == "app":
        application = create_app()
        globals()["app"] = application
        return application
    raise AttributeError(name)
