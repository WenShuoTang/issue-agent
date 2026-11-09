"""API 安全：API Key 鉴权、租户解析、滑动窗口限流、请求上下文中间件。"""
from __future__ import annotations

import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field

from fastapi import Header, HTTPException, Request

from ..config import Settings
from ..errors import AuthenticationError, RateLimitedError

_LOOPBACK = {"127.0.0.1", "::1", "localhost", "testclient"}


@dataclass
class RequestContext:
    request_id: str
    tenant: str
    started: float = field(default_factory=time.monotonic)


class RateLimiter:
    """单实例滑动窗口限流。

    多实例部署时必须换成 Redis 令牌桶，否则「每实例各自限流」等于没限流。
    这个限制在 docs/PRODUCTION.md 里被显式记录，而不是假装已经解决。
    """

    def __init__(self, limit_per_minute: int):
        self.limit = max(0, int(limit_per_minute))
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, tenant: str) -> None:
        if self.limit <= 0:
            return
        now = time.time()
        window_start = now - 60
        with self._lock:
            bucket = self._hits[tenant]
            while bucket and bucket[0] < window_start:
                bucket.popleft()
            if len(bucket) >= self.limit:
                retry_after = max(1, int(bucket[0] + 60 - now))
                raise RateLimitedError(
                    f"请求过于频繁（{self.limit}/分钟），请 {retry_after}s 后重试")
            bucket.append(now)

    def snapshot(self) -> dict:
        with self._lock:
            return {tenant: len(bucket) for tenant, bucket in self._hits.items()}


class ApiKeyAuthenticator:
    """API Key → 租户映射。

    - 配置了 ``API_KEYS`` 时：必须带 ``X-API-Key``，且 Key 必须命中映射。
    - 未配置时（仅 dev）：回退到 loopback 白名单，公网来源直接 401。
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def tenant_for(self, api_key: str | None, client_host: str | None) -> str:
        if self.settings.api_keys:
            if not api_key:
                raise AuthenticationError("缺少 X-API-Key 请求头")
            tenant = self.settings.api_keys.get(api_key)
            if not tenant:
                raise AuthenticationError("无效的 API Key")
            return tenant
        # dev 模式：没有配置任何 Key，只允许本机访问，避免裸奔到公网。
        host = (client_host or "").lower()
        if host and host not in _LOOPBACK:
            raise AuthenticationError("未配置 API_KEYS 时仅允许本机访问")
        return self.settings.dev_tenant

    def dependency(self):
        async def _dep(request: Request,
                       x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> str:
            tenant = self.tenant_for(x_api_key, request.client.host if request.client else None)
            request.state.tenant = tenant
            return tenant

        return _dep


async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
    request.state.request_id = request_id
    started = time.monotonic()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = str(int((time.monotonic() - started) * 1000))
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


def http_exception_for(exc: Exception, request_id: str) -> HTTPException:
    from ..errors import AgentError

    if isinstance(exc, AgentError):
        headers = {"X-Request-ID": request_id}
        if isinstance(exc, RateLimitedError):
            headers["Retry-After"] = "60"
        return HTTPException(status_code=exc.http_status,
                             detail={"code": exc.code, "message": exc.public_message,
                                     "request_id": request_id},
                             headers=headers)
    return HTTPException(status_code=500,
                         detail={"code": "internal_error",
                                 "message": "内部执行失败，请稍后重试",
                                 "request_id": request_id})
