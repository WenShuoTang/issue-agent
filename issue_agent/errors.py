"""统一错误分类。

设计要点：
- 每个异常都携带 ``code``（稳定、可被前端/告警消费）和 ``retryable``（决定重试策略）。
- 对外错误消息（``public_message``）与内部诊断信息分离，避免把 Key、路径、堆栈泄露给调用方。
"""
from __future__ import annotations


class AgentError(Exception):
    """所有可预期错误的基类。"""

    code = "agent_error"
    retryable = False
    http_status = 400

    def __init__(self, message: str, *, detail: str = "", retryable: bool | None = None):
        super().__init__(message)
        self.message = message
        self.detail = detail
        if retryable is not None:
            self.retryable = retryable

    @property
    def public_message(self) -> str:
        return self.message

    def to_dict(self) -> dict:
        payload = {"code": self.code, "message": self.public_message, "retryable": self.retryable}
        if self.detail:
            payload["detail"] = self.detail[:500]
        return payload


class ConfigError(AgentError):
    code = "config_error"
    http_status = 500


class ValidationError(AgentError):
    code = "invalid_request"


class NotFoundError(AgentError):
    code = "not_found"
    http_status = 404


class AuthenticationError(AgentError):
    code = "unauthenticated"
    http_status = 401


class RateLimitedError(AgentError):
    code = "rate_limited"
    http_status = 429
    retryable = True


class BudgetExceeded(AgentError):
    """Token / 成本 / 工具调用 / 轮次等预算耗尽。"""

    code = "budget_exceeded"
    http_status = 429

    def __init__(self, message: str, *, kind: str = "unknown"):
        super().__init__(message)
        self.kind = kind


class LLMError(AgentError):
    code = "llm_error"
    http_status = 502
    retryable = True


class LLMOutputError(LLMError):
    """模型返回内容无法通过结构化校验（不是网络问题，重试需改写 Prompt）。"""

    code = "llm_output_invalid"
    retryable = True


class ToolError(AgentError):
    code = "tool_error"
    http_status = 502
    retryable = True


class ToolDenied(ToolError):
    """工具调用被策略拒绝（越权、只读边界、参数非法）。"""

    code = "tool_denied"
    http_status = 403
    retryable = False


class GitHubError(ToolError):
    code = "github_error"


class UpstreamUnavailable(AgentError):
    code = "upstream_unavailable"
    http_status = 503
    retryable = True


class GuardrailViolation(AgentError):
    """输出未通过安全/依据性校验。"""

    code = "guardrail_violation"
    http_status = 422
