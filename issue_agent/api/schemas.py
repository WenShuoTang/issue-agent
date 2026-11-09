"""API 请求 / 响应模型。"""
from __future__ import annotations

from pydantic import BaseModel, Field, SecretStr


class LLMOverride(BaseModel):
    """单次请求的模型选择（BYOK）。

    ``api_key`` 用 ``SecretStr`` 包裹，避免被 repr/日志意外打印；服务端不会持久化它。
    ``base_url`` **刻意不开放**：允许客户端指定地址会把服务端变成 SSRF 跳板，
    自建网关请通过服务端 ``LLM_BASE_URL`` 配置。
    """

    provider: str | None = Field(default=None, max_length=32,
                                 description="供应商预设名，如 openai / anthropic / qwen")
    model: str | None = Field(default=None, max_length=120,
                              description="可选，覆盖该供应商默认模型")
    api_key: SecretStr | None = Field(default=None, description="仅本次请求使用，不落盘")


class GitHubOverride(BaseModel):
    """单次请求的 GitHub Token（BYOK），用于访问私有仓库或提高额度。"""

    token: SecretStr | None = Field(default=None, description="仅本次请求使用，不落盘")


class AnalyzeRequest(BaseModel):
    repo: str = Field(min_length=3, max_length=205,
                      examples=["idosal/git-mcp"])
    issue_number: int = Field(ge=1, le=2_147_483_647, examples=[214])
    user_context: str = Field(default="", max_length=2000,
                              description="本机环境、版本、已尝试步骤；会被脱敏后使用")
    llm: LLMOverride | None = Field(default=None, description="可选：用户自带模型 Key")
    github: GitHubOverride | None = Field(default=None, description="可选：用户自带 GitHub Token")


class FeedbackRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=64)
    rating: int = Field(ge=-1, le=1, description="1=有帮助，-1=没帮助")
    comment: str = Field(default="", max_length=1000)


class JobCreated(BaseModel):
    job_id: str
    status: str
    events_url: str
