"""API 请求 / 响应模型。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    repo: str = Field(min_length=3, max_length=205,
                      examples=["idosal/git-mcp"])
    issue_number: int = Field(ge=1, le=2_147_483_647, examples=[214])
    user_context: str = Field(default="", max_length=2000,
                              description="本机环境、版本、已尝试步骤；会被脱敏后使用")


class FeedbackRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=64)
    rating: int = Field(ge=-1, le=1, description="1=有帮助，-1=没帮助")
    comment: str = Field(default="", max_length=1000)


class JobCreated(BaseModel):
    job_id: str
    status: str
    events_url: str
