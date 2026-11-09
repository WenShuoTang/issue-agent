"""HTTP 层：FastAPI 应用、鉴权、限流、作业与 SSE。"""
from .app import create_app

__all__ = ["create_app"]
