"""结构化日志：JSON 行、自动脱敏、request_id / trace_id 贯穿。"""
from __future__ import annotations

import json
import logging
import sys
from typing import Any

from ..security.guard import redact


class JsonFormatter(logging.Formatter):
    def __init__(self, service: str = "issue-agent"):
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "service": self.service,
            "logger": record.name,
            "msg": redact(record.getMessage())[:2000],
        }
        for attr in ("request_id", "trace_id", "tenant", "run_id", "duration_ms", "status_code"):
            value = getattr(record, attr, None)
            if value is not None:
                payload[attr] = value
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info))[:4000]
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", service: str = "issue-agent") -> None:
    root = logging.getLogger()
    level_value = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(level_value)
    # 幂等：重复调用（测试里很常见）不重复挂 handler。
    for handler in list(root.handlers):
        if getattr(handler, "_issue_agent", False):
            root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter(service))
    handler._issue_agent = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    # 第三方库的 DEBUG 噪音会泄露底层细节，统一抬到 WARNING。
    for name in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
