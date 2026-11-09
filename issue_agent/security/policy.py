"""工具权限策略：白名单、只读边界、出口域名白名单、结果大小上限。

真实事故里，Agent「能调用什么」比「模型多聪明」更决定风险。这里的策略是
**代码级**强制，不依赖 Prompt：即使模型被注入，也只可能调用白名单内的只读工具。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..errors import ToolDenied
from .guard import is_safe_url

# 只读、无副作用、数据可公开获取的工具。
READ_ONLY_TOOLS = frozenset({
    "get_issue",
    "list_issue_comments",
    "get_readme",
    "search_issues",
    "list_related_pull_requests",
    "get_file",
    "search_knowledge_base",
    "get_document",
})
# 写操作：本项目一律不注册。保留常量是为了在策略层显式拒绝并留下审计记录。
WRITE_TOOLS = frozenset({"create_comment", "close_issue", "create_pull_request", "push_code"})

ALLOWED_EGRESS_HOSTS = frozenset({"api.github.com", "github.com", "raw.githubusercontent.com"})


@dataclass
class ToolPolicy:
    allowed_tools: frozenset[str] = READ_ONLY_TOOLS
    allowed_hosts: frozenset[str] = ALLOWED_EGRESS_HOSTS
    max_result_chars: int = 6000
    allow_write: bool = False
    denied_calls: list[str] = field(default_factory=list)

    def authorize(self, tool_name: str) -> None:
        if tool_name in WRITE_TOOLS:
            self.denied_calls.append(tool_name)
            raise ToolDenied(f"工具 {tool_name} 属于写操作，本项目只读，已拒绝")
        if tool_name not in self.allowed_tools:
            self.denied_calls.append(tool_name)
            raise ToolDenied(f"工具 {tool_name} 不在白名单内")

    def check_source_url(self, url: str) -> bool:
        """证据 URL 只接受可信代码托管域名，防止报告里出现钓鱼链接。"""
        return is_safe_url(url, allowed_hosts=set(self.allowed_hosts) | {"docs.github.com"})

    def snapshot(self) -> dict:
        return {
            "allowed_tools": sorted(self.allowed_tools),
            "allowed_hosts": sorted(self.allowed_hosts),
            "max_result_chars": self.max_result_chars,
            "allow_write": self.allow_write,
            "denied_calls": list(self.denied_calls),
        }
