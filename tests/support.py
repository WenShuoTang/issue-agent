"""测试支撑：从 issue_agent.testing 统一导出，避免测试与生产代码各维护一份假实现。"""
from issue_agent.testing import FakeGitHub, make_settings, sample_store

__all__ = ["FakeGitHub", "make_settings", "sample_store"]
