#!/usr/bin/env python3
"""开源前 / 提交前的隐私与密钥自检。

检查项：
1. 是否把 `.env`、`data/`、`.db` 之类的本地文件纳入了版本控制（本脚本只看工作区）；
2. 是否有真实形态的 API Key / Token / 私钥；
3. 是否写入了本机绝对路径（`C:\\Users\\...`、`D:\\...`）与用户名；
4. 是否残留个人邮箱（文档里的示例邮箱白名单除外）。

用法：
    python scripts/preflight.py
    python scripts/preflight.py --root .
退出码 1 表示发现问题，可直接放进 CI 或 pre-commit。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", "data", ".venv", "venv", "node_modules",
             ".pytest_cache", ".mypy_cache", "dist", "build"}
SKIP_SUFFIXES = {".db", ".db-wal", ".db-shm", ".pyc", ".png", ".jpg", ".jpeg",
                 ".gif", ".webp", ".ico", ".zip", ".whl"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".html", ".yml", ".yaml", ".toml", ".json",
                 ".ini", ".cfg", ".example", ".ps1", ".sh", ""}

SECRET_PATTERNS = [
    ("deepseek/openai key", re.compile(r"\bsk-[A-Za-z0-9]{24,}\b")),
    ("github token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b")),
    ("github fine-grained token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b")),
    ("aws access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
]
LOCAL_PATH_PATTERNS = [
    ("windows user path", re.compile(r"[A-Za-z]:\\Users\\[^\s\"'<>]+")),
    ("posix home path", re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+/")),
    ("local drive path", re.compile(r"\b[A-Za-z]:\\(?:AI|Users|Projects|code|dev)\\", re.I)),
]
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[A-Za-z]{2,}\b")
ALLOWED_EMAILS = {"dev@example.com", "security@example.com", "noreply@example.com",
                  "you@example.com", "user@example.com"}

# 测试里故意写的假凭据，不视为泄露
ALLOW_PREFIXES = ("tests/", "issue_agent/testing.py")


def iter_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        yield path


def main() -> int:
    parser = argparse.ArgumentParser(description="隐私与密钥自检")
    parser.add_argument("--root", default=".", help="仓库根目录")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    problems: list[str] = []
    warnings: list[str] = []

    for name in (".env", "data", "report.md"):
        if (root / name).exists():
            warnings.append(f"本地敏感文件/目录存在：{name}（已在 .gitignore 中，确认未被 git 跟踪即可）")

    for path in iter_files(root):
        rel = path.relative_to(root).as_posix()
        if path.name == ".env":
            # 本地密钥文件不扫描内容（本来就不该存在），只在上面的存在性检查里提示。
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for label, pattern in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                if rel.startswith(ALLOW_PREFIXES):
                    continue
                line = text[: match.start()].count("\n") + 1
                problems.append(f"[密钥] {rel}:{line} 命中 {label}：{match.group(0)[:24]}...")
        for label, pattern in LOCAL_PATH_PATTERNS:
            for match in pattern.finditer(text):
                line = text[: match.start()].count("\n") + 1
                problems.append(f"[本机路径] {rel}:{line} 命中 {label}：{match.group(0)[:60]}")
        for match in EMAIL_PATTERN.finditer(text):
            email = match.group(0)
            if email in ALLOWED_EMAILS:
                continue
            line = text[: match.start()].count("\n") + 1
            problems.append(f"[邮箱] {rel}:{line} 出现个人邮箱：{email}")

    if warnings:
        print("提示（不影响通过）：")
        for item in dict.fromkeys(warnings):
            print("  - " + item)

    if problems:
        print("\n发现以下潜在隐私/密钥问题：\n", file=sys.stderr)
        for item in dict.fromkeys(problems):
            print("  " + item, file=sys.stderr)
        print("\n处理建议：删除或替换为占位符；确认 .env / data/ 未被 git 跟踪。", file=sys.stderr)
        return 1

    print("preflight 通过：未发现密钥、本机路径或个人邮箱。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
