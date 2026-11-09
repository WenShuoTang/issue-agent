"""安全护栏：不可信内容净化、提示注入检测、密钥脱敏、引用/断言校验。

核心信任模型（写进 docs/SECURITY.md 的第一句话）：
    GitHub Issue / 评论 / README / 搜索结果 **全部** 是外部不可信数据。
它们可以包含指令、角色设定、伪造的系统消息、零宽字符、超长 Base64。Agent 必须
把「数据」和「指令」在结构上分开，而不是靠模型「自觉」。

因此这里提供四件事：
1. ``sanitize_untrusted``：去掉控制字符/零宽/HTML 注释等不可见载荷，并给出注入风险分。
2. ``wrap_untrusted``：用每次运行随机 nonce 的边界标记包裹数据，防止数据伪造边界。
3. ``redact``：日志、trace、错误消息里抹掉 Key / Token / 邮箱 / 私钥。
4. ``validate_citations`` / ``check_unsafe_claims``：让「有依据」成为可执行校验，
   而不是 Prompt 里的一句请求。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# 1. 脱敏
# ---------------------------------------------------------------------------
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"(?i)\b(authorization|api[_-]?key|token|secret|password)\b\s*[:=]\s*['\"]?([^\s'\"]{8,})"),
     r"\1=[REDACTED]"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{12,}"), "Bearer [REDACTED]"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
     "[REDACTED_PRIVATE_KEY]"),
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"), "[REDACTED_JWT]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b"), "[REDACTED_EMAIL]"),
)


def redact(text: str) -> str:
    """幂等地抹掉常见凭据与个人邮箱；用于日志、trace、错误返回。"""
    if not text:
        return ""
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def contains_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern, _ in _SECRET_PATTERNS[:8])


# ---------------------------------------------------------------------------
# 2. 不可信内容净化 + 注入检测
# ---------------------------------------------------------------------------
_INVISIBLE = re.compile(
    "[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]"
)
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_HTML_COMMENT = re.compile(r"<!--[\s\S]{0,4000}?-->")
_FENCE = re.compile(r"```")

# (标签, 正则, 权重)。权重只用于排序与告警，不用于「一票否决」。
_INJECTION_SIGNALS: tuple[tuple[str, re.Pattern[str], float], ...] = (
    ("ignore_instructions", re.compile(
        r"(?i)\b(ignore|disregard|forget|override)\b[^\n]{0,40}\b(previous|prior|above|earlier|all)\b"
        r"[^\n]{0,20}\b(instruction|prompt|rule|direction)s?\b"), 0.9),
    ("ignore_instructions_zh", re.compile(
        r"(忽略|无视|忘记|跳过)[^\n]{0,12}(之前|以上|上面|前面|所有|全部)[^\n]{0,8}"
        r"(指令|提示|规则|要求|设定)"), 0.9),
    ("system_prompt_probe", re.compile(
        r"(?i)(system\s*prompt|developer\s*message|hidden\s*instruction|你的系统提示|系统提示词|初始指令)"), 0.55),
    ("role_override", re.compile(
        r"(?i)(you\s+are\s+now|from\s+now\s+on\s+you|act\s+as\s+(an?\s+)?(admin|root|developer)|"
        r"现在你是|从现在开始你是|扮演)"), 0.7),
    ("credential_exfiltration", re.compile(
        r"(?i)(send|post|upload|exfiltrate|print|reveal|泄露|输出|发送|回传)[^\n]{0,30}"
        r"(api[_\s-]?key|token|secret|password|env|环境变量|密钥)"), 0.95),
    ("shell_injection", re.compile(
        r"(?i)(curl|wget|powershell|bash\s+-c|rm\s+-rf|nc\s+-e|Invoke-WebRequest)[^\n]{0,80}"), 0.6),
    ("fake_tool_result", re.compile(
        r"(?i)(<\/?(system|assistant|tool|function)[^>]*>|\[\s*(system|tool)\s*\]|"
        r"^\s*(observation|action)\s*:)", re.M), 0.6),
    ("prompt_delimiter_forgery", re.compile(
        r"(?i)(<<<\s*UNTRUSTED|END\s+UNTRUSTED|BEGIN\s+UNTRUSTED|###\s*(system|instruction))"), 0.8),
    ("urgency_pressure", re.compile(
        r"(?i)(urgent|immediately|do\s+not\s+tell|不要告诉|立即执行|马上执行|绕过)"), 0.35),
)

_LONG_BASE64 = re.compile(r"[A-Za-z0-9+/]{200,}={0,2}")


@dataclass
class InjectionScan:
    risk: float = 0.0
    labels: list[str] = field(default_factory=list)
    hits: list[str] = field(default_factory=list)

    @property
    def is_suspicious(self) -> bool:
        return self.risk >= 0.5

    def to_dict(self) -> dict:
        return {"risk": round(self.risk, 3), "labels": self.labels, "hits": self.hits[:5]}


def scan_injection(text: str) -> InjectionScan:
    """对单个不可信文本打分。风险分用于：降权该证据、在报告中提示、写入告警指标。"""
    scan = InjectionScan()
    if not text:
        return scan
    for label, pattern, weight in _INJECTION_SIGNALS:
        match = pattern.search(text)
        if match:
            scan.labels.append(label)
            scan.hits.append(match.group(0)[:120].replace("\n", " "))
            scan.risk = min(1.0, scan.risk + weight)
    if _LONG_BASE64.search(text):
        scan.labels.append("long_base64_blob")
        scan.risk = min(1.0, scan.risk + 0.2)
    invisible = len(_INVISIBLE.findall(text))
    if invisible > 20:
        scan.labels.append("invisible_characters")
        scan.risk = min(1.0, scan.risk + 0.25)
    if text.count("```") > 6:
        scan.labels.append("fence_flood")
        scan.risk = min(1.0, scan.risk + 0.1)
    scan.labels = list(dict.fromkeys(scan.labels))
    scan.risk = round(scan.risk, 3)
    return scan


def sanitize_untrusted(text: str, *, max_chars: int = 6000) -> tuple[str, InjectionScan]:
    """清掉不可见载荷后返回 (clean_text, scan)。**不删除注入句子本身**。

    这是一个刻意的取舍：直接删句子会破坏排障语境（Issue 里可能正在讨论
    "ignore previous instructions" 这类提示词攻击）。正确做法是保留原文、
    标记风险、并由结构化的引用校验兜底，见 docs/SECURITY.md。
    """
    if not text:
        return "", InjectionScan()
    scan = scan_injection(text)
    clean = _ANSI.sub("", text)
    clean = _INVISIBLE.sub("", clean)
    clean = _HTML_COMMENT.sub("[removed-html-comment]", clean)
    clean = clean.replace("\r\n", "\n").replace("\r", "\n")
    # 折叠 3 个以上连续空行，降低上下文浪费。
    clean = re.sub(r"\n{4,}", "\n\n\n", clean).strip()
    if len(clean) > max_chars:
        clean = clean[:max_chars] + "\n...[内容已截断]"
    return clean, scan


_RUN_NONCE: str = ""


def new_run_nonce() -> str:
    """每次运行生成新 nonce，防止 Issue 正文提前伪造边界标记。"""
    global _RUN_NONCE
    import secrets

    _RUN_NONCE = secrets.token_hex(4)
    return _RUN_NONCE


def current_nonce() -> str:
    global _RUN_NONCE
    if not _RUN_NONCE:
        return new_run_nonce()
    return _RUN_NONCE


def wrap_untrusted(source_id: str, text: str, *, nonce: str | None = None) -> str:
    token = nonce or current_nonce()
    return (
        f"<<<UNTRUSTED:{token}:{source_id}>>>\n{text}\n<<<END_UNTRUSTED:{token}:{source_id}>>>"
    )


# ---------------------------------------------------------------------------
# 3. 引用与断言校验
# ---------------------------------------------------------------------------
_URL_OK_SCHEMES = {"https"}
_ALLOWED_HOSTS = {"github.com", "www.github.com", "docs.github.com", "raw.githubusercontent.com"}


def is_safe_url(url: str, *, allowed_hosts: set[str] | None = None) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in _URL_OK_SCHEMES:
        return False
    host = (parsed.hostname or "").lower()
    if not host or parsed.username or parsed.password:
        return False
    return host in (allowed_hosts or _ALLOWED_HOSTS)


def validate_citations(citation_ids, allowed_ids: set[str]) -> tuple[list[str], list[str]]:
    """返回 (有效引用, 被丢弃的引用)。模型幻觉出来的 ID 一律丢弃。"""
    valid: list[str] = []
    invalid: list[str] = []
    if not isinstance(citation_ids, (list, tuple)):
        return valid, invalid
    for raw in citation_ids:
        if not isinstance(raw, str):
            invalid.append(str(raw))
            continue
        cid = raw.strip()
        if cid in allowed_ids:
            if cid not in valid:
                valid.append(cid)
        else:
            invalid.append(cid)
    return valid, invalid


_UNSAFE_CLAIM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("claims_fixed", re.compile(r"(?i)(已(经)?(修复|解决)|修复完成|问题已解决|\bfixed\s+the\s+(bug|issue)\b|\bresolved\s+by\s+me\b)")),
    ("claims_executed", re.compile(r"(?i)(我已(经)?(执行|运行|测试|部署)|我运行了|已实际运行|\bi\s+(ran|executed|deployed)\b)")),
    ("claims_merged", re.compile(r"(?i)(已合并|已发布到|已上线|will\s+be\s+merged)")),
    ("claims_certainty", re.compile(r"(?i)((根本|确定)原因(就)?是|root\s+cause\s+is\s+definitely|100%\s*(确定|确定是))")),
)


def check_unsafe_claims(text: str) -> list[str]:
    """检测「声称已修复/已执行」等越界表述。Agent 只能给假设，不能邀功。"""
    found: list[str] = []
    for label, pattern in _UNSAFE_CLAIM_PATTERNS:
        if pattern.search(text or ""):
            found.append(label)
    return found


def scrub_unsafe_claims(text: str) -> str:
    """对最终报告做兜底改写，保证对外措辞不越界。"""
    replacements = (
        (re.compile(r"(?i)已(经)?修复"), "建议修复"),
        (re.compile(r"(?i)已(经)?解决"), "疑似已解决（待验证）"),
        (re.compile(r"(?i)我已(经)?(执行|运行|测试)"), "建议执行"),
        (re.compile(r"(?i)\bi\s+fixed\b"), "the suggested fix for"),
    )
    for pattern, replacement in replacements:
        text = pattern.sub(replacement, text)
    return text
