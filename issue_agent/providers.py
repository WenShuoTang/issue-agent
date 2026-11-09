"""主流大模型供应商预设。

设计目标：让用户只改 ``LLM_PROVIDER`` 一个变量就能切换模型，而不是去翻各家文档。

两条接入路线：
- ``openai``：绝大多数厂商都提供 OpenAI 兼容端点（DeepSeek / OpenAI / 通义千问 /
  Moonshot / 智谱 / SiliconFlow / OpenRouter / Gemini / Ollama / vLLM）。
- ``anthropic``：Claude 的 Messages API 与 OpenAI 不兼容，由
  :class:`issue_agent.llm.transport.AnthropicTransport` 单独适配。

注意：默认模型名只是"能用"的起点，各家会持续改名。推荐用 ``LLM_MODEL`` 显式覆盖，
或直接看 ``docs/PROVIDERS.md`` 里给出的最新型号链接。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderPreset:
    name: str
    label: str
    kind: str                 # openai | anthropic
    base_url: str
    default_model: str
    api_key_env: str
    docs_url: str
    requires_key: bool = True
    notes: str = ""
    default_tool_protocol: str = "json"


PRESETS: dict[str, ProviderPreset] = {
    "deepseek": ProviderPreset(
        name="deepseek", label="DeepSeek", kind="openai",
        base_url="https://api.deepseek.com",
        default_model="deepseek-flash",
        api_key_env="DEEPSEEK_API_KEY",
        docs_url="https://platform.deepseek.com/api_keys",
        notes="OpenAI 兼容；本项目默认供应商。模型名以控制台为准（如 deepseek-chat）。",
    ),
    "openai": ProviderPreset(
        name="openai", label="OpenAI", kind="openai",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        docs_url="https://platform.openai.com/api-keys",
        notes="支持 JSON mode 与 function calling；模型名请按账号可用列表修改。",
    ),
    "anthropic": ProviderPreset(
        name="anthropic", label="Anthropic Claude", kind="anthropic",
        base_url="https://api.anthropic.com",
        default_model="claude-sonnet-4-5",
        api_key_env="ANTHROPIC_API_KEY",
        docs_url="https://console.anthropic.com/settings/keys",
        notes="Messages API；无 JSON mode，靠 Prompt + 解析兜底；支持 tool_use。",
    ),
    "qwen": ProviderPreset(
        name="qwen", label="通义千问（阿里云 DashScope）", kind="openai",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen-plus",
        api_key_env="DASHSCOPE_API_KEY",
        docs_url="https://bailian.console.aliyun.com/",
        notes="使用 DashScope 的 OpenAI 兼容模式；注意不是 DashScope 原生 HTTP 接口。",
    ),
    "moonshot": ProviderPreset(
        name="moonshot", label="Moonshot Kimi", kind="openai",
        base_url="https://api.moonshot.cn/v1",
        default_model="moonshot-v1-8k",
        api_key_env="MOONSHOT_API_KEY",
        docs_url="https://platform.moonshot.cn/console/api-keys",
        notes="OpenAI 兼容；长上下文模型按需替换 LLM_MODEL。",
    ),
    "zhipu": ProviderPreset(
        name="zhipu", label="智谱 GLM", kind="openai",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        default_model="glm-4-flash",
        api_key_env="ZHIPU_API_KEY",
        docs_url="https://bigmodel.cn/usercenter/apikeys",
        notes="OpenAI 兼容端点；不同 GLM 型号的工具调用支持程度不同，建议先用 json 协议。",
    ),
    "siliconflow": ProviderPreset(
        name="siliconflow", label="SiliconFlow（硅基流动）", kind="openai",
        base_url="https://api.siliconflow.cn/v1",
        default_model="Qwen/Qwen2.5-7B-Instruct",
        api_key_env="SILICONFLOW_API_KEY",
        docs_url="https://cloud.siliconflow.cn/account/ak",
        notes="聚合多种开源模型；模型名带组织前缀（如 Qwen/...）。",
    ),
    "openrouter": ProviderPreset(
        name="openrouter", label="OpenRouter", kind="openai",
        base_url="https://openrouter.ai/api/v1",
        default_model="deepseek/deepseek-chat",
        api_key_env="OPENROUTER_API_KEY",
        docs_url="https://openrouter.ai/keys",
        notes="一个 Key 聚合多家模型；模型名形如 vendor/model。",
    ),
    "gemini": ProviderPreset(
        name="gemini", label="Google Gemini", kind="openai",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        default_model="gemini-2.0-flash",
        api_key_env="GEMINI_API_KEY",
        docs_url="https://aistudio.google.com/app/apikey",
        notes="走 Google 的 OpenAI 兼容端点；部分参数（如 thinking）会被忽略。",
    ),
    "ollama": ProviderPreset(
        name="ollama", label="本地 Ollama", kind="openai",
        base_url="http://127.0.0.1:11434/v1",
        default_model="qwen2.5:7b",
        api_key_env="OLLAMA_API_KEY",
        docs_url="https://ollama.com/library",
        requires_key=False,
        notes="无需 Key；小模型对 JSON 与工具调用的稳定性较弱，预算要留余量。",
    ),
    "vllm": ProviderPreset(
        name="vllm", label="本地 vLLM / 自建 OpenAI 兼容服务", kind="openai",
        base_url="http://127.0.0.1:8000/v1",
        default_model="Qwen/Qwen2.5-7B-Instruct",
        api_key_env="VLLM_API_KEY",
        docs_url="https://docs.vllm.ai/",
        requires_key=False,
        notes="需用 LLM_BASE_URL / LLM_MODEL 指定实际地址与模型；适合私有化部署。",
    ),
    "custom": ProviderPreset(
        name="custom", label="自定义 OpenAI 兼容端点", kind="openai",
        base_url="https://your-endpoint.example.com/v1",
        default_model="your-model",
        api_key_env="LLM_API_KEY",
        docs_url="",
        requires_key=True,
        notes="必须同时设置 LLM_BASE_URL 与 LLM_MODEL；企业内网网关常用这个。",
    ),
}


def get_preset(name: str) -> ProviderPreset:
    key = (name or "").strip().lower()
    preset = PRESETS.get(key)
    if preset is None:
        available = ", ".join(sorted(PRESETS))
        raise ValueError(f"未知的 LLM_PROVIDER: {name!r}；可选：{available}")
    return preset


def list_presets() -> list[ProviderPreset]:
    return [PRESETS[name] for name in sorted(PRESETS)]
