"""LLM 适配层：供应商预设、传输协议、结构化契约、离线替身。"""
from ..providers import PRESETS, ProviderPreset, get_preset, list_presets
from .client import AssistantTurn, LLMClient, Usage, estimate_tokens, extract_json_object
from .fake import HeuristicLLM, ScriptedLLM
from .schemas import (
    ActionDecision,
    ActionStep,
    Assessment,
    CitedClaim,
    Critique,
    DraftReport,
    SearchPlan,
)
from .transport import AnthropicTransport, OpenAICompatibleTransport, build_transport

__all__ = [
    "LLMClient", "Usage", "AssistantTurn", "extract_json_object", "estimate_tokens",
    "SearchPlan", "ActionDecision", "Assessment", "CitedClaim", "ActionStep", "DraftReport", "Critique",
    "HeuristicLLM", "ScriptedLLM",
    "ProviderPreset", "PRESETS", "get_preset", "list_presets",
    "AnthropicTransport", "OpenAICompatibleTransport", "build_transport",
]
