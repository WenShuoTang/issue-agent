"""多供应商接入测试：预设解析、OpenAI 兼容回退、Anthropic 协议适配。"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from issue_agent.config import Settings, load_settings
from issue_agent.errors import ConfigError, LLMError
from issue_agent.llm import (
    AnthropicTransport,
    LLMClient,
    SearchPlan,
    get_preset,
    list_presets,
)
from issue_agent.providers import PRESETS


def _anthropic_settings(tmp: str, **overrides) -> Settings:
    values = dict(
        llm_provider="anthropic", llm_kind="anthropic",
        llm_api_key="test-anthropic-key", llm_key_env="ANTHROPIC_API_KEY",
        llm_model="claude-test-model", llm_base_url="https://api.anthropic.com",
        data_dir=Path(tmp), db_path=Path(tmp) / "a.db", trace_dir=Path(tmp) / "traces",
    )
    values.update(overrides)
    return Settings(**values).validate()


class TestProviderPresets(unittest.TestCase):
    def test_presets_are_consistent(self):
        names = {preset.name for preset in list_presets()}
        self.assertEqual(names, set(PRESETS))
        for preset in list_presets():
            self.assertIn(preset.kind, {"openai", "anthropic"})
            self.assertTrue(preset.base_url.startswith(("http://", "https://")))
            self.assertTrue(preset.default_model)
            self.assertTrue(preset.api_key_env)
        self.assertEqual(get_preset("DEEPSEEK").name, "deepseek")

    def test_unknown_provider_raises(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "no-such-provider"}, clear=False):
            with self.assertRaises(ConfigError):
                load_settings(env_path=Path(tempfile.gettempdir()) / "no-such.env")

    def test_openai_preset_and_env_override(self):
        env = {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test-openai-1234567890",
               "LLM_MODEL": "gpt-4o-mini"}
        with patch.dict(os.environ, env, clear=False):
            settings = load_settings(env_path=Path(tempfile.gettempdir()) / "no-such.env")
        self.assertEqual(settings.llm_provider, "openai")
        self.assertEqual(settings.llm_kind, "openai")
        self.assertEqual(settings.llm_base_url, "https://api.openai.com/v1")
        self.assertEqual(settings.llm_model, "gpt-4o-mini")
        self.assertEqual(settings.llm_api_key, "sk-test-openai-1234567890")
        self.assertEqual(settings.llm_key_env, "OPENAI_API_KEY")
        self.assertTrue(settings.llm_requires_key)

    def test_anthropic_preset(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic",
                                     "ANTHROPIC_API_KEY": "sk-ant-test-1234567890"},
                        clear=False):
            settings = load_settings(env_path=Path(tempfile.gettempdir()) / "no-such.env")
        self.assertEqual(settings.llm_kind, "anthropic")
        self.assertEqual(settings.llm_base_url, "https://api.anthropic.com")
        self.assertEqual(settings.llm_key_env, "ANTHROPIC_API_KEY")

    def test_local_provider_does_not_require_key(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}, clear=False):
            settings = load_settings(env_path=Path(tempfile.gettempdir()) / "no-such.env")
        self.assertFalse(settings.llm_requires_key)
        # 没有 Key 也应能构造客户端（真正调用时才需要本地服务在线）
        LLMClient(settings.with_overrides(llm_api_key=""))

    def test_for_llm_switch_provider_uses_preset(self):
        settings = Settings(llm_provider="deepseek", llm_kind="openai",
                            llm_api_key="server-key", llm_model="deepseek-flash",
                            llm_base_url="https://api.deepseek.com")
        switched = settings.for_llm(provider="openai", api_key="sk-user-key",
                                    model="gpt-4o-mini")
        self.assertEqual(switched.llm_provider, "openai")
        self.assertEqual(switched.llm_base_url, "https://api.openai.com/v1")
        self.assertEqual(switched.llm_model, "gpt-4o-mini")
        self.assertEqual(switched.llm_api_key, "sk-user-key")
        self.assertTrue(switched.llm_requires_key)

    def test_for_llm_same_provider_keeps_server_endpoint(self):
        settings = Settings(llm_provider="custom", llm_kind="openai",
                            llm_api_key="server-key", llm_model="internal",
                            llm_base_url="http://10.0.0.5:8000/v1")
        same = settings.for_llm(provider="custom", model="internal-v2")
        self.assertEqual(same.llm_base_url, "http://10.0.0.5:8000/v1")
        self.assertEqual(same.llm_api_key, "server-key")
        self.assertEqual(same.llm_model, "internal-v2")

    def test_for_llm_other_provider_without_key_has_no_key(self):
        settings = Settings(llm_provider="deepseek", llm_api_key="server-key")
        switched = settings.for_llm(provider="openai")
        self.assertEqual(switched.llm_api_key, "")

    def test_for_llm_rejects_unknown_provider(self):
        with self.assertRaises(ConfigError):
            Settings().for_llm(provider="nope")

    def test_custom_provider_requires_explicit_endpoint(self):
        env = {"LLM_PROVIDER": "custom", "LLM_BASE_URL": "http://10.0.0.5:8000/v1",
               "LLM_MODEL": "internal-model", "LLM_API_KEY": "internal-key-123456"}
        with patch.dict(os.environ, env, clear=False):
            settings = load_settings(env_path=Path(tempfile.gettempdir()) / "no-such.env")
        self.assertEqual(settings.llm_base_url, "http://10.0.0.5:8000/v1")
        self.assertEqual(settings.llm_model, "internal-model")
        self.assertEqual(settings.llm_api_key, "internal-key-123456")


class _BadRequest(Exception):
    status_code = 400

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class TestOpenAICompatibleTransport(unittest.TestCase):
    def _settings(self, **overrides) -> Settings:
        base = dict(llm_provider="deepseek", llm_kind="openai", llm_api_key="test-key",
                    llm_model="deepseek-flash", llm_base_url="https://api.deepseek.com",
                    llm_thinking_disable=True)
        base.update(overrides)
        return Settings(**base)

    def test_sends_json_mode_and_thinking_switch(self):
        calls: list[dict] = []

        class Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(
                            content='{"queries":["a"],"focus":"f","checks":[],"reason":""}',
                            tool_calls=None))],
                    usage=SimpleNamespace(prompt_tokens=11, completion_tokens=5))

        settings = self._settings()
        llm = LLMClient(settings, client=SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())))
        plan = llm.chat_json("sys", "user", SearchPlan)
        self.assertEqual(plan.queries, ["a"])
        self.assertEqual(calls[0]["response_format"], {"type": "json_object"})
        self.assertEqual(calls[0]["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertEqual(llm.usage.prompt_tokens, 11)

    def test_retries_without_unsupported_params(self):
        calls: list[dict] = []

        class Completions:
            def create(self, **kwargs):
                calls.append(dict(kwargs))
                if "response_format" in kwargs or "extra_body" in kwargs:
                    raise _BadRequest("400 invalid parameter: response_format is not supported")
                return SimpleNamespace(
                    choices=[SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(
                            content='{"queries":["b"],"focus":"","checks":[],"reason":""}',
                            tool_calls=None))],
                    usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2))

        settings = self._settings(llm_provider="openai", llm_thinking_disable=False,
                                  llm_base_url="https://api.openai.com/v1")
        llm = LLMClient(settings, client=SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())))
        plan = llm.chat_json("sys", "user", SearchPlan)
        self.assertEqual(plan.queries, ["b"])
        self.assertEqual(len(calls), 2)
        self.assertNotIn("response_format", calls[1])

    def test_missing_key_fails_fast(self):
        settings = self._settings(llm_api_key="")
        with self.assertRaises(LLMError):
            LLMClient(settings)


class TestAnthropicTransport(unittest.TestCase):
    def test_messages_api_text_and_tool_use(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.path
            captured["headers"] = dict(request.headers)
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={
                "content": [
                    {"type": "text", "text": '{"queries":["c"],"focus":"","checks":[],"reason":""}'},
                    {"type": "tool_use", "id": "tool-1", "name": "search_issues",
                     "input": {"query": "c"}},
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 20, "output_tokens": 8},
            })

        with tempfile.TemporaryDirectory() as folder:
            settings = _anthropic_settings(folder)
            http_client = httpx.Client(base_url="https://api.anthropic.com",
                                       transport=httpx.MockTransport(handler),
                                       headers={"x-api-key": "test-anthropic-key",
                                                "anthropic-version": "2023-06-01"})
            transport = AnthropicTransport(settings, client=http_client)
            turn, usage = transport.complete(
                [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}],
                max_tokens=100, temperature=0.1,
                tools=[{"type": "function", "function": {
                    "name": "search_issues", "description": "search",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}])

            self.assertEqual(captured["path"], "/v1/messages")
            self.assertEqual(captured["headers"]["x-api-key"], "test-anthropic-key")
            self.assertEqual(captured["body"]["system"], "SYS")
            self.assertEqual(captured["body"]["tools"][0]["name"], "search_issues")
            self.assertIn("input_schema", captured["body"]["tools"][0])
            self.assertEqual(turn.finish_reason, "stop")
            self.assertEqual(turn.tool_calls[0]["name"], "search_issues")
            self.assertEqual(json.loads(turn.tool_calls[0]["arguments"]), {"query": "c"})
            self.assertEqual(usage["prompt_tokens"], 20)

    def test_llm_client_end_to_end_json(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "content": [{"type": "text",
                             "text": '{"queries":["d"],"focus":"f","checks":[],"reason":""}'}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 12, "output_tokens": 7},
            })

        with tempfile.TemporaryDirectory() as folder:
            settings = _anthropic_settings(folder)
            http_client = httpx.Client(base_url="https://api.anthropic.com",
                                       transport=httpx.MockTransport(handler))
            transport = AnthropicTransport(settings, client=http_client)
            llm = LLMClient(settings, transport=transport)
            plan = llm.chat_json("SYS", "user", SearchPlan)
            self.assertEqual(plan.queries, ["d"])
            self.assertEqual(llm.usage.prompt_tokens + llm.usage.completion_tokens, 19)
            snapshot = llm.usage.snapshot()
            self.assertEqual(snapshot["calls"], 1)


if __name__ == "__main__":
    unittest.main()
