"""API 层测试：鉴权、租户隔离、限流、同步/异步分析、SSE、反馈、指标。"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
import time
import unittest
from unittest.mock import PropertyMock, patch

from fastapi.testclient import TestClient

from issue_agent.agent import AgentRunner
from issue_agent.api.app import create_app
from issue_agent.llm import HeuristicLLM
from issue_agent.testing import FakeGitHub, make_settings

VALID_KEY = "test-api-key-123456"


def _app(folder: str, **overrides):
    settings = make_settings(folder, **overrides)
    runner = AgentRunner(settings, llm=HeuristicLLM(), github=FakeGitHub())
    return create_app(settings, runner=runner), runner


class TestApi(unittest.TestCase):
    def test_public_endpoints(self):
        with tempfile.TemporaryDirectory() as folder:
            app, _ = _app(folder)
            with TestClient(app) as client:
                self.assertEqual(client.get("/healthz").status_code, 200)
                self.assertEqual(client.get("/readyz").json()["checks"]["db"], True)
                meta = client.get("/api/v1/meta").json()
                self.assertNotIn("llm_api_key", json.dumps(meta))
                self.assertIn("budgets", meta)
                metrics = client.get("/metrics").text
                self.assertIn("agent_runs_total", metrics)
                home = client.get("/")
                self.assertEqual(home.status_code, 200)
                self.assertIn("GitHub Issue", home.text)

    def test_dev_mode_rejects_non_loopback(self):
        with tempfile.TemporaryDirectory() as folder:
            app, _ = _app(folder)
            with TestClient(app, client=("203.0.113.9", 5000)) as client:
                response = client.get("/api/v1/stats")
                self.assertEqual(response.status_code, 401)
                self.assertEqual(response.json()["error"]["code"], "unauthenticated")

    def test_api_key_authentication(self):
        with tempfile.TemporaryDirectory() as folder:
            app, _ = _app(folder, api_keys={VALID_KEY: "acme"})
            with TestClient(app) as client:
                self.assertEqual(client.get("/api/v1/stats").status_code, 401)
                self.assertEqual(
                    client.get("/api/v1/stats", headers={"X-API-Key": "wrong-key-000"}).status_code,
                    401)
                ok = client.get("/api/v1/stats", headers={"X-API-Key": VALID_KEY})
                self.assertEqual(ok.status_code, 200)

    def test_analyze_and_request_id(self):
        with tempfile.TemporaryDirectory() as folder:
            app, _ = _app(folder, api_keys={VALID_KEY: "acme"})
            with TestClient(app) as client:
                response = client.post("/api/v1/analyze",
                                       headers={"X-API-Key": VALID_KEY,
                                                "X-Request-ID": "req-abc"},
                                       json={"repo": "example/tool", "issue_number": 17,
                                             "user_context": "Python 3.10 / Windows"})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.headers["X-Request-ID"], "req-abc")
                body = response.json()
                self.assertEqual(body["status"], "succeeded")
                self.assertIn("排障报告", body["report"])
                self.assertTrue(body["citations"])

    def test_invalid_payload_is_422(self):
        with tempfile.TemporaryDirectory() as folder:
            app, _ = _app(folder, api_keys={VALID_KEY: "acme"})
            with TestClient(app) as client:
                response = client.post("/api/v1/analyze", headers={"X-API-Key": VALID_KEY},
                                       json={"repo": "x", "issue_number": 0})
                self.assertEqual(response.status_code, 422)

    def test_rate_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            app, _ = _app(folder, api_keys={VALID_KEY: "acme"}, rate_limit_per_min=1)
            with TestClient(app) as client:
                first = client.post("/api/v1/analyze", headers={"X-API-Key": VALID_KEY},
                                    json={"repo": "example/tool", "issue_number": 17})
                self.assertEqual(first.status_code, 200)
                second = client.post("/api/v1/analyze", headers={"X-API-Key": VALID_KEY},
                                     json={"repo": "example/tool", "issue_number": 17})
                self.assertEqual(second.status_code, 429)
                self.assertEqual(second.json()["error"]["code"], "rate_limited")

    def test_ui_has_byok_controls(self):
        html = (Path(__file__).resolve().parent.parent / "issue_agent" / "index.html").read_text(
            encoding="utf-8")
        self.assertIn('id="provider"', html)
        self.assertIn("payload.llm", html)
        self.assertIn("不保存", html)

    def test_meta_exposes_provider_presets(self):
        with tempfile.TemporaryDirectory() as folder:
            app, _ = _app(folder, api_keys={VALID_KEY: "acme"})
            with TestClient(app) as client:
                meta = client.get("/api/v1/meta").json()
        self.assertEqual(meta["default_provider"], "deepseek")
        names = {item["name"] for item in meta["providers"]}
        self.assertTrue({"deepseek", "openai", "anthropic", "ollama"} <= names)
        self.assertNotIn("sk-", json.dumps(meta))

    def test_byok_override_is_used_and_key_not_echoed(self):
        secret = "sk-byok-test-secret-1234567890"
        with tempfile.TemporaryDirectory() as folder:
            app, _ = _app(folder, api_keys={VALID_KEY: "acme"})
            with TestClient(app) as client:
                response = client.post(
                    "/api/v1/analyze", headers={"X-API-Key": VALID_KEY},
                    json={"repo": "example/tool", "issue_number": 17,
                          "llm": {"provider": "openai", "api_key": secret,
                                  "model": "gpt-4o-mini"}})
                self.assertEqual(response.status_code, 200, response.text)
                text = response.text
                body = response.json()
        self.assertEqual(body["metrics"]["provider"], "openai")
        self.assertEqual(body["metrics"]["model"], "gpt-4o-mini")
        self.assertNotIn(secret, text)
        self.assertNotIn("sk-", text)

    def test_byok_requires_key_when_server_has_none(self):
        with tempfile.TemporaryDirectory() as folder:
            app, runner = _app(folder, api_keys={VALID_KEY: "acme"})
            # 模拟"真实客户端"（非测试注入），触发 Key 预校验
            with patch.object(type(runner), "llm_is_injected",
                              new_callable=PropertyMock, return_value=False):
                with TestClient(app) as client:
                    response = client.post(
                        "/api/v1/analyze", headers={"X-API-Key": VALID_KEY},
                        json={"repo": "example/tool", "issue_number": 17,
                              "llm": {"provider": "openai"}})
        self.assertEqual(response.status_code, 400)
        self.assertIn("需要 API Key", response.json()["error"]["message"])

    def test_unknown_provider_returns_400(self):
        with tempfile.TemporaryDirectory() as folder:
            app, _ = _app(folder, api_keys={VALID_KEY: "acme"})
            with TestClient(app) as client:
                response = client.post(
                    "/api/v1/analyze", headers={"X-API-Key": VALID_KEY},
                    json={"repo": "example/tool", "issue_number": 17,
                          "llm": {"provider": "no-such-provider"}})
        self.assertEqual(response.status_code, 400)

    def test_analyze_reuses_result_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            app, _ = _app(folder, api_keys={VALID_KEY: "acme"})
            payload = {"repo": "example/tool", "issue_number": 17,
                       "user_context": "Python 3.10 / Windows"}
            with TestClient(app) as client:
                first = client.post("/api/v1/analyze", headers={"X-API-Key": VALID_KEY},
                                    json=payload).json()
                second = client.post("/api/v1/analyze", headers={"X-API-Key": VALID_KEY},
                                     json=payload).json()
            self.assertNotIn("cache", first)
            self.assertTrue(second["cache"]["hit"])
            self.assertEqual(first["run_id"], second["run_id"])
            self.assertEqual(first["report"], second["report"])

    def test_job_and_sse_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            app, _ = _app(folder, api_keys={VALID_KEY: "acme"})
            with TestClient(app) as client:
                created = client.post("/api/v1/jobs", headers={"X-API-Key": VALID_KEY},
                                      json={"repo": "example/tool", "issue_number": 17})
                self.assertEqual(created.status_code, 200, created.text)
                job_id = created.json()["job_id"]
                deadline = time.time() + 30
                job = {}
                while time.time() < deadline:
                    job = client.get(f"/api/v1/jobs/{job_id}",
                                     headers={"X-API-Key": VALID_KEY}).json()
                    if job["status"] in {"succeeded", "partial", "failed"}:
                        break
                    time.sleep(0.2)
                self.assertEqual(job["status"], "succeeded")
                self.assertIn("report", job["result"] or {})
                with client.stream("GET", f"/api/v1/jobs/{job_id}/events",
                                   headers={"X-API-Key": VALID_KEY}) as response:
                    self.assertEqual(response.status_code, 200)
                    text = "".join(response.iter_text())
                self.assertIn("event: span", text)
                self.assertIn("event: final", text)
                self.assertIn("event: end", text)

    def test_job_is_tenant_scoped(self):
        with tempfile.TemporaryDirectory() as folder:
            other_key = "other-api-key-98765"
            app, _ = _app(folder, api_keys={VALID_KEY: "acme", other_key: "globex"})
            with TestClient(app) as client:
                job_id = client.post("/api/v1/jobs", headers={"X-API-Key": VALID_KEY},
                                     json={"repo": "example/tool", "issue_number": 17}).json()["job_id"]
                stolen = client.get(f"/api/v1/jobs/{job_id}",
                                    headers={"X-API-Key": other_key})
                self.assertEqual(stolen.status_code, 404)
                # 等作业结束再退出，避免后台线程写已删除的临时数据库（Windows 会报错）。
                deadline = time.time() + 30
                while time.time() < deadline:
                    status = client.get(f"/api/v1/jobs/{job_id}",
                                        headers={"X-API-Key": VALID_KEY}).json()["status"]
                    if status in {"succeeded", "partial", "failed"}:
                        break
                    time.sleep(0.2)

    def test_sessions_and_feedback(self):
        with tempfile.TemporaryDirectory() as folder:
            app, runner = _app(folder, api_keys={VALID_KEY: "acme"})
            with TestClient(app) as client:
                result = client.post("/api/v1/analyze", headers={"X-API-Key": VALID_KEY},
                                     json={"repo": "example/tool", "issue_number": 17}).json()
                session_id = result["session_id"]
                session = client.get(f"/api/v1/sessions/{session_id}",
                                     headers={"X-API-Key": VALID_KEY})
                self.assertEqual(session.status_code, 200)
                listed = client.get("/api/v1/sessions", headers={"X-API-Key": VALID_KEY}).json()
                self.assertTrue(listed["sessions"])
                feedback = client.post("/api/v1/feedback", headers={"X-API-Key": VALID_KEY},
                                       json={"run_id": result["run_id"], "rating": 1,
                                             "comment": "准确"}).json()
                self.assertEqual(feedback["feedback"]["approval_rate"], 1.0)
                stats = client.get("/api/v1/stats", headers={"X-API-Key": VALID_KEY}).json()
                self.assertEqual(stats["memory"]["sessions"], 1)
                resumed = client.post(f"/api/v1/sessions/{session_id}/resume",
                                      headers={"X-API-Key": VALID_KEY})
                self.assertEqual(resumed.status_code, 200, resumed.text)
                self.assertEqual(resumed.json()["report"], result["report"])

    def test_unknown_session_404(self):
        with tempfile.TemporaryDirectory() as folder:
            app, _ = _app(folder, api_keys={VALID_KEY: "acme"})
            with TestClient(app) as client:
                response = client.get("/api/v1/sessions/nope",
                                      headers={"X-API-Key": VALID_KEY})
                self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
