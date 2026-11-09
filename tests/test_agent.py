"""Agent 编排测试：图路由、预算、护栏、引用校验、自纠错、记忆、检查点续跑。

全部离线：用 FakeGitHub + HeuristicLLM / ScriptedLLM，不访问网络、不需要 API Key。
"""
from __future__ import annotations

import re
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from issue_agent.agent import AgentRunner
from issue_agent.budget import Budget
from issue_agent.config import Settings
from issue_agent.errors import BudgetExceeded, LLMError, LLMOutputError
from issue_agent.llm import HeuristicLLM, LLMClient, ScriptedLLM
from issue_agent.llm.schemas import ActionDecision, Assessment, Critique, DraftReport, SearchPlan
from issue_agent.memory.store import MemoryStore
from issue_agent.results import SingleFlight, analysis_key
from issue_agent.testing import FakeGitHub, make_settings, sample_store

REPO = "example/tool"
ISSUE = 17


class StrictFakeGitHub(FakeGitHub):
    """模拟真实 GitHub 搜索：自然语言整句召回为 0，只有错误码/技术词才命中。"""

    def search_issues(self, repo, query, limit=8):
        if not re.search(r"readtimeout|elasticsearch", query, re.I):
            return {"items": [], "query": query, "total": 0}
        return super().search_issues(repo, query, limit)


def _runner(folder: str, llm, github=None, **overrides) -> AgentRunner:
    settings = make_settings(folder, **overrides)
    return AgentRunner(settings, llm=llm, github=github or FakeGitHub())


class TestAgentGraph(unittest.TestCase):
    def test_happy_path(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = _runner(folder, HeuristicLLM())
            try:
                result = runner.run(REPO, ISSUE, tenant="t1")
            finally:
                runner.close()
        self.assertEqual(result["status"], "succeeded")
        self.assertTrue(result["report"].startswith("# GitHub Issue 排障报告"))
        self.assertIn("免责声明", result["report"])
        self.assertNotIn("已修复", result["report"])
        self.assertTrue(result["citations"])
        nodes = [item["node"] for item in result["trace"]]
        self.assertEqual(nodes[:3], ["ingest", "plan", "act"])
        self.assertIn("verify", nodes)
        self.assertEqual(nodes[-1], "finalize")
        kinds = {doc["kind"] for doc in result["evidence"]}
        self.assertIn("target_issue", kinds)
        self.assertIn("readme", kinds)
        self.assertIn("comment", kinds)
        self.assertIn("related_issue", kinds)
        self.assertGreaterEqual(result["metrics"]["rounds"], 1)

    def test_injection_flagged_and_quarantined(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = _runner(folder, HeuristicLLM())
            try:
                result = runner.run(REPO, ISSUE, tenant="t1")
            finally:
                runner.close()
        self.assertTrue(any("可疑指令" in warning for warning in result["warnings"]))
        risky = [doc for doc in result["evidence"] if doc["risk"] >= 0.5]
        self.assertEqual(len(risky), 1)
        self.assertEqual(risky[0]["kind"], "comment")

    def test_second_round_after_assessment(self):
        script = {
            "SearchPlan": [{"queries": ["index timeout"], "focus": "timeout",
                            "checks": ["known workaround"], "reason": "test"}],
            "Assessment": [
                {"sufficient": False, "gaps": ["缺少配置项证据"],
                 "next_query": "SEARCH_TIMEOUT elasticsearch", "reason": "需要更多证据"},
                {"sufficient": True, "gaps": [], "next_query": "", "reason": "够了"},
            ],
            "ActionDecision": [
                {"thought": "搜索历史 Issue", "action": "search_issues",
                 "args": {"query": "index timeout", "limit": 5}},
                {"thought": "结束", "action": "finish", "args": {}},
                # 第三轮由 assess 强制触发（不再消耗 ActionDecision），这里只需再收尾一次
                {"thought": "结束", "action": "finish", "args": {}},
            ],
            "DraftReport": [{
                "summary": "可能是默认超时过短，需验证。",
                "hypotheses": [{"claim": "SEARCH_TIMEOUT 默认 30s 对 ES 7.10 偏短",
                                "citations": ["E1", "E3"], "confidence": "medium"}],
                "actions": [{"step": "把 SEARCH_TIMEOUT 调到 90s 后重试", "citations": ["E3"]}],
                "verification": ["重试后不再 ReadTimeout"], "gaps": [], "risk_notes": [],
            }],
        }
        with tempfile.TemporaryDirectory() as folder:
            runner = _runner(folder, ScriptedLLM(script), enable_llm_critique=False)
            try:
                result = runner.run(REPO, ISSUE, tenant="t1")
            finally:
                runner.close()
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["metrics"]["rounds"], 2)
        self.assertIn("SEARCH_TIMEOUT elasticsearch", result["queries"])
        self.assertIn("E3", [citation["id"] for citation in result["citations"]])

    def test_ingest_failure_degrades_without_exception(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = _runner(folder, HeuristicLLM(), FakeGitHub(sample_store()))
            try:
                result = runner.run("other/repo", 1, tenant="t1")
            finally:
                runner.close()
        self.assertEqual(result["status"], "failed")
        self.assertIn("资源不存在", result["error"])
        self.assertIn("排障分析未完成", result["report"])
        self.assertTrue(result["report"])

    def test_budget_exceeded_returns_partial(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = _runner(folder, HeuristicLLM(), max_llm_calls=1)
            try:
                result = runner.run(REPO, ISSUE, tenant="t1")
            finally:
                runner.close()
        self.assertEqual(result["status"], "partial")
        self.assertIn("llm_calls", result["metrics"]["exhausted"])
        self.assertTrue(result["report"])

    def test_hallucinated_citations_are_dropped(self):
        script = {
            "SearchPlan": [{"queries": ["index timeout"], "focus": "x", "checks": [],
                            "reason": ""}],
            "Assessment": [{"sufficient": True, "gaps": [], "next_query": "", "reason": ""}],
            "ActionDecision": [{"thought": "", "action": "finish", "args": {}}],
            "DraftReport": [{
                "summary": "结论（含幻觉引用）",
                "hypotheses": [{"claim": "无依据的结论", "citations": ["E999"], "confidence": "high"},
                               {"claim": "有依据的结论", "citations": ["E1"], "confidence": "low"}],
                "actions": [], "verification": [], "gaps": [], "risk_notes": [],
            }],
        }
        with tempfile.TemporaryDirectory() as folder:
            runner = _runner(folder, ScriptedLLM(script), enable_llm_critique=False,
                             max_rewrites=0)
            try:
                result = runner.run(REPO, ISSUE, tenant="t1")
            finally:
                runner.close()
        self.assertNotIn("E999", result["report"])
        self.assertIn("E1", result["report"])
        self.assertTrue(any("丢弃" in warning for warning in result["warnings"]))

    def test_unsupported_claim_triggers_one_rewrite(self):
        script = {
            "SearchPlan": [{"queries": ["index timeout"], "focus": "x", "checks": [],
                            "reason": ""}],
            "Assessment": [{"sufficient": True, "gaps": [], "next_query": "", "reason": ""}],
            "ActionDecision": [{"thought": "", "action": "finish", "args": {}}],
            "DraftReport": [
                {"summary": "第一版", "hypotheses": [
                    {"claim": "完全没有依据的结论", "citations": [], "confidence": "high"}],
                 "actions": [], "verification": [], "gaps": [], "risk_notes": []},
                {"summary": "第二版（已修正）", "hypotheses": [
                    {"claim": "有依据的结论", "citations": ["E1"], "confidence": "medium"}],
                 "actions": [{"step": "核对版本", "citations": ["E1"]}],
                 "verification": [], "gaps": [], "risk_notes": []},
            ],
        }
        with tempfile.TemporaryDirectory() as folder:
            runner = _runner(folder, ScriptedLLM(script), enable_llm_critique=False,
                             max_rewrites=1)
            try:
                result = runner.run(REPO, ISSUE, tenant="t1")
            finally:
                runner.close()
        self.assertEqual(result["status"], "succeeded")
        self.assertIn("第二版", result["report"])
        self.assertTrue(any("重写" in warning for warning in result["warnings"]))

    def test_unsafe_claim_rewritten_away(self):
        script = {
            "SearchPlan": [{"queries": ["q"], "focus": "x", "checks": [], "reason": ""}],
            "Assessment": [{"sufficient": True, "gaps": [], "next_query": "", "reason": ""}],
            "ActionDecision": [{"thought": "", "action": "finish", "args": {}}],
            "DraftReport": [{
                "summary": "问题已修复，我已运行测试。",
                "hypotheses": [{"claim": "该问题已修复", "citations": ["E1"], "confidence": "high"}],
                "actions": [], "verification": [], "gaps": [], "risk_notes": [],
            }],
        }
        with tempfile.TemporaryDirectory() as folder:
            runner = _runner(folder, ScriptedLLM(script), enable_llm_critique=False,
                             max_rewrites=0)
            try:
                result = runner.run(REPO, ISSUE, tenant="t1")
            finally:
                runner.close()
        self.assertNotIn("已修复", result["report"])
        self.assertNotIn("我已运行", result["report"])

    def test_search_failure_is_a_warning_not_a_crash(self):
        github = FakeGitHub()
        github.fail_search = True
        with tempfile.TemporaryDirectory() as folder:
            runner = _runner(folder, HeuristicLLM(), github)
            try:
                result = runner.run(REPO, ISSUE, tenant="t1")
            finally:
                runner.close()
        self.assertNotEqual(result["status"], "failed")
        self.assertTrue(any("搜索" in warning or "失败" in warning or "不可用" in warning
                            for warning in result["warnings"]))

    def test_session_and_checkpoint_persisted(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = _runner(folder, HeuristicLLM())
            try:
                result = runner.run(REPO, ISSUE, tenant="t1")
                checkpoint = runner.memory.latest_checkpoint(result["session_id"])
                session = runner.memory.load_session(result["session_id"], tenant="t1")
            finally:
                runner.close()
        self.assertIsNotNone(checkpoint)
        self.assertEqual(checkpoint["node"], "finalize")
        self.assertIsNotNone(session)
        self.assertEqual(session["status"], "succeeded")
        self.assertIn("report", session["payload"])

    def test_resume_completed_session_returns_stored_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = _runner(folder, HeuristicLLM())
            try:
                first = runner.run(REPO, ISSUE, tenant="t1")
                resumed = runner.resume(first["session_id"], tenant="t1")
            finally:
                runner.close()
        self.assertEqual(resumed["status"], "succeeded")
        self.assertEqual(resumed["report"], first["report"])

    def test_resume_after_budget_failure_completes(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(folder, max_llm_calls=1)
            first_runner = AgentRunner(settings, llm=HeuristicLLM(), github=FakeGitHub())
            try:
                first = first_runner.run(REPO, ISSUE, tenant="t1")
            finally:
                first_runner.close()
            self.assertEqual(first["status"], "partial")

            # 新进程/新 Runner：预算重新计算，但证据从检查点复用。
            bigger = settings.with_overrides(max_llm_calls=8)
            second_runner = AgentRunner(bigger, llm=HeuristicLLM(), github=FakeGitHub())
            try:
                resumed = second_runner.resume(first["session_id"], tenant="t1")
            finally:
                second_runner.close()
        self.assertIn(resumed["status"], {"succeeded", "partial"})
        self.assertGreater(len(resumed["evidence"]), 1)

    def test_episodic_memory_is_recalled_on_second_run(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(folder)
            runner = AgentRunner(settings, llm=HeuristicLLM(), github=FakeGitHub())
            try:
                runner.run(REPO, ISSUE, tenant="t1", use_cache=False)
                second = runner.run(REPO, ISSUE, tenant="t1", use_cache=False)
            finally:
                runner.close()
        memory_docs = [doc for doc in second["evidence"] if doc["kind"] == "memory"]
        self.assertTrue(memory_docs)
        self.assertIn("历史记忆", memory_docs[0]["title"])

    def test_tenant_isolation_of_memory(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(folder)
            runner = AgentRunner(settings, llm=HeuristicLLM(), github=FakeGitHub())
            try:
                first = runner.run(REPO, ISSUE, tenant="tenant-a")
                other = runner.run(REPO, ISSUE, tenant="tenant-b")
                stolen = runner.memory.load_session(first["session_id"], tenant="tenant-b")
            finally:
                runner.close()
        self.assertIsNone(stolen)
        self.assertFalse([doc for doc in other["evidence"] if doc["kind"] == "memory"])

    def test_missing_api_key_raises_config_error(self):
        """缺少 Key 是配置错误：必须上抛给 API 层，而不是伪装成"分析失败"。"""
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(folder, llm_api_key="")
            runner = AgentRunner(settings, github=FakeGitHub())  # 不注入 llm
            try:
                with self.assertRaises(LLMError):
                    runner.run(REPO, ISSUE, tenant="t1")
            finally:
                runner.close()


class TestResultReuse(unittest.TestCase):
    """公网部署时最常见的两个问题：重复请求与缓存击穿。"""

    def test_second_identical_run_hits_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = _runner(folder, HeuristicLLM())
            try:
                first = runner.run(REPO, ISSUE, tenant="t1")
                calls = runner._llm.calls
                second = runner.run(REPO, ISSUE, tenant="t1")
                third = runner.run(REPO, ISSUE, tenant="t1")
            finally:
                runner.close()
        self.assertTrue(second["cache"]["hit"])
        self.assertTrue(third["cache"]["hit"])
        self.assertGreaterEqual(second["cache"]["hits"], 1)
        self.assertEqual(runner._llm.calls, calls)  # 二次请求没有产生任何 LLM 调用
        self.assertEqual(first["report"], second["report"])

    def test_context_change_bypasses_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = _runner(folder, HeuristicLLM())
            try:
                runner.run(REPO, ISSUE, tenant="t1", user_context="Python 3.10")
                other = runner.run(REPO, ISSUE, tenant="t1", user_context="Node 20")
            finally:
                runner.close()
        self.assertNotIn("cache", other)

    def test_cache_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = _runner(folder, HeuristicLLM())
            try:
                runner.run(REPO, ISSUE, tenant="t1", use_cache=False)
                calls = runner._llm.calls
                second = runner.run(REPO, ISSUE, tenant="t1", use_cache=False)
            finally:
                runner.close()
        self.assertNotIn("cache", second)
        self.assertGreater(runner._llm.calls, calls)

    def test_partial_results_are_not_cached(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = _runner(folder, HeuristicLLM(), max_llm_calls=1)
            try:
                first = runner.run(REPO, ISSUE, tenant="t1")
                second = runner.run(REPO, ISSUE, tenant="t1")
            finally:
                runner.close()
        self.assertEqual(first["status"], "partial")
        # 降级结果不能被复用成"标准答案"
        self.assertNotIn("cache", second)

    def test_cache_key_binds_provider_and_model(self):
        base = dict(tenant="t", repo="a/b", issue_number=1, user_context="",
                    model="same-model", prompt_fingerprint="fp")
        k_deepseek = analysis_key(provider="deepseek", **base)
        k_openai = analysis_key(provider="openai", **base)
        k_other_model = analysis_key(provider="deepseek", **{**base, "model": "other"})
        self.assertNotEqual(k_deepseek, k_openai)
        self.assertNotEqual(k_deepseek, k_other_model)
        self.assertEqual(k_deepseek, analysis_key(provider="deepseek", **base))

    def test_llm_override_is_isolated_from_server_cache(self):
        override = {"provider": "openai", "api_key": "sk-test-override-123456",
                    "model": "gpt-4o-mini"}
        with tempfile.TemporaryDirectory() as folder:
            runner = _runner(folder, HeuristicLLM())
            try:
                first = runner.run(REPO, ISSUE, tenant="t1")
                second = runner.run(REPO, ISSUE, tenant="t1", llm_override=override)
                third = runner.run(REPO, ISSUE, tenant="t1", llm_override=override)
            finally:
                runner.close()
        self.assertEqual(first["metrics"]["provider"], "deepseek")
        self.assertEqual(second["metrics"]["provider"], "openai")
        self.assertEqual(second["metrics"]["model"], "gpt-4o-mini")
        self.assertNotIn("cache", second)       # 换模型不能复用服务器默认模型的结果
        self.assertTrue(third["cache"]["hit"])  # 同一 override 可复用

    def test_github_token_isolates_result_cache(self):
        token_a = {"token": "ghp_" + "a" * 36}
        token_b = {"token": "ghp_" + "b" * 36}
        with tempfile.TemporaryDirectory() as folder:
            runner = _runner(folder, HeuristicLLM())
            try:
                first = runner.run(REPO, ISSUE, tenant="t1", github_override=token_a)
                second = runner.run(REPO, ISSUE, tenant="t1", github_override=token_a)
                other = runner.run(REPO, ISSUE, tenant="t1", github_override=token_b)
            finally:
                runner.close()
        self.assertEqual(first["metrics"]["github_auth"], "user")
        self.assertTrue(second["cache"]["hit"])
        self.assertNotIn("cache", other)

    def test_cache_is_tenant_scoped(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = _runner(folder, HeuristicLLM())
            try:
                runner.run(REPO, ISSUE, tenant="tenant-a")
                other = runner.run(REPO, ISSUE, tenant="tenant-b")
                stats = runner.results.stats(tenant="tenant-a")
            finally:
                runner.close()
        self.assertNotIn("cache", other)
        self.assertEqual(stats["entries"], 1)


class TestSingleFlight(unittest.TestCase):
    def test_concurrent_calls_execute_once(self):
        flight = SingleFlight()
        calls = {"n": 0}
        results: list[dict] = []
        shared_flags: list[bool] = []

        def work() -> dict:
            calls["n"] += 1
            time.sleep(0.2)
            return {"value": 42}

        def call() -> None:
            result, shared = flight.do("same-key", work)
            results.append(result)
            shared_flags.append(shared)

        threads = [threading.Thread(target=call) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(calls["n"], 1)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(item == {"value": 42} for item in results))
        self.assertEqual(sum(1 for flag in shared_flags if flag), 2)

    def test_exception_is_propagated_to_waiters(self):
        flight = SingleFlight()

        def boom() -> dict:
            raise RuntimeError("upstream down")

        with self.assertRaises(RuntimeError):
            flight.do("k", boom)


class TestRetrievalFallback(unittest.TestCase):
    def test_empty_search_falls_back_to_error_codes(self):
        script = {
            "SearchPlan": [{"queries": ["index timeout"], "focus": "timeout",
                            "checks": [], "reason": ""}],
            "Assessment": [{"sufficient": True, "gaps": [], "next_query": "", "reason": ""}],
            "ActionDecision": [
                {"thought": "先按计划检索", "action": "search_issues",
                 "args": {"query": "index timeout", "limit": 5}},
                {"thought": "收尾", "action": "finish", "args": {}},
            ],
            "DraftReport": [{
                "summary": "回退检索后找到疑似相关历史 Issue。",
                "hypotheses": [{"claim": "可能是索引未重建", "citations": ["E1", "E6"],
                                "confidence": "low"}],
                "actions": [], "verification": [], "gaps": [], "risk_notes": [],
            }],
        }
        with tempfile.TemporaryDirectory() as folder:
            runner = _runner(folder, ScriptedLLM(script), StrictFakeGitHub(),
                             enable_llm_critique=False)
            try:
                result = runner.run(REPO, ISSUE, tenant="t1")
            finally:
                runner.close()
        self.assertEqual(result["status"], "succeeded")
        self.assertTrue(any("回退为错误码检索" in warning for warning in result["warnings"]))
        self.assertTrue(any(doc["kind"] == "related_issue" for doc in result["evidence"]))
        self.assertTrue(any("ReadTimeout" in query for query in result["queries"]))


class TestLLMClient(unittest.TestCase):
    def _settings(self, folder: str, **overrides) -> Settings:
        return make_settings(folder, llm_api_key="test-key", **overrides)

    def _client(self, folder: str, responses: list[SimpleNamespace], **overrides):
        settings = self._settings(folder, **overrides)
        calls = {"n": 0}

        class Completions:
            def create(self, **kwargs):  # noqa: ANN003
                index = min(calls["n"], len(responses) - 1)
                calls["n"] += 1
                return responses[index]

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        return LLMClient(settings, client=client), calls

    @staticmethod
    def _response(content: str, finish: str = "stop") -> SimpleNamespace:
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason=finish,
                                     message=SimpleNamespace(content=content, tool_calls=None))],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=10))

    def test_json_repair_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            llm, calls = self._client(folder, [
                self._response("这不是 JSON"),
                self._response('{"queries":["a"],"focus":"f","checks":[],"reason":""}'),
            ])
            plan = llm.chat_json("sys", "user", SearchPlan)
        self.assertEqual(plan.queries, ["a"])
        self.assertEqual(calls["n"], 2)
        self.assertEqual(llm.usage.calls, 2)
        self.assertGreater(llm.usage.prompt_tokens, 0)

    def test_truncated_output_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            llm, _ = self._client(folder, [self._response('{"queries":["a"]', "length")])
            with self.assertRaises(LLMOutputError):
                llm.chat_json("sys", "user", SearchPlan, max_tokens=10)

    def test_budget_blocks_before_http(self):
        with tempfile.TemporaryDirectory() as folder:
            llm, calls = self._client(folder, [self._response("{}")])
            llm.budget = Budget(max_llm_calls=0)
            with self.assertRaises(BudgetExceeded):
                llm.chat_json("sys", "user", SearchPlan)
            self.assertEqual(calls["n"], 0)

    def test_missing_key_fails_fast(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(folder, llm_api_key="")
            with self.assertRaises(LLMError):
                LLMClient(settings)

    def test_action_protocol_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            llm, _ = self._client(folder, [
                self._response('{"thought":"x","action":"delete_repo","args":{}}')])
            with self.assertRaises(LLMOutputError):
                llm.chat_action("sys", "user", tool_specs=[], allowed={"search_issues"})


class TestMemoryStore(unittest.TestCase):
    def test_episode_recall_and_feedback(self):
        with tempfile.TemporaryDirectory() as folder:
            store = MemoryStore(Path(folder) / "m.db")
            store.remember_episode(run_id="r1", tenant="t", repo="a/b", issue_number=7,
                                   status="succeeded",
                                   summary="ReadTimeout 由 SEARCH_TIMEOUT 过短导致",
                                   root_cause="默认超时 30s", confidence=0.7,
                                   sources=[{"id": "E1", "url": "https://github.com/a/b/issues/7"}])
            recalled = store.recall_episodes(tenant="t", repo="a/b",
                                             query="ReadTimeout SEARCH_TIMEOUT")
            self.assertEqual(len(recalled), 1)
            self.assertEqual(recalled[0]["issue_number"], 7)
            self.assertEqual(store.recall_episodes(tenant="other", repo="a/b",
                                                   query="ReadTimeout"), [])
            store.record_feedback(run_id="r1", tenant="t", rating=1, comment="ok")
            stats = store.feedback_stats(tenant="t")
            self.assertEqual(stats["approval_rate"], 1.0)

    def test_fact_upsert_and_recall(self):
        with tempfile.TemporaryDirectory() as folder:
            store = MemoryStore(Path(folder) / "m.db")
            store.remember_fact(tenant="t", key="a/b 需要 Node 18+", value="node18",
                                confidence=0.8)
            store.remember_fact(tenant="t", key="a/b 需要 Node 18+", value="node20",
                                confidence=0.9)
            facts = store.recall_facts(tenant="t", query="Node")
            self.assertEqual(len(facts), 1)
            self.assertEqual(facts[0]["value"], "node20")


if __name__ == "__main__":
    unittest.main()
