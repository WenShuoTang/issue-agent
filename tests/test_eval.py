"""评测框架测试：指标数学、离线执行、回归门禁。"""
from __future__ import annotations

import tempfile
import unittest

from issue_agent.eval.dataset import offline_cases
from issue_agent.eval.metrics import aggregate, evaluate_case, mrr, recall_at_k
from issue_agent.eval.runner import run_offline
from issue_agent.testing import make_settings


class TestMetrics(unittest.TestCase):
    def test_recall_and_mrr(self):
        self.assertEqual(recall_at_k([1, 2, 3, 4, 5, 6], [3, 9], k=5), 0.5)
        self.assertEqual(recall_at_k([9, 3], [3], k=5), 1.0)
        self.assertEqual(mrr([9, 8, 3], [3]), round(1 / 3, 4))
        self.assertEqual(mrr([9, 8], [3]), 0.0)
        self.assertIsNone(recall_at_k([1], []))
        self.assertIsNone(mrr([1], []))

    def test_aggregate_gate(self):
        cases = [
            {"status": "succeeded", "recall_at_5": 1.0, "mrr": 1.0,
             "citation_in_evidence": 1.0, "citation_coverage": 1.0, "tool_calls": 3,
             "retrieval_rounds": 1, "tokens": 100, "duration_ms": 500, "relevant": [1]},
            {"status": "succeeded", "recall_at_5": 0.0, "mrr": 0.0,
             "citation_in_evidence": 1.0, "citation_coverage": 0.5, "tool_calls": 4,
             "retrieval_rounds": 2, "tokens": 200, "duration_ms": 900, "relevant": [2]},
        ]
        summary = aggregate(cases, min_recall=0.5, min_citation=0.9)
        self.assertFalse(summary["gate"]["passed"])
        self.assertEqual(summary["recall_at_5_mean"], 0.5)
        self.assertEqual(summary["latency_p95_ms"], 900)
        summary_ok = aggregate(cases, min_recall=0.0, min_citation=0.9)
        self.assertTrue(summary_ok["gate"]["passed"])

    def test_evaluate_case_handles_missing_scores(self):
        case = next(case for case in offline_cases() if case.case_id == "no-answer-004")
        result = {"status": "succeeded", "evidence": [], "citations": [], "draft": {},
                  "metrics": {"rounds": 1, "tool_calls": 2, "duration_ms": 10},
                  "report": "x", "warnings": [], "retrieval": []}
        record = evaluate_case(case, result)
        self.assertIsNone(record["recall_at_5"])
        self.assertEqual(record["status"], "succeeded")


class TestOfflineRunner(unittest.TestCase):
    def test_offline_run_passes_and_writes_report(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(folder)
            report = run_offline(settings=settings, emit=lambda *_: None)
            latest = settings.data_dir / "eval" / "latest.json"
            self.assertTrue(latest.exists())
            self.assertEqual(len(report["cases"]), 4)
            self.assertTrue(report["passed"])
            self.assertGreaterEqual(report["aggregate"]["recall_at_5_mean"], 0.5)
            self.assertEqual(report["aggregate"]["citation_validity_mean"], 1.0)
            # 合成数据上的离线评测必须明确标注，不能冒充真实效果
            self.assertIn("合成", report["dataset"]["note"])
            self.assertIn("HeuristicLLM", report["model"])

    def test_limit_and_gate_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(folder)
            report = run_offline(settings=settings, limit=1, min_recall=1.5,
                                 emit=lambda *_: None)
            self.assertEqual(len(report["cases"]), 1)
            self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
