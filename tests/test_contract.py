import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ascend_optimizer.engine import Engine, initialize
from ascend_optimizer.statistics import summarize
from ascend_optimizer.tasks import default_task, render, validate_task


def evidence(ratio=1.25):
    return {
        "correctness": {"baseline": {"passed": True}, "candidate": {"passed": True}},
        "cases": [
            {
                "case": {"shape": [65537], "dtype": "float32"},
                "pairs": [
                    {"baseline_us": 10.0, "candidate_us": 10.0 / ratio}
                    for _ in range(7)
                ],
            }
        ],
    }


class FakeWorker:
    def __init__(self):
        self.fail_compile = False

    def call(self, request):
        if request["action"] == "compile":
            if self.fail_compile:
                raise RuntimeError("compiler failure")
            return {"compiled_and_launched": True}
        if request["action"] == "validate":
            return {"passed": True}
        if request["action"] == "benchmark":
            return evidence()
        return {}

    def close(self):
        pass


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        initialize(self.root, "vector_add")
        self.worker = FakeWorker()
        self.engine = Engine(self.root, worker=self.worker)
        self.cid = self.engine.propose_candidate(
            {"block": 1024, "unroll": 1}, "larger tile"
        )["candidate_id"]

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, cid):
        self.engine.compile_kernel(cid)
        self.engine.validate_kernel(cid)
        return self.engine.bench_kernel(cid)

    def test_cannot_promote_without_real_pipeline(self):
        with self.assertRaises(ValueError):
            self.engine.accept_candidate(self.cid)
        self.engine.compile_kernel(self.cid)
        with self.assertRaises(ValueError):
            self.engine.bench_kernel(self.cid)
        self.prepare(self.cid)
        result = self.engine.accept_candidate(self.cid)
        self.assertTrue(result["accepted"])
        self.assertEqual(
            (self.root / "best.py").read_text(),
            render("vector_add", {"block": 1024, "unroll": 1}),
        )
        with self.assertRaises(ValueError):
            self.engine.accept_candidate(self.cid)

    def test_model_cannot_replace_candidate_before_decision(self):
        self.prepare(self.cid)
        with patch("sys.stdout", new=io.StringIO()):
            blocked = self.engine.call(
                "propose_candidate",
                {
                    "config": {"block": 4096, "unroll": 1},
                    "reason": "skip prior decision",
                },
            )
            self.assertFalse(blocked["ok"])
            self.assertIn("pending", blocked["error"])
            accepted = self.engine.call("accept_candidate", {"candidate_id": self.cid})
            self.assertTrue(accepted["ok"])
            allowed = self.engine.call(
                "propose_candidate",
                {"config": {"block": 4096, "unroll": 1}, "reason": "after decision"},
            )
            self.assertTrue(allowed["ok"])

    def test_source_tampering_invalidates_evidence(self):
        self.prepare(self.cid)
        (self.root / "candidates" / f"{self.cid}.py").write_text("# forged output")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.engine.accept_candidate(self.cid)

    def test_changed_incumbent_requires_fresh_pairing(self):
        cid2 = self.engine.propose_candidate(
            {"block": 4096, "unroll": 1}, "another tile"
        )["candidate_id"]
        self.prepare(self.cid)
        self.prepare(cid2)
        self.engine.accept_candidate(self.cid)
        with self.assertRaisesRegex(ValueError, "stale"):
            self.engine.accept_candidate(cid2)

    def test_resume_preserves_best_but_discards_benchmark_authority(self):
        self.prepare(self.cid)
        self.engine.accept_candidate(self.cid)
        resumed = Engine(self.root, worker=FakeWorker())
        self.assertEqual(resumed.state["best"], self.cid)
        with self.assertRaises(ValueError):
            resumed.accept_candidate(self.cid)
        with self.assertRaises(FileExistsError):
            initialize(self.root, "vector_add")

    def test_recompile_failure_clears_previous_validation(self):
        self.prepare(self.cid)
        self.worker.fail_compile = True
        with self.assertRaises(RuntimeError):
            self.engine.compile_kernel(self.cid)
        with self.assertRaises(ValueError):
            self.engine.bench_kernel(self.cid)
        with self.assertRaises(ValueError):
            self.engine.accept_candidate(self.cid)

    def test_task_edits_fail_closed(self):
        task = default_task("vector_add")
        task["benchmark_shapes"] = [[32]]
        (self.root / "task.json").write_text(json.dumps(task))
        with self.assertRaisesRegex(ValueError, "contract changed"):
            Engine(self.root, worker=FakeWorker())

    def test_unknown_tool_or_invalid_config_is_not_executed(self):
        with patch("sys.stdout", new=io.StringIO()):
            result = self.engine.call("__getattribute__", {"name": "worker"})
        self.assertFalse(result["ok"])
        with self.assertRaises(ValueError):
            self.engine.propose_candidate(
                {"block": 1024, "unroll": 1, "source": "print(0)"}, "test"
            )


class StatisticsTests(unittest.TestCase):
    def test_strong_win_promotes_and_regression_is_rejected(self):
        policy = default_task("vector_add")["acceptance"]
        cases = evidence()["cases"]
        self.assertTrue(summarize(cases, policy)["accepted"])
        bad_case = copy.deepcopy(cases[0])
        bad_case["case"] = {"shape": [1048576], "dtype": "float32"}
        for pair in bad_case["pairs"]:
            pair["candidate_us"] = 11
        verdict = summarize(cases + [bad_case], policy)
        self.assertFalse(verdict["accepted"])
        self.assertEqual(verdict["reason"], "case_regression")

    def test_nan_zero_negative_and_missing_samples_rejected(self):
        policy = default_task("vector_add")["acceptance"]
        for value in (0, -1, float("nan"), float("inf")):
            cases = evidence()["cases"]
            cases[0]["pairs"][0]["candidate_us"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                summarize(cases, policy)
        cases = evidence()["cases"]
        cases[0]["pairs"] = cases[0]["pairs"][:3]
        with self.assertRaises(ValueError):
            summarize(cases, policy)

    def test_contract_rejects_unbounded_allocation(self):
        task = default_task("rms_norm")
        task["benchmark_shapes"] = [[1, 2**30]]
        with self.assertRaises(ValueError):
            validate_task(task)


if __name__ == "__main__":
    unittest.main()
