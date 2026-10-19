import ast
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ascend_optimizer.agent import run_agent, tool_schema
from ascend_optimizer.engine import Engine, initialize
from ascend_optimizer.gdn_templates import SOURCE_ROOT, kernel_source
from ascend_optimizer.llm import Client
from ascend_optimizer.tasks import default_task, render, validate_task
from test_contract import FakeWorker
from build_project_candidate import adapter, build, build_batch_compatibility


class GDNAdapterTests(unittest.TestCase):
    def test_batch_compatibility_keeps_memory_addresses_inside_branches(self):
        original, candidate = build_batch_compatibility()
        before = ast.parse(original)
        after = ast.parse(candidate)
        a = next(
            n
            for n in before.body
            if isinstance(n, ast.FunctionDef) and n.name == "_batch_commit"
        )
        b = next(
            n
            for n in after.body
            if isinstance(n, ast.FunctionDef) and n.name == "_batch_commit"
        )
        # State replay stays exactly unchanged; only conv address lowering changes.
        self.assertEqual(ast.dump(a.body[-2]), ast.dump(b.body[-2]))
        self.assertEqual(ast.dump(a.args), ast.dump(b.args))
        branch = b.body[-1].body[-1]
        self.assertIsInstance(branch, ast.If)
        for path in (branch.body, branch.orelse):
            calls = [
                n.func.attr
                for n in ast.walk(ast.Module(body=path, type_ignores=[]))
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            ]
            self.assertIn("load", calls)
            self.assertIn("store", calls)

    def test_project_export_preserves_public_apis_and_fallback_kernels(self):
        original, untouched = build({})
        self.assertEqual(original, untouched)
        _, tuned = build(
            {
                "gdn_commit": {"value_tile": 64, "unroll": True},
                "gdn_verify": {"value_tile": 32, "unroll": True},
            }
        )
        functions = {
            n.name: n for n in ast.parse(tuned).body if isinstance(n, ast.FunctionDef)
        }
        for node in ast.parse(original).body:
            if isinstance(node, ast.FunctionDef):
                if node.name in ("gdn_recover_commit", "gdn_recover_verify"):
                    self.assertEqual(
                        ast.dump(node.args), ast.dump(functions[node.name].args)
                    )
                else:
                    self.assertEqual(ast.dump(node), ast.dump(functions[node.name]))
        self.assertEqual(
            tuned.count(
                "window == 4 and heads == 48 and vd == 128 and kd == 128 and indices.numel() <= 32"
            ),
            2,
        )
        for op in ("gdn_verify", "gdn_commit"):
            compile(adapter(tuned, op), "project_adapter", "exec")

    def test_arithmetic_body_matches_frozen_project_for_every_candidate(self):
        original = ast.parse((SOURCE_ROOT / "gdn_recover.py").read_text())
        for operator, name in [
            ("gdn_commit", "_commit_kernel"),
            ("gdn_verify", "_verify_kernel"),
        ]:
            expected = next(
                n
                for n in original.body
                if isinstance(n, ast.FunctionDef) and n.name == name
            )
            for tile in (0, 8, 16, 32, 64, 128):
                candidate = ast.parse(render(operator, {"value_tile": tile}))
                actual = next(
                    n
                    for n in candidate.body
                    if isinstance(n, ast.FunctionDef) and n.name == name
                )
                self.assertEqual(
                    ast.dump(actual, include_attributes=False),
                    ast.dump(expected, include_attributes=False),
                )

    def test_contract_and_model_schema_accept_only_supported_geometry_and_config(self):
        commit_task = default_task("gdn_commit")
        commit_task["dtypes"] = ["bfloat16"]
        with self.assertRaises(ValueError):
            validate_task(commit_task)
        for operator in ("gdn_commit", "gdn_verify"):
            task = default_task(operator)
            validate_task(task)
            proposed = next(
                s for s in tool_schema(task) if s["name"] == "propose_candidate"
            )
            self.assertEqual(
                proposed["parameters"]["properties"]["config"]["required"],
                ["value_tile"],
            )
            task["benchmark_shapes"] = [[1, 48, 128, 2**20, 4]]
            with self.assertRaises(ValueError):
                validate_task(task)
            with self.assertRaises(ValueError):
                render(operator, {"value_tile": 64, "enable_fp_fusion": True})

    def test_short_window_unrolling_preserves_original_and_only_changes_loop_form(self):
        for operator, name in [
            ("gdn_commit", "_commit_kernel"),
            ("gdn_verify", "_verify_kernel"),
        ]:
            source = render(operator, {"value_tile": 64, "unroll": True})
            self.assertIn("if window <= 4 else", source)
            expected = ast.parse(kernel_source(name)).body[0]
            actual = next(
                n
                for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == name + "_unrolled"
            )
            actual.name = name
            for node in ast.walk(actual):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "static_range"
                ):
                    node.func = ast.Name(id="range", ctx=ast.Load())
            self.assertEqual(
                ast.dump(actual, include_attributes=False),
                ast.dump(expected, include_attributes=False),
            )
            with self.assertRaises(ValueError):
                render(operator, {"value_tile": 64, "unroll": 1})

    def test_early_model_prose_cannot_claim_complete(self):
        class ClientFixture:
            model = "fixture"
            api_mode = "chat"

            def turn(self, *args):
                return [], "done"

            def redact(self, s):
                return s

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            initialize(root, "vector_add")
            engine = Engine(root, worker=FakeWorker())
            with patch("sys.stdout"):
                result = run_agent(engine, ClientFixture(), max_steps=4)
            self.assertEqual(result["status"], "incomplete_early_stop")
            self.assertFalse(result["independent_improvement_established"])

    def test_private_key_file_and_request_options(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {}, clear=True):
            path = Path(d) / "key"
            path.write_text("fixture-private-key")
            path.chmod(0o600)
            client = Client(
                base_url="https://api.deepseek.com",
                model="deepseek-v4-flash",
                key_file=path,
                request_options={"max_tokens": 1024, "thinking": {"type": "disabled"}},
            )
            self.assertEqual(client.redact("fixture-private-key"), "[REDACTED]")
            path.chmod(0o644)
            with self.assertRaises(ValueError):
                Client(
                    base_url="https://api.deepseek.com",
                    model="deepseek-v4-flash",
                    key_file=path,
                )


if __name__ == "__main__":
    unittest.main()
