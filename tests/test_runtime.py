import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from ascend_optimizer.runtime import Worker
from ascend_optimizer.worker import summarize_trace


class RuntimeTests(unittest.TestCase):
    def test_timeout_terminates_process_and_keeps_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            best = root / "best.py"
            best.write_text("# retained")
            worker = Worker(root, timeout=0.05)
            worker.process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import sys,time; sys.stdin.readline(); time.sleep(60)",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                bufsize=0,
            )
            process = worker.process
            start = time.monotonic()
            try:
                with patch.object(worker, "start"), self.assertRaises(TimeoutError):
                    worker.call({"action": "fixture"})
                self.assertIsNotNone(process.poll())
                self.assertLess(time.monotonic() - start, 4)
                self.assertEqual(best.read_text(), "# retained")
            finally:
                worker.close()

    def test_custom_key_environment_is_removed_by_value(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = Worker(Path(directory))
            worker.exclude_secret("fixture-custom-model-secret")
            with patch.dict(
                os.environ, {"CUSTOM_MODEL_ACCESS": "fixture-custom-model-secret"}
            ):
                with patch("ascend_optimizer.runtime.subprocess.Popen") as spawn:
                    spawn.return_value.poll.return_value = 0
                    worker.start()
                    self.assertNotIn(
                        "CUSTOM_MODEL_ACCESS", spawn.call_args.kwargs["env"]
                    )
                    worker.close()

    def test_profiler_trace_summary_does_not_need_key_averages(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            {"ph": "M", "name": "process_name"},
                            {
                                "ph": "X",
                                "cat": "Kernel",
                                "name": "add",
                                "dur": 12,
                                "pid": 3,
                            },
                            {
                                "ph": "X",
                                "cat": "cpu_op",
                                "name": "launch",
                                "dur": 24,
                                "pid": 2,
                            },
                        ]
                    }
                )
            )
            result = summarize_trace(path)
            self.assertEqual(result["categories"]["Kernel"], 1)
            self.assertEqual(result["longest_complete_events"][0]["name"], "launch")
            self.assertEqual(result["events"], 3)


if __name__ == "__main__":
    unittest.main()
