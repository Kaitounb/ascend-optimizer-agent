"""Hardware integration checks. Run inside the supplied Ascend notebook image."""

import json
from pathlib import Path
import sys
import traceback

from ascend_optimizer.agent import search
from ascend_optimizer.engine import Engine, atomic_json, initialize, workspace_lock
from ascend_optimizer.runtime import Worker
from ascend_optimizer.tasks import default_task, render


def main():
    root = Path(__file__).resolve().parent
    for name in ("vector_add", "rms_norm"):
        run = root / "runs" / name
        with workspace_lock(run):
            if not (run / "task.json").exists():
                initialize(run, name)
            engine = Engine(run, timeout=300)
            try:
                result = search(engine, max_candidates=6, profile=True)
                print("OPERATOR_RESULT", name, json.dumps(result), flush=True)
            finally:
                engine.close()
    # Real correctness failure must be rejected, not just a mocked assertion.
    negative = root / "runs" / "negative_check"
    negative.mkdir(parents=True, exist_ok=True)
    bad = negative / "wrong_add.py"
    bad.write_text(
        render("vector_add", {"block": 1024, "unroll": 1}).replace("a + b", "a - b")
    )
    task = default_task("vector_add")
    worker = Worker(negative, timeout=300)
    try:
        try:
            worker.call(
                {
                    "action": "validate",
                    "task": task,
                    "candidate": str(bad),
                    "device": "npu:0",
                }
            )
        except RuntimeError as exc:
            atomic_json(
                negative / "result.json",
                {"incorrect_candidate_rejected": True, "error": str(exc)},
            )
            print("NEGATIVE_CHECK_PASS", flush=True)
        else:
            raise AssertionError("Incorrect kernel was not rejected")
    finally:
        worker.close()
    print("NPU_INTEGRATION_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
