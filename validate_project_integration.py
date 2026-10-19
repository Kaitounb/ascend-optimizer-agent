"""Measure exported public APIs and run the frozen project regression suite serially."""

import json
from pathlib import Path
import subprocess
import sys

from ascend_optimizer.engine import atomic_json
from ascend_optimizer.runtime import Worker
from ascend_optimizer.statistics import summarize


def main():
    project = Path(__file__).resolve().parent
    root = project / "runs-20261016"
    out = root / "project-integration"
    configs = json.loads((out / "selection.json").read_text())["configs"]
    results = {"configs": configs, "public_api_benchmarks": {}}
    worker = Worker(out, timeout=900)
    try:
        for operator in configs:
            task = json.loads((root / ("unroll-" + operator) / "task.json").read_text())
            raw = worker.call(
                {
                    "action": "benchmark",
                    "candidate": str(out / (operator + "_candidate.py")),
                    "baseline": str(out / (operator + "_baseline.py")),
                    "task": task,
                    "device": "npu:0",
                }
            )
            summary = summarize(raw["cases"], task["acceptance"])
            atomic_json(
                out / (operator + "_public_api_benchmark.json"),
                {"task": task, "raw": raw, "summary": summary},
            )
            results["public_api_benchmarks"][operator] = summary
            print(
                "PUBLIC_API_BENCHMARK",
                operator,
                summary["speedup_geomean"],
                summary["accepted"],
                flush=True,
            )
    finally:
        worker.close()
    with (project / "project_regression_candidate.log").open("w") as log:
        code = subprocess.run(
            [
                sys.executable,
                "run_project_regression.py",
                str(out / "gdn_recover.py"),
                str(out / "gdn_recover_batch.py"),
            ],
            cwd=project,
            stdout=log,
            stderr=subprocess.STDOUT,
        ).returncode
    results["original_project_tests_exit"] = code
    results["release_ready"] = code == 0 and all(
        row["accepted"] for row in results["public_api_benchmarks"].values()
    )
    atomic_json(out / "validation.json", results)
    print("PROJECT_INTEGRATION_FINISHED", json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
