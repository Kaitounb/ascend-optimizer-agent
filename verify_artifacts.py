"""Verify downloaded evidence without loading torch or executing candidate code."""

import hashlib
import json
import math
from pathlib import Path
import shutil

from ascend_optimizer.statistics import summarize
from ascend_optimizer.tasks import digest, render


def main():
    project = Path(__file__).resolve().parent
    root = project / "evidence/910b4-20261015"
    data = json.loads((root / "runs/summary.json").read_text())
    for name, item in data.items():
        run = root / "runs" / name
        task = json.loads((run / "task.json").read_text())
        state = json.loads((run / "state.json").read_text())
        assert digest(task) == state["task_sha256"]
        assert (run / "best.py").read_text() == render(
            task["operator"], item["best_config"]
        )
        evaluation = json.loads(
            (
                run
                / "evaluations"
                / (item["result"]["final_comparison"]["evaluation_id"] + ".json")
            ).read_text()
        )
        stats = summarize(evaluation["raw"]["cases"], task["acceptance"])
        assert math.isclose(
            stats["speedup_geomean"],
            item["result"]["final_comparison"]["speedup_geomean"],
            rel_tol=1e-12,
        )
        assert stats["accepted"] == item["result"]["final_comparison"]["accepted"]
        for manifest_path in (run / "candidates").glob("*.json"):
            manifest = json.loads(manifest_path.read_text())
            assert (
                digest(manifest_path.with_suffix(".py").read_text())
                == manifest["source_sha256"]
            )
        if name != "http_agent_npu":
            profile = item["result"]["profile"]["result"]
            trace = run / "profiles" / Path(profile["trace"]).parent.name / "trace.json"
            payload = json.loads(trace.read_text())
            assert payload if isinstance(payload, list) else payload["traceEvents"]
            example = project / "examples/verified-910b4" / name
            example.mkdir(parents=True, exist_ok=True)
            for file in (
                "REPORT.md",
                "best.py",
                "task.json",
                "result.json",
                "environment.json",
            ):
                shutil.copyfile(run / file, example / file)
            shutil.copyfile(
                run / "evaluations" / (evaluation["evaluation_id"] + ".json"),
                example / "final_evaluation.json",
            )
        print(name, "PASS", stats["speedup_geomean"], item["best_config"])
    assert json.loads((root / "runs/negative_check/result.json").read_text())[
        "incorrect_candidate_rejected"
    ]
    protocol = json.loads(
        (root / "runs/http_agent_npu/protocol_check.json").read_text()
    )
    assert protocol["passed"] and protocol["real_npu"] and protocol["model_is_fixture"]
    archive = root / "ascend-optimizer-evidence-20261015.zip"
    expected = "a8717d9f9eb8993b308d3dbdb72dacf76a4a418459f3aabd34cb68e125750728"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected
    print("OFFLINE_EVIDENCE_VERIFIED", expected)


if __name__ == "__main__":
    main()
