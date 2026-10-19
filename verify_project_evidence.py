"""Offline integrity/statistics verification; does not import torch or call the API."""

import json
import math
from pathlib import Path
import sys

from ascend_optimizer.statistics import summarize
from ascend_optimizer.tasks import digest, render


def main():
    root = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parent / "evidence/gdn-20261016"
    )
    totals = {
        "workspaces": 0,
        "evaluations": 0,
        "real_api_turns": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    for path in sorted(root.rglob("task.json")):
        workspace = path.parent
        state = json.loads((workspace / "state.json").read_text())
        task = json.loads(path.read_text())
        assert digest(task) == state["task_sha256"], workspace
        for manifest_path in (workspace / "candidates").glob("*.json"):
            manifest = json.loads(manifest_path.read_text())
            source = manifest_path.with_suffix(".py").read_text()
            assert (
                digest(source)
                == manifest["source_sha256"]
                == digest(render(task["operator"], manifest["config"]))
            )
            assert digest(source)[:16] == manifest["id"]
        for file in (workspace / "evaluations").glob("*.json"):
            evidence = json.loads(file.read_text())
            assert evidence["task_sha256"] == state["task_sha256"]
            stats = summarize(evidence["raw"]["cases"], task["acceptance"])
            assert math.isclose(
                stats["speedup_geomean"],
                evidence["summary"]["speedup_geomean"],
                rel_tol=1e-12,
            )
            assert stats["accepted"] == evidence["summary"]["accepted"]
            assert all(row["passed"] for row in evidence["raw"]["correctness"].values())
            totals["evaluations"] += 1
        for file in workspace.glob("conversation-*.jsonl"):
            for line in file.read_text().splitlines():
                api = json.loads(line).get("api", {})
                if api.get("returned_model"):
                    totals["real_api_turns"] += 1
                    usage = api.get("usage") or {}
                    totals["input_tokens"] += usage.get(
                        "prompt_tokens", usage.get("input_tokens", 0)
                    )
                    totals["output_tokens"] += usage.get(
                        "completion_tokens", usage.get("output_tokens", 0)
                    )
        if (workspace / "result.json").exists():
            result = json.loads((workspace / "result.json").read_text())
            assert result["best"] == state["best"]
            assert (workspace / "best.py").read_bytes() == (
                workspace / "candidates" / (state["best"] + ".py")
            ).read_bytes()
            print(
                workspace.name,
                result["status"],
                result["final_comparison"]["speedup_geomean"],
                result["independent_improvement_established"],
            )
        totals["workspaces"] += 1
    assert totals["workspaces"], "No workspaces found"
    print("OFFLINE_PROJECT_EVIDENCE_PASS", json.dumps(totals))


if __name__ == "__main__":
    main()
