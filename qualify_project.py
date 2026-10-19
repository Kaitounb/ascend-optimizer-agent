"""Independent confirmation after selection; no model requests and no concurrent NPU work."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from ascend_optimizer.engine import Engine, atomic_json, workspace_lock
from ascend_optimizer.statistics import summarize
from ascend_optimizer.tasks import digest


def main():
    project = Path(__file__).resolve().parent
    summary = {}
    for name in ("gdn_commit", "gdn_verify"):
        root = project / "runs-20261016" / ("unroll-" + name)
        result = json.loads((root / "result.json").read_text())
        if (
            result["status"] != "complete"
            or not result["independent_improvement_established"]
        ):
            summary[name] = {
                "qualified": False,
                "reason": "No completed independently improved selection",
            }
            continue
        with workspace_lock(root):
            engine = Engine(root, timeout=900)
            try:
                manifest, source = engine.candidate(engine.state["best"])
                confirmations = []
                for repeat in range(3):
                    measured = engine._benchmark(
                        engine.state["best"],
                        engine.state["baseline"],
                        f"qualification_repeat_{repeat}",
                    )
                    confirmations.append(measured)
                    print(
                        "QUALIFICATION",
                        name,
                        repeat,
                        measured["speedup_geomean"],
                        measured["accepted"],
                        flush=True,
                    )
                heldout_task = deepcopy(engine.task)
                heldout_task["benchmark_shapes"] = [
                    [b, 48, 128, 128, 4] for b in (2, 16)
                ]
                _, baseline = engine.candidate(engine.state["baseline"])
                raw = engine.worker.call(
                    {
                        "action": "benchmark",
                        "candidate": str(source),
                        "baseline": str(baseline),
                        "task": heldout_task,
                        "device": engine.device,
                    }
                )
                heldout = summarize(raw["cases"], heldout_task["acceptance"])
                atomic_json(
                    root / "heldout.json",
                    {
                        "task": heldout_task,
                        "task_sha256": digest(heldout_task),
                        "source_sha256": hashlib.sha256(
                            source.read_bytes()
                        ).hexdigest(),
                        "raw": raw,
                        "summary": heldout,
                    },
                )
                summary[name] = {
                    "qualified": all(c["accepted"] for c in confirmations)
                    and heldout["accepted"],
                    "config": manifest["config"],
                    "source_sha256": manifest["source_sha256"],
                    "confirmations": confirmations,
                    "heldout": heldout,
                }
            finally:
                engine.close()
        atomic_json(project / "runs-20261016/qualification.json", summary)
    atomic_json(project / "runs-20261016/qualification.json", summary)
    print("QUALIFICATION_FINISHED", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
