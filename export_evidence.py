"""Bundle only this project's experiment artifacts for download from JupyterLab."""

import hashlib
import json
from pathlib import Path
import zipfile


def main():
    project = Path(__file__).resolve().parent
    summary = {}
    for operator in ("vector_add", "rms_norm", "http_agent_npu"):
        root = project / "runs" / operator
        result = json.loads((root / "result.json").read_text())
        manifest = json.loads(
            (root / "candidates" / f"{result['best']}.json").read_text()
        )
        events = [
            json.loads(line)
            for line in (root / "trajectory.jsonl").read_text().splitlines()
        ]
        decisions = [r["result"] for r in events if r["event"] == "candidate_decision"]
        failures = []
        for row in events:
            if row["event"] == "tool_call" and not row["result"]["ok"]:
                error = row["result"].get("error", "")
                try:
                    error = json.loads(error).get("error", error)
                except (ValueError, AttributeError):
                    pass
                failures.append(
                    {
                        "tool": row["tool"],
                        "arguments": row["arguments"],
                        "error": error[-2500:],
                    }
                )
        summary[operator] = {
            "best_config": manifest["config"],
            "result": result,
            "decisions": decisions,
            "failures": failures,
        }
    (project / "runs" / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    target = project.parent / "ascend-optimizer-evidence-20261015.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in (project / "runs").rglob("*"):
            if (
                path.is_file()
                and path.name != ".lock"
                and "__pycache__" not in path.parts
            ):
                archive.write(path, str(path.relative_to(project)))
        for name in (
            "npu_validation.log",
            "delivery_validation.log",
            "final_unit_tests.log",
        ):
            path = project / name
            if path.exists():
                archive.write(path, name)
    print(
        "EVIDENCE",
        target.name,
        target.stat().st_size,
        hashlib.sha256(target.read_bytes()).hexdigest(),
    )
    for name, value in summary.items():
        result = value["result"]
        print(
            "RESULT",
            name,
            "config",
            value["best_config"],
            "speedup",
            result["final_comparison"]["speedup_geomean"],
            "profile_ok",
            result.get("profile", {}).get("ok"),
            "decisions",
            len(value["decisions"]),
            "failures",
            len(value["failures"]),
        )


if __name__ == "__main__":
    main()
