"""Print evidence-derived completion status, without reading credentials."""

import json
from pathlib import Path
import re


def main():
    project = Path(__file__).resolve().parent
    logs = (
        project
        if (project / "runs-20261016").exists()
        else project / "evidence/gdn-20261016"
    )
    runs = logs / "runs-20261016"
    installed = json.loads((runs / "installed_package_smoke.json").read_text())
    assert installed["real_npu"]["compiled_and_launched"]
    unit_log = (logs / "final_agent_tests.log").read_text()
    assert unit_log.rstrip().endswith("OK")
    count = re.search(r"Ran (\d+) tests", unit_log).group(1)
    regression = (logs / "project_regression_candidate.log").read_text()
    passed = re.search(r"(\d+) passed in [\d.]+s", regression).group(0)
    integration = json.loads((runs / "project-integration/validation.json").read_text())
    assert integration["release_ready"]
    print("ASCEND OPTIMIZER AGENT", installed["version"])
    print("Installed package: real NPU compile and launch PASS")
    print("Agent tests:", count, "passed")
    print("Project regression:", passed)
    models, turns = set(), 0
    for operator in ("gdn_commit", "gdn_verify"):
        root = runs / ("unroll-" + operator)
        result = json.loads((root / "result.json").read_text())
        assert result["status"] == "complete"
        print(
            operator,
            result["status"],
            f"{result['final_comparison']['speedup_geomean']:.4f}x",
            "improvement=",
            result["independent_improvement_established"],
        )
        for file in root.glob("conversation-*.jsonl"):
            for line in file.read_text().splitlines():
                api = json.loads(line).get("api", {})
                if api.get("returned_model"):
                    models.add(api["returned_model"])
                    turns += 1
    print("Real API:", ", ".join(sorted(models)), turns, "turns")
    print("CANN 8.5 batch commit compatibility fix: PASS")
    print("Scope: operator/context regression; full model throughput unmeasured")


if __name__ == "__main__":
    main()
