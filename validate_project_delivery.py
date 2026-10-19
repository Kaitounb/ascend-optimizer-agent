"""Serial validation avoids benchmark interference from other NPU test processes."""

from pathlib import Path
import subprocess
import sys

from ascend_optimizer.engine import atomic_json


def main():
    root = Path(__file__).resolve().parent
    (root / "runs-20261016").mkdir(exist_ok=True)
    status = {}
    with (root / "regression_unit_tests.log").open("w") as log:
        code = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
        ).returncode
    status["agent_unit_tests_exit"] = code
    atomic_json(root / "runs-20261016/validation_status.json", status)
    if code:
        raise RuntimeError("Agent unit tests failed")
    with (root / "project_regression_baseline.log").open("w") as log:
        code = subprocess.run(
            [sys.executable, "run_project_regression.py"],
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
        ).returncode
    status["original_project_tests_exit"] = code
    atomic_json(root / "runs-20261016/validation_status.json", status)
    print("BASELINE_PROJECT_TESTS_EXIT", code, flush=True)
    from run_project_agent import main as optimize

    optimize()
    print("REAL_AGENT_PHASE_FINISHED", flush=True)


if __name__ == "__main__":
    main()
