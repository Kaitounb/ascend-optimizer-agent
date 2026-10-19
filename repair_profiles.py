"""Re-collect profiler artifacts without repeating the completed performance search."""

import json
from pathlib import Path

from ascend_optimizer.engine import Engine, atomic_json, workspace_lock, write_report


def main():
    for name in ("vector_add", "rms_norm"):
        root = Path(__file__).resolve().parent / "runs" / name
        with workspace_lock(root):
            engine = Engine(root, timeout=300)
            try:
                engine.preflight()
                outcome = engine.call(
                    "profile_kernel", {"candidate_id": engine.state["best"]}
                )
                if not outcome["ok"]:
                    raise RuntimeError(outcome)
                result = json.loads((root / "result.json").read_text())
                result["profile"] = outcome
                atomic_json(root / "result.json", result)
                write_report(root)
                print("PROFILE_REPAIRED", name, outcome["result"]["trace"], flush=True)
            finally:
                engine.close()


if __name__ == "__main__":
    main()
