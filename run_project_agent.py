"""Real DeepSeek planner drives frozen project GDN kernels on this notebook's NPU."""

import json
from pathlib import Path
import traceback

from ascend_optimizer.agent import run_agent
from ascend_optimizer.engine import Engine, atomic_json, initialize, workspace_lock
from ascend_optimizer.llm import Client


def main():
    project = Path(__file__).resolve().parent
    results = {}
    key_file = Path.home() / ".config/ascend-optimizer/api-key"
    for name in ("gdn_commit", "gdn_verify"):
        root = project / "runs-20261016" / ("unroll-" + name)
        with workspace_lock(root):
            if not (root / "task.json").exists():
                initialize(root, name)
            engine = Engine(root, timeout=900)
            client = Client(
                base_url="https://api.deepseek.com",
                model="deepseek-v4-flash",
                key_file=key_file,
                timeout=120,
                request_options={"max_tokens": 4096, "thinking": {"type": "disabled"}},
            )
            try:
                result = run_agent(
                    engine, client, max_steps=40, max_tools=96, profile=True
                )
                results[name] = result
                print("REAL_PROJECT_AGENT_RESULT", name, json.dumps(result), flush=True)
            finally:
                engine.close()
            atomic_json(project / "runs-20261016/summary.json", results)
            from export_project_evidence import main as checkpoint

            checkpoint()
    print("REAL_PROJECT_AGENT_FINISHED", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
