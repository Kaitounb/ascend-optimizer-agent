from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from . import __version__
from .agent import run_agent, search
from .engine import Engine, atomic_json, initialize, workspace_lock, write_report
from .llm import Client
from .runtime import Worker
from .tasks import TASKS


def bounded_integer(low, high):
    def parse(value):
        n = int(value)
        if not low <= n <= high:
            raise argparse.ArgumentTypeError(f"Use an integer between {low} and {high}")
        return n

    return parse


def parser():
    p = argparse.ArgumentParser(
        description="Ascend operator optimizer: evidence-driven LLM tools + NPU evaluator"
    )
    p.add_argument("--version", action="version", version=__version__)
    subs = p.add_subparsers(dest="command", required=True)
    init = subs.add_parser("init", help="Create a new immutable task workspace")
    init.add_argument("--task", choices=TASKS, default="vector_add")
    init.add_argument(
        "--task-file",
        type=Path,
        help="Optional full JSON task contract; frozen at creation",
    )
    init.add_argument("--workspace", type=Path, required=True)
    doctor = subs.add_parser(
        "doctor", help="Inspect NPU/toolchain without calling a model"
    )
    doctor.add_argument("--output", type=Path)
    for name in ("search", "agent", "evaluate"):
        item = subs.add_parser(name)
        item.add_argument("--workspace", type=Path, required=True)
        item.add_argument("--device", default="npu:0")
        item.add_argument("--timeout", type=bounded_integer(10, 1800), default=240)
        item.add_argument(
            "--profile", action="store_true", help="Export CPU/NPU trace of final best"
        )
        if name == "search":
            item.add_argument(
                "--max-candidates", type=bounded_integer(1, 64), default=6
            )
        elif name == "agent":
            item.add_argument(
                "--base-url",
                default=os.environ.get(
                    "ASCEND_OPT_BASE_URL", "https://api.openai.com/v1"
                ),
            )
            item.add_argument("--model", default=os.environ.get("ASCEND_OPT_MODEL", ""))
            item.add_argument(
                "--api-mode",
                choices=("chat", "responses"),
                default=os.environ.get("ASCEND_OPT_API_MODE", "chat"),
            )
            item.add_argument("--key-env", default="OPENAI_API_KEY")
            item.add_argument("--key-file", type=Path)
            item.add_argument(
                "--request-options",
                default="{}",
                help="Optional model settings JSON: max_tokens, thinking, reasoning_effort",
            )
            item.add_argument("--max-steps", type=bounded_integer(1, 100), default=24)
            item.add_argument("--max-tools", type=bounded_integer(1, 256), default=64)
        else:
            item.add_argument(
                "--config", required=True, help="Candidate configuration JSON"
            )
            item.add_argument("--reason", default="Manual optimization hypothesis")
    report = subs.add_parser(
        "report", help="Regenerate report from saved evidence; no NPU required"
    )
    report.add_argument("--workspace", type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "init":
            with workspace_lock(args.workspace):
                root = initialize(
                    args.workspace,
                    args.task,
                    json.loads(args.task_file.read_text()) if args.task_file else None,
                )
            print(root)
            return 0
        if args.command == "doctor":
            with tempfile.TemporaryDirectory(prefix="ascend-doctor-") as directory:
                worker = Worker(Path(directory))
                try:
                    result = worker.call({"action": "environment"})
                finally:
                    worker.close()
            if args.output:
                atomic_json(args.output, result)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if args.command == "report":
            print(write_report(args.workspace))
            return 0
        # Validate model configuration before spending NPU time.
        client = (
            Client(
                base_url=args.base_url,
                model=args.model,
                api_mode=args.api_mode,
                key_env=args.key_env,
                key_file=args.key_file,
                request_options=json.loads(args.request_options),
            )
            if args.command == "agent"
            else None
        )
        with workspace_lock(args.workspace):
            engine = Engine(args.workspace, device=args.device, timeout=args.timeout)
            try:
                if args.command == "search":
                    result = search(
                        engine, max_candidates=args.max_candidates, profile=args.profile
                    )
                elif args.command == "agent":
                    result = run_agent(
                        engine,
                        client,
                        max_steps=args.max_steps,
                        max_tools=args.max_tools,
                        profile=args.profile,
                    )
                else:
                    engine.preflight()
                    proposal = engine.call(
                        "propose_candidate",
                        {"config": json.loads(args.config), "reason": args.reason},
                    )
                    if not proposal["ok"]:
                        raise ValueError(proposal["error"])
                    cid = proposal["result"]["candidate_id"]
                    success = True
                    for tool in (
                        "compile_kernel",
                        "validate_kernel",
                        "bench_kernel",
                        "accept_candidate",
                    ):
                        if not engine.call(tool, {"candidate_id": cid})["ok"]:
                            success = False
                            break
                    result = engine.finalize(
                        "manual",
                        "complete" if success else "candidate_failed",
                        profile=args.profile,
                    )
            except BaseException as exc:
                engine.event(
                    "run_interrupted",
                    {
                        "error_type": type(exc).__name__,
                        "best_retained": engine.state["best"],
                    },
                )
                raise
            finally:
                engine.close()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["status"] == "complete" else 2
    except KeyboardInterrupt:
        print(
            "Interrupted. Workspace and best candidate retained; rerun the same command to continue.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
