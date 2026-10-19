"""Authoritative workspace, tool lifecycle, immutable evidence, and promotion rules."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import shutil
import uuid

from .runtime import Worker
from .statistics import summarize
from .tasks import default_task, digest, render, validate_config, validate_task


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(path)


@contextmanager
def workspace_lock(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another optimizer is using this workspace") from None
        yield root


def initialize(root, name, task=None):
    root = Path(root).resolve()
    if (root / "task.json").exists() or (root / "state.json").exists():
        raise FileExistsError(
            "Workspace exists; use search/agent to resume, or a new directory"
        )
    task = task or default_task(name)
    validate_task(task)
    root.mkdir(parents=True, exist_ok=True)
    (root / "candidates").mkdir()
    (root / "evaluations").mkdir()
    source = render(task["operator"], task["baseline_config"])
    sha = digest(source)
    cid = sha[:16]
    (root / "candidates" / f"{cid}.py").write_text(source)
    atomic_json(
        root / "candidates" / f"{cid}.json",
        {
            "id": cid,
            "source_sha256": sha,
            "config": task["baseline_config"],
            "reason": "Original task baseline",
            "created_at": now(),
        },
    )
    (root / "baseline.py").write_text(source)
    (root / "best.py").write_text(source)
    atomic_json(root / "task.json", task)
    atomic_json(
        root / "state.json",
        {
            "schema_version": 1,
            "task_sha256": digest(task),
            "baseline": cid,
            "best": cid,
            "promotions": 0,
            "created_at": now(),
        },
    )
    return root


class Engine:
    def __init__(self, root, *, device="npu:0", timeout=240, worker=None):
        self.root = Path(root).resolve()
        self.task = json.loads((self.root / "task.json").read_text())
        validate_task(self.task)
        self.state = json.loads((self.root / "state.json").read_text())
        if digest(self.task) != self.state["task_sha256"]:
            raise ValueError("Frozen task contract changed; create a new workspace")
        self.device = device
        self.session = uuid.uuid4().hex
        self.worker = worker or Worker(self.root, timeout)
        self.stages = {}
        self.benches = {}
        self.decisions = 0
        self.environment_info = None
        self.candidate(self.state["baseline"])
        _, best = self.candidate(self.state["best"])
        if (self.root / "best.py").read_bytes() != best.read_bytes():
            raise ValueError("best.py changed outside the acceptance tool")

    def close(self):
        self.worker.close()

    def event(self, name, result, **extra):
        with (self.root / "trajectory.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "time": now(),
                        "session": self.session,
                        "event": name,
                        **extra,
                        "result": result,
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )

    def candidate(self, candidate_id):
        if not isinstance(candidate_id, str) or not re.fullmatch(
            r"[0-9a-f]{16}", candidate_id
        ):
            raise ValueError("Invalid candidate ID")
        path = self.root / "candidates" / f"{candidate_id}.py"
        manifest = json.loads(path.with_suffix(".json").read_text())
        source = path.read_text()
        expected = render(self.task["operator"], manifest["config"])
        if (
            source != expected
            or digest(source) != manifest["source_sha256"]
            or digest(source)[:16] != candidate_id
        ):
            raise ValueError("Candidate source/manifest changed after creation")
        return manifest, path

    def request(self, action, candidate_id, **extra):
        manifest, path = self.candidate(candidate_id)
        return self.worker.call(
            {
                "action": action,
                "candidate": str(path),
                "task": self.task,
                "device": self.device,
                **extra,
            }
        )

    def get_task(self):
        return {
            "task": self.task,
            "state": self.state,
            "scope": "Template-based Triton-Ascend optimization including frozen project GDN kernels; no arbitrary source or shell execution",
        }

    def get_environment(self):
        if self.environment_info is None:
            self.environment_info = self.worker.call({"action": "environment"})
            atomic_json(self.root / "environment.json", self.environment_info)
        return self.environment_info

    def read_candidate(self, candidate_id):
        manifest, path = self.candidate(candidate_id)
        return {**manifest, "source": path.read_text()}

    def propose_candidate(self, config, reason):
        validate_config(self.task["operator"], config)
        if not isinstance(reason, str) or not 1 <= len(reason) <= 2000:
            raise ValueError("Supply a concise optimization hypothesis")
        source = render(self.task["operator"], config)
        sha = digest(source)
        cid = sha[:16]
        path = self.root / "candidates" / f"{cid}.py"
        if not path.exists():
            path.write_text(source)
            atomic_json(
                path.with_suffix(".json"),
                {
                    "id": cid,
                    "source_sha256": sha,
                    "config": config,
                    "reason": reason,
                    "created_at": now(),
                },
            )
            self.event(
                "candidate_created",
                {"candidate_id": cid, "config": config, "reason": reason},
            )
        self.candidate(cid)
        return {
            "candidate_id": cid,
            "source_sha256": sha,
            "already_best": cid == self.state["best"],
        }

    def compile_kernel(self, candidate_id):
        self.stages.pop(candidate_id, None)
        self.benches.pop(candidate_id, None)
        result = self.request("compile", candidate_id)
        self.stages[candidate_id] = "compiled"
        return result

    def validate_kernel(self, candidate_id):
        if candidate_id not in self.stages:
            raise ValueError("Compile this exact candidate before validation")
        self.stages[candidate_id] = "compiled"
        self.benches.pop(candidate_id, None)
        result = self.request("validate", candidate_id)
        if not result.get("passed"):
            raise ValueError("Candidate failed correctness")
        self.stages[candidate_id] = "validated"
        return result

    def _benchmark(self, candidate_id, baseline_id, label):
        _, baseline = self.candidate(baseline_id)
        result = self.request("benchmark", candidate_id, baseline=str(baseline))
        summary = summarize(result["cases"], self.task["acceptance"])
        evaluation_id = uuid.uuid4().hex
        evidence = {
            "evaluation_id": evaluation_id,
            "session": self.session,
            "task_sha256": self.state["task_sha256"],
            "device": self.device,
            "candidate": candidate_id,
            "incumbent": baseline_id,
            "created_at": now(),
            "label": label,
            "raw": result,
            "summary": summary,
        }
        path = self.root / "evaluations" / f"{evaluation_id}.json"
        atomic_json(path, evidence)
        return {"evaluation_id": evaluation_id, "evidence_file": str(path), **summary}

    def bench_kernel(self, candidate_id):
        if self.stages.get(candidate_id) != "validated":
            raise ValueError(
                "Compile and validate this exact candidate before benchmarking"
            )
        self.benches.pop(candidate_id, None)
        result = self._benchmark(
            candidate_id, self.state["best"], "candidate_vs_incumbent"
        )
        self.benches[candidate_id] = result["evaluation_id"]
        return result

    def accept_candidate(self, candidate_id):
        self.candidate(candidate_id)
        evaluation_id = self.benches.pop(candidate_id, None)
        if evaluation_id is None:
            raise ValueError("No fresh benchmark for this candidate in this session")
        evidence = json.loads(
            (self.root / "evaluations" / f"{evaluation_id}.json").read_text()
        )
        if (
            evidence["session"] != self.session
            or evidence["candidate"] != candidate_id
            or evidence["incumbent"] != self.state["best"]
        ):
            raise ValueError(
                "Benchmark is stale; benchmark against the current best again"
            )
        if evidence["task_sha256"] != self.state["task_sha256"]:
            raise ValueError("Benchmark used a different task")
        if not all(
            evidence["raw"]["correctness"][k].get("passed")
            for k in ("baseline", "candidate")
        ):
            raise ValueError("Benchmark correctness was not established")
        verdict = summarize(evidence["raw"]["cases"], self.task["acceptance"])
        if candidate_id == self.state["best"]:
            verdict.update(accepted=False, reason="already_incumbent")
        old_best = self.state["best"]
        if verdict["accepted"]:
            self.state["best"] = candidate_id
            self.state["promotions"] += 1
            _, path = self.candidate(candidate_id)
            temporary = self.root / "best.py.tmp"
            shutil.copyfile(path, temporary)
            temporary.replace(self.root / "best.py")
            atomic_json(self.root / "state.json", self.state)
        self.decisions += 1
        result = {
            "candidate_id": candidate_id,
            "incumbent_before": old_best,
            "best": self.state["best"],
            "evaluation_id": evaluation_id,
            **verdict,
        }
        self.event("candidate_decision", result)
        return result

    def profile_kernel(self, candidate_id):
        if self.stages.get(candidate_id) != "validated":
            raise ValueError("Compile and validate candidate before profiling")
        directory = self.root / "profiles" / f"{candidate_id}-{uuid.uuid4().hex[:8]}"
        return self.request("profile", candidate_id, directory=str(directory))

    def get_history(self):
        path = self.root / "trajectory.jsonl"
        rows = (
            [json.loads(line) for line in path.read_text().splitlines()]
            if path.exists()
            else []
        )
        return {"state": self.state, "events": rows[-30:]}

    def call(self, name, arguments):
        allowed = {
            "get_task",
            "get_environment",
            "read_candidate",
            "propose_candidate",
            "compile_kernel",
            "validate_kernel",
            "bench_kernel",
            "accept_candidate",
            "profile_kernel",
            "get_history",
        }
        if name not in allowed:
            result = {"ok": False, "error": "Unknown tool"}
        else:
            try:
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be a JSON object")
                if self.benches:
                    pending = sorted(self.benches)
                    if name in {
                        "propose_candidate",
                        "compile_kernel",
                        "validate_kernel",
                    }:
                        raise ValueError(
                            f"Benchmarked candidate {pending} is pending. Call accept_candidate before starting/revalidating a candidate."
                        )
                    if (
                        name in {"bench_kernel", "profile_kernel", "accept_candidate"}
                        and arguments.get("candidate_id") not in self.benches
                    ):
                        raise ValueError(
                            f"Decide pending candidate {pending} first, using its exact candidate_id."
                        )
                result = {"ok": True, "result": getattr(self, name)(**arguments)}
            except Exception as exc:
                result = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[-12000:],
                }
        self.event("tool_call", result, tool=name, arguments=arguments)
        print(
            json.dumps(
                {
                    "tool": name,
                    "ok": result["ok"],
                    "candidate_id": arguments.get("candidate_id")
                    if isinstance(arguments, dict)
                    else None,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return result

    def preflight(self):
        self.get_environment()
        best = self.state["best"]
        for tool in ("compile_kernel", "validate_kernel"):
            result = self.call(tool, {"candidate_id": best})
            if not result["ok"]:
                raise RuntimeError("Baseline/best preflight failed: " + result["error"])
        self.event("preflight", {"passed": True, "best": best})

    def finalize(self, planner, status, *, model_text=None, profile=False):
        # Independent final measurement against the original baseline, not a product of promotion ratios.
        comparison = self._benchmark(
            self.state["best"], self.state["baseline"], "independent_final_vs_original"
        )
        improved = (
            self.state["best"] != self.state["baseline"] and comparison["accepted"]
        )
        result = {
            "status": status,
            "planner": planner,
            "best": self.state["best"],
            "baseline": self.state["baseline"],
            "promotions": self.state["promotions"],
            "independent_improvement_established": improved,
            "final_comparison": comparison,
            "finished_at": now(),
            "model_text": model_text,
        }
        if profile:
            # Current best is known valid from the final benchmark, even if selected in an earlier session.
            self.stages[self.state["best"]] = "validated"
            result["profile"] = self.call(
                "profile_kernel", {"candidate_id": self.state["best"]}
            )
        atomic_json(self.root / "result.json", result)
        write_report(self.root)
        return result


def write_report(root):
    root = Path(root)
    result = json.loads((root / "result.json").read_text())
    task = json.loads((root / "task.json").read_text())
    manifest = json.loads((root / "candidates" / f"{result['best']}.json").read_text())
    final = result["final_comparison"]
    lines = [
        "# Ascend 算子优化报告",
        "",
        f"算子：`{task['operator']}`；规划器：`{result['planner']}`；状态：`{result['status']}`。",
        "",
        f"最佳配置：`{json.dumps(manifest['config'])}`；累计晋升 {result['promotions']} 次。",
        "",
        "最终独立复测"
        + (
            "支持相对原始基线的改进。"
            if result["independent_improvement_established"]
            else "尚未建立相对原始基线的稳定改进；保留数据，不宣称加速。"
        ),
        "",
        f"等权几何平均加速比：**{final['speedup_geomean']:.4f}×**；探索性 bootstrap 95% 区间："
        f"[{final['bootstrap_95_interval'][0]:.4f}, {final['bootstrap_95_interval'][1]:.4f}]。",
        "",
        "| shape | dtype | 原始基线 μs | 最终版本 μs | 配对加速比 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in final["per_case"]:
        lines.append(
            f"| {row['case']['shape']} | {row['case']['dtype']} | {row['baseline_us']:.3f} | {row['candidate_us']:.3f} | {row['speedup']:.4f}× |"
        )
    lines += [
        "",
        "## 计时与验收",
        "",
        "- 固定 shape、dtype、epsilon；CPU 参考检查、输入不可变检查、输出 guard 与 NaN 检查。",
        f"- NPU event 计时，预热 {task['measurement']['warmup']} 次，每组 {task['measurement']['iterations']} 次调用，交替 ABBA/BAAB，共 {task['measurement']['rounds']} 个配对轮次。",
        f"- 晋升要求：整体加速比探索性区间下界 ≥ {task['acceptance']['min_speedup']}，每项中位加速比 ≥ {task['acceptance']['min_case_speedup']}。",
        "- 计时包含 run 内的分配与 host 发射间隔；预先分配的输入/输出、编译和热身不计时。",
        "- RMSNorm 基线为 PyTorch eager 分解表达式；GDN 基线为冻结项目原 kernel 默认配置，验收还要求与原 NPU kernel 逐位一致。",
        "- GDN commit 是状态写入算子：每个计时段之前在 event 外恢复原始状态；重复计算保持有限，原记录和元数据不可变。",
        "- 测量复用输入，可能命中缓存；未据此推断真实 HBM 带宽或整模型端到端收益。",
        "- bootstrap 区间用于探索，样本较少且复用设备；不等价于严格性能认证。",
        "",
        "## 产物",
        "",
        "`best.py`、`task.json`、`environment.json`、`state.json`、`trajectory.jsonl`、`evaluations/*.json`、`result.json`。",
        "",
        "LLM 负责选配置和解释反馈；所有源码由固定模板生成，模型不能修改任务、参考实现、计时器或验收阈值。此版本支持预置算子与冻结项目 GDN kernel，尚不支持任意算子源码改写。",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    return root / "REPORT.md"
