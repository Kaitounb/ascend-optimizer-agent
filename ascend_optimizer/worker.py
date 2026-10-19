"""Persistent NPU worker. No API credentials, no model-supplied commands or evaluators."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback

PREFIX = "@@ASCEND_RESULT@@"
_torch = None


def torch_module(device):
    global _torch
    if _torch is None:
        import torch

        torch.set_num_threads(1)
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            raise RuntimeError(
                "Ascend NPU unavailable; CPU timings are never a substitute"
            )
        torch.npu.set_device(device)
        _torch = torch
    return _torch


def environment():
    packages = {}
    for name in ("torch", "torch-npu", "triton-ascend"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        smi = subprocess.run(
            ["npu-smi", "info"], capture_output=True, text=True, timeout=15
        )
        smi_text = (smi.stdout + smi.stderr)[-12000:]
    except (OSError, subprocess.TimeoutExpired) as exc:
        smi_text = str(exc)
    return {
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "packages": packages,
        "tools": {n: shutil.which(n) for n in ("msprof", "bisheng", "ccec")},
        "npu_smi": smi_text,
    }


def load_module(path):
    path = Path(path).resolve()
    name = "candidate_" + __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case_inputs(task, shape, dtype_name, device, seed):
    torch = torch_module(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    dtype = getattr(torch, dtype_name)
    xc = torch.randn(tuple(shape), generator=generator, dtype=torch.float32).to(dtype)
    if task["operator"] == "vector_add":
        yc = torch.randn(tuple(shape), generator=generator, dtype=torch.float32).to(
            dtype
        )
        reference = xc + yc
        cpu_inputs = [xc, yc]
    else:
        wc = torch.randn(shape[-1], generator=generator, dtype=torch.float32).to(dtype)
        xf = xc.float()
        reference = (
            xf
            * torch.rsqrt((xf * xf).mean(-1, keepdim=True) + task["epsilon"])
            * wc.float()
        ).to(dtype)
        cpu_inputs = [xc, wc]
    inputs = [t.to(device) for t in cpu_inputs]
    count = math.prod(shape)
    storage = torch.full((count + 64,), 123, dtype=dtype, device=device)
    out = storage[32:-32].view(shape)
    out.fill_(float("nan"))
    args = [*inputs, out]
    if task["operator"] == "rms_norm":
        args.append(task["epsilon"])
    return args, reference, cpu_inputs, storage


def all_cases(task):
    shapes = list(
        dict.fromkeys(
            tuple(s) for s in task["validation_shapes"] + task["benchmark_shapes"]
        )
    )
    return [(list(s), d) for s in shapes for d in task["dtypes"]]


def validate(module, task, device):
    torch = torch_module(device)
    cases = []
    for shape, dtype in all_cases(task):
        max_error = 0.0
        for seed in (task["seed"], task["seed"] + 17):
            args, reference, cpu_inputs, storage = case_inputs(
                task, shape, dtype, device, seed
            )
            module.run(*args)
            torch.npu.synchronize()
            actual = args[2].cpu()
            rtol, atol = (1e-4, 1e-5) if dtype == "float32" else (5e-3, 5e-3)
            torch.testing.assert_close(
                actual, reference, rtol=rtol, atol=atol, equal_nan=False
            )
            for n, original in enumerate(cpu_inputs):
                torch.testing.assert_close(args[n].cpu(), original, rtol=0, atol=0)
            if not bool((storage[:32] == 123).all()) or not bool(
                (storage[-32:] == 123).all()
            ):
                raise AssertionError("Output guard region was overwritten")
            max_error = max(
                max_error, float((actual.float() - reference.float()).abs().max())
            )
        cases.append(
            {"shape": shape, "dtype": dtype, "seeds": 2, "max_abs_error": max_error}
        )
    return {
        "passed": True,
        "cases": cases,
        "checks": [
            "CPU reference",
            "input immutability",
            "output guard",
            "NaN detection",
        ],
    }


def compile_kernel(module, task, device):
    torch = torch_module(device)
    for dtype in task["dtypes"]:
        args, *_ = case_inputs(
            task, task["benchmark_shapes"][0], dtype, device, task["seed"]
        )
        module.run(*args)
        torch.npu.synchronize()
    return {
        "compiled_and_launched": True,
        "note": "JIT was triggered on NPU; full validation is separate",
    }


def measure(call, iterations):
    torch = _torch
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    end.synchronize()
    us = start.elapsed_time(end) * 1000.0 / iterations
    if not math.isfinite(us) or us <= 0:
        raise RuntimeError("NPU event returned invalid timing")
    return us


def benchmark(base, candidate, task, device):
    torch = torch_module(device)
    # Repeat correctness here, independently of earlier compile/validate tool calls.
    checked = {
        "baseline": validate(base, task, device),
        "candidate": validate(candidate, task, device),
    }
    policy = task["measurement"]
    cases = []
    for shape in task["benchmark_shapes"]:
        for dtype in task["dtypes"]:
            args, *_ = case_inputs(task, shape, dtype, device, task["seed"] + 99)
            calls = {
                "baseline": lambda: base.run(*args),
                "candidate": lambda: candidate.run(*args),
            }
            for _ in range(policy["warmup"]):
                calls["baseline"]()
                calls["candidate"]()
            torch.npu.synchronize()
            pairs = []
            for r in range(policy["rounds"]):
                order = ["baseline", "candidate", "candidate", "baseline"]
                if r % 2:
                    order = ["candidate", "baseline", "baseline", "candidate"]
                timings = {"baseline": [], "candidate": []}
                for label in order:
                    timings[label].append(measure(calls[label], policy["iterations"]))
                pairs.append(
                    {
                        "baseline_us": sum(timings["baseline"]) / 2,
                        "candidate_us": sum(timings["candidate"]) / 2,
                        "order": "ABBA" if r % 2 == 0 else "BAAB",
                    }
                )
            cases.append({"case": {"shape": shape, "dtype": dtype}, "pairs": pairs})
    return {
        "correctness": checked,
        "cases": cases,
        "timing_scope": "NPU event around repeated host launches; allocations inside run are included; compile and warmup excluded",
    }


def profile_kernel(module, task, device, directory):
    torch = torch_module(device)
    import torch_npu

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    args, *_ = case_inputs(
        task, task["benchmark_shapes"][-1], task["dtypes"][0], device, task["seed"]
    )
    for _ in range(5):
        module.run(*args)
    torch.npu.synchronize()
    profiler = torch_npu.profiler
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        record_shapes=True,
    ) as prof:
        for _ in range(5):
            module.run(*args)
        torch.npu.synchronize()
    trace = directory / "trace.json"
    prof.export_chrome_trace(str(trace))
    # torch_npu.profiler.profile in some supported releases has no key_averages().
    # Parse the exported Chrome trace instead of depending on a PyTorch profiler wrapper API.
    summary = summarize_trace(trace)
    (directory / "trace_summary.json").write_text(json.dumps(summary, indent=2))
    return {
        "trace": str(trace),
        "summary": summary,
        "invocations": 5,
        "note": "Trace durations include profiling overhead and are not acceptance timings; inspect categories/processes to distinguish CPU from NPU.",
    }


def summarize_trace(path):
    trace = json.loads(Path(path).read_text())
    events = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
    categories = {}
    durations = []
    for event in events:
        category = str(event.get("cat", "uncategorized"))
        categories[category] = categories.get(category, 0) + 1
        duration = event.get("dur")
        if (
            event.get("ph") == "X"
            and isinstance(duration, (int, float))
            and math.isfinite(duration)
            and duration >= 0
        ):
            durations.append(
                {
                    "name": event.get("name"),
                    "category": category,
                    "process": event.get("pid"),
                    "thread": event.get("tid"),
                    "trace_duration_us": duration,
                }
            )
    return {
        "events": len(events),
        "categories": categories,
        "longest_complete_events": sorted(
            durations, key=lambda x: x["trace_duration_us"], reverse=True
        )[:20],
    }


def handle(request):
    action = request["action"]
    if action == "environment":
        return environment()
    from .tasks import validate_task

    task = request["task"]
    validate_task(task)
    device = request.get("device", "npu:0")
    torch_module(device)
    if task["operator"].startswith("gdn_"):
        from .gdn_evaluator import handle as handle_gdn

        return handle_gdn(request)
    candidate = load_module(request["candidate"])
    if action == "compile":
        return compile_kernel(candidate, task, device)
    if action == "validate":
        return validate(candidate, task, device)
    if action == "benchmark":
        return benchmark(load_module(request["baseline"]), candidate, task, device)
    if action == "profile":
        return profile_kernel(candidate, task, device, request["directory"])
    raise ValueError(f"Unknown worker action: {action}")


def main():
    for line in sys.stdin:
        try:
            request = json.loads(line)
            result = {"ok": True, "result": handle(request)}
        except Exception as exc:
            result = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc)[-6000:],
                "traceback": traceback.format_exc()[-10000:],
            }
        print(PREFIX + json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
