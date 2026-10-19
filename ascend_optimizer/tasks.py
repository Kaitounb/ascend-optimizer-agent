"""Fixed task contracts and host-owned kernel templates; models cannot edit evaluators."""

from __future__ import annotations

import hashlib
import json

from .gdn_templates import specs as _gdn_specs

TASKS = {
    "vector_add": {
        "operator": "vector_add",
        "description": "Contiguous out = x + y; inputs must remain unchanged.",
        "dtypes": ["float32", "float16"],
        "validation_shapes": [[1], [1023], [1024], [65537]],
        "benchmark_shapes": [[65537], [1048576]],
        "baseline_config": {"block": 256, "unroll": 1},
        "search_configs": [
            {"block": b, "unroll": u}
            for b, u in [
                (1024, 1),
                (4096, 1),
                (16384, 1),
                (4096, 4),
                (1024, 4),
                (16384, 4),
            ]
        ],
    },
    "rms_norm": {
        "operator": "rms_norm",
        "description": "out = x * rsqrt(mean(x.float()**2, -1) + eps) * weight; FP32 accumulation.",
        "dtypes": ["float32", "float16"],
        "epsilon": 1e-6,
        "validation_shapes": [[3, 127], [7, 1025]],
        "benchmark_shapes": [[32, 1024], [128, 4096]],
        "baseline_config": {"implementation": "torch_eager", "rows": 1},
        "search_configs": [
            {"implementation": "triton", "rows": r} for r in (1, 2, 4, 8)
        ],
    },
}


TASKS.update(_gdn_specs())


def digest(value) -> str:
    if not isinstance(value, str):
        value = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    return hashlib.sha256(value.encode()).hexdigest()


def default_task(name: str) -> dict:
    task = json.loads(json.dumps(TASKS[name]))
    task.update(
        {
            "schema_version": 1,
            "seed": 20261015,
            "measurement": {"warmup": 10, "iterations": 20, "rounds": 7},
            "acceptance": {
                "min_speedup": 1.02,
                "min_case_speedup": 0.98,
                "bootstrap_samples": 2000,
            },
        }
    )
    if name.startswith("gdn_"):
        task["measurement"] = {"warmup": 4, "iterations": 5, "rounds": 9}
        if name == "gdn_commit":
            task["dtypes"] = ["float32"]
    return task


def validate_task(task: dict) -> None:
    if task.get("operator") not in TASKS or task.get("schema_version") != 1:
        raise ValueError("Unsupported task/schema; use init to create a task contract")
    supported_dtypes = TASKS[task["operator"]]["dtypes"]
    if task["operator"] == "gdn_commit":
        supported_dtypes = ["float32"]
    if not task.get("dtypes") or any(d not in supported_dtypes for d in task["dtypes"]):
        raise ValueError(f"Supported dtypes for this task: {supported_dtypes}")
    if task["operator"].startswith("gdn_"):
        from .gdn_templates import validate_shapes

        validate_shapes(task)
    rank = 1 if task["operator"] == "vector_add" else 2
    for field in (
        ()
        if task["operator"].startswith("gdn_")
        else ("validation_shapes", "benchmark_shapes")
    ):
        if not task.get(field):
            raise ValueError(f"{field} cannot be empty")
        for shape in task[field]:
            if len(shape) != rank or any(type(n) is not int or n < 1 for n in shape):
                raise ValueError(f"Invalid shape: {shape}")
            if (rank == 1 and shape[0] > 2**26) or (
                rank == 2 and (shape[0] > 4096 or shape[1] > 8192)
            ):
                raise ValueError("Shape exceeds this MVP's memory/tiling bounds")
    if task["operator"] == "rms_norm" and not (0 < task.get("epsilon", 0) <= 0.1):
        raise ValueError("Invalid epsilon")
    m = task["measurement"]
    for key, low, high in (
        ("warmup", 1, 100),
        ("iterations", 1, 200),
        ("rounds", 5, 31),
    ):
        if type(m[key]) is not int or not low <= m[key] <= high:
            raise ValueError(f"Invalid measurement.{key}")
    a = task["acceptance"]
    if not (1 <= a["min_speedup"] <= 2 and 0.9 <= a["min_case_speedup"] <= 1):
        raise ValueError("Invalid acceptance thresholds")
    if (
        type(a["bootstrap_samples"]) is not int
        or not 100 <= a["bootstrap_samples"] <= 10000
    ):
        raise ValueError("Invalid bootstrap_samples")
    if type(task.get("seed")) is not int or not 0 <= task["seed"] < 2**63:
        raise ValueError("Invalid random seed")
    if (
        not isinstance(task.get("search_configs"), list)
        or len(task["search_configs"]) > 64
    ):
        raise ValueError("Provide at most 64 search configurations")
    for config in task["search_configs"]:
        validate_config(task["operator"], config)
    validate_config(task["operator"], task["baseline_config"])


def validate_config(operator: str, config: dict) -> None:
    if operator.startswith("gdn_"):
        from .gdn_templates import validate_config as validate_gdn_config

        validate_gdn_config(config)
        return
    if operator == "vector_add":
        if set(config) != {"block", "unroll"}:
            raise ValueError("vector_add config requires exactly block and unroll")
        if type(config["block"]) is not int or config["block"] not in (
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
        ):
            raise ValueError("block must be a supported power of two from 128 to 16384")
        if type(config["unroll"]) is not int or config["unroll"] not in (1, 2, 4, 8):
            raise ValueError("unroll must be 1, 2, 4, or 8")
    elif operator == "rms_norm":
        if set(config) != {"implementation", "rows"}:
            raise ValueError("rms_norm config requires exactly implementation and rows")
        if config["implementation"] not in ("torch_eager", "triton"):
            raise ValueError("Unknown implementation")
        if type(config["rows"]) is not int or config["rows"] not in (1, 2, 4, 8):
            raise ValueError("rows must be 1, 2, 4, or 8")
        if config["implementation"] == "torch_eager" and config["rows"] != 1:
            raise ValueError("torch_eager only supports rows=1")
    else:
        raise ValueError("Unknown operator")


def render(operator: str, config: dict) -> str:
    validate_config(operator, config)
    if operator.startswith("gdn_"):
        from .gdn_templates import render as render_gdn

        return render_gdn(operator, config)
    header = (
        "# Generated by ascend_optimizer; edit templates rather than evidence files.\n"
    )
    if operator == "vector_add":
        return (
            header
            + """import triton
import triton.language as tl

@triton.jit
def kernel(X, Y, O, n, BLOCK: tl.constexpr, UNROLL: tl.constexpr):
    base = tl.program_id(0) * BLOCK * UNROLL + tl.arange(0, BLOCK)
    for i in range(UNROLL):
        offsets = base + i * BLOCK
        a = tl.load(X + offsets, offsets < n, other=0)
        b = tl.load(Y + offsets, offsets < n, other=0)
        tl.store(O + offsets, a + b, offsets < n)

def run(x, y, out):
"""
            + f"""    kernel[(triton.cdiv(x.numel(), {config["block"] * config["unroll"]}),)](x, y, out, x.numel(), BLOCK={config["block"]}, UNROLL={config["unroll"]})
"""
        )
    if config["implementation"] == "torch_eager":
        return (
            header
            + """import torch

def run(x, weight, out, eps):
    xf = x.float()
    value = xf * torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + eps)
    out.copy_((value * weight.float()).to(x.dtype))
"""
        )
    return (
        header
        + """import triton
import triton.language as tl

@triton.jit
def kernel(X, W, O, m, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr, ROWS: tl.constexpr):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, BLOCK)
    mask = (row[:, None] < m) & (col[None, :] < N)
    x = tl.load(X + row[:, None] * N + col[None, :], mask, other=0).to(tl.float32)
    weight = tl.load(W + col, col < N, other=0).to(tl.float32)
    variance = tl.sum(x * x, axis=1) / N
    inv = tl.rsqrt(variance + EPS)
    value = x * inv[:, None] * weight[None, :]
    tl.store(O + row[:, None] * N + col[None, :], value, mask)

def run(x, weight, out, eps):
"""
        + f"""    m, n = x.shape
    kernel[(triton.cdiv(m, {config["rows"]}),)](x, weight, out, m, n, eps, BLOCK=triton.next_power_of_2(n), ROWS={config["rows"]})
"""
    )
