"""Frozen kernels with value-axis tiling and bounded short-window loop unrolling."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent / "project_sources"


def kernel_source(name):
    origin = json.loads((SOURCE_ROOT / "origin.json").read_text())
    path = SOURCE_ROOT / "gdn_recover.py"
    content = path.read_text()
    if (
        hashlib.sha256(path.read_bytes()).hexdigest()
        != origin["files"][path.name]["sha256"]
    ):
        raise ValueError("Frozen project kernel source changed")
    tree = ast.parse(content)
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    start = min([node.lineno, *[d.lineno for d in node.decorator_list]]) - 1
    return "\n".join(content.splitlines()[start : node.end_lineno]) + "\n"


def specs():
    common = {
        "dtypes": ["float32", "bfloat16"],
        "validation_shapes": [
            [6, 6, 17, 33, 3],
            [5, 4, 96, 64, 6],
            [1, 2, 256, 256, 1],
        ],
        "benchmark_shapes": [
            [1, 48, 128, 128, 4],
            [8, 48, 128, 128, 4],
            [32, 48, 128, 128, 4],
        ],
        "baseline_config": {"value_tile": 0},
        "search_configs": (
            [{"value_tile": v, "unroll": True} for v in (64, 32, 16)]
            + [{"value_tile": v} for v in (16, 32, 64, 128)]
        ),
        "source_origin": json.loads((SOURCE_ROOT / "origin.json").read_text()),
        "min_decisions": 3,
        "unroll_policy": "unroll=True statically expands only windows <=4; larger windows use the unchanged original loop",
        "qk_heads": "heads/3 when divisible by 3, otherwise heads/2; production 48 value heads / 16 QK heads",
        "metadata": "Padded slots, reordered and null IDs, mixed query lengths; prefix counts are processed input counts",
    }
    return {
        name: {
            **common,
            "operator": name,
            "description": "Project GDN "
            + (
                "checkpoint verification and replay recording"
                if name.endswith("verify")
                else "in-place accepted-prefix state replay commit"
            ),
        }
        for name in ("gdn_verify", "gdn_commit")
    }


def validate_shapes(task):
    if any(d not in ("float32", "bfloat16") for d in task["dtypes"]):
        raise ValueError(
            "GDN inputs must be FP32/BF16; states and records are always FP32"
        )
    for field in ("validation_shapes", "benchmark_shapes"):
        if not task.get(field):
            raise ValueError("GDN shapes cannot be empty")
        for s in task[field]:
            if len(s) != 5 or any(type(n) is not int or n < 1 for n in s):
                raise ValueError(
                    "GDN shape is [batch, heads, value_dim, key_dim, window]"
                )
            b, h, v, k, w = s
            if b > 64 or h > 64 or max(v, k) > 256 or w > 16 or h % 2:
                raise ValueError("GDN shape exceeds supported geometry")


def validate_config(config):
    if (
        not {"value_tile"} <= set(config) <= {"value_tile", "unroll"}
        or type(config["value_tile"]) is not int
        or config["value_tile"] not in (0, 8, 16, 32, 64, 128)
    ):
        raise ValueError(
            "GDN config requires value_tile in 0(project default),8,16,32,64,128"
        )
    if "unroll" in config and type(config["unroll"]) is not bool:
        raise ValueError("unroll must be a boolean")


def render(operator, config):
    validate_config(config)
    header = "# SPDX-License-Identifier: Apache-2.0\n# Kernel extracted from frozen project source; origin.json records commit and SHA256.\nimport triton\nimport triton.language as tl\n\n"
    tile = config["value_tile"]
    choose = f"    tile = (32 if kd > 128 else 64) if {tile} == 0 else (32 if kd > 128 else {tile})\n"
    name = "_commit_kernel" if operator == "gdn_commit" else "_verify_kernel"
    source = kernel_source(name)
    launch = name
    if config.get("unroll", False):
        source += "\n" + kernel_source(name).replace(
            "def " + name + "(", "def " + name + "_unrolled("
        ).replace("range(W)", "tl.static_range(W)")
        choose += f"    kernel = {name}_unrolled if window <= 4 else {name}\n"
        launch = "kernel"
    if operator == "gdn_commit":
        return (
            header
            + source
            + """
def run(state, records, ids, offsets, prefixes):
    _, heads, vd, kd = state.shape
    window = records.shape[2]
"""
            + choose
            + f"""    {launch}[(ids.numel(), heads, triton.cdiv(vd, tile))](
        state, records, ids, offsets, prefixes,
        SS=state.stride(0), RS=records.stride(0), W=window, HV=heads, KD=kd, VD=vd,
        BK=triton.next_power_of_2(kd), BV=tile, enable_fp_fusion=False)
"""
        )
    return (
        header
        + source
        + """
def run(state, q, k, v, g, beta, ids, offsets, out, records):
    _, heads, vd, kd = state.shape
    window = records.shape[2]
"""
        + choose
        + f"""    {launch}[(ids.numel(), heads, triton.cdiv(vd, tile))](
        state,q,k,v,g,beta,ids,offsets,out,records,
        SS=state.stride(0),RS=records.stride(0),QT=q.stride(0),KT=k.stride(0),
        VT=v.stride(0),GT=g.stride(0),BT=beta.stride(0),W=window,NK=q.shape[1],HV=heads,
        KD=kd,VD=vd,BK=triton.next_power_of_2(kd),BV=tile,SCALE=kd**-0.5,enable_fp_fusion=False)
"""
    )
