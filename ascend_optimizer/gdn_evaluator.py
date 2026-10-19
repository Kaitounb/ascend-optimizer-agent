"""Project GDN evaluator: independent CPU recurrence + original-NPU bitwise control."""

from __future__ import annotations

import json
import math
from pathlib import Path

from . import worker
from .gdn_templates import render


def padded(torch, shape, dtype, padding=31, offset=7):
    inner = math.prod(shape[1:])
    raw = torch.full((offset + shape[0] * (inner + padding) + 11,), 0.125, dtype=dtype)
    strides = [inner + padding] + [
        math.prod(shape[i + 1 :]) for i in range(1, len(shape))
    ]
    return raw, raw.as_strided(shape, strides, offset)


class Fixture:
    def __init__(self, task, shape, dtype_name, device, seed, mode="full"):
        t = worker.torch_module(device)
        self.t, self.task = t, task
        b, h, vd, kd, w = shape
        self.shape = shape
        gen = t.Generator().manual_seed(seed)
        slots = b + 3
        self.cpu_raw, self.cpu_views, self.meta = {}, {}, {}
        self.add("S", (slots, h, vd, kd), t.float32)
        self.add("R", (slots, h, w, vd + kd + 1), t.float32, padding=53, offset=13)
        self.cpu_views["S"].copy_(t.randn((slots, h, vd, kd), generator=gen) * 0.03)
        self.cpu_views["R"].copy_(
            t.randn((slots, h, w, vd + kd + 1), generator=gen) * 0.02
        )
        self.cpu_views["R"][..., -1].copy_(
            t.rand((slots, h, w), generator=gen) * 0.1 + 0.85
        )
        ids = list(range(1, b + 1))[::-1]
        lengths = [w] * b
        if mode == "edge":
            lengths = [(w, 1, max(w - 1, 1), 0)[i % 4] for i in range(b)]
            if b > 2:
                ids[-2:] = [0, -1]
        self.ids, self.lengths = ids, lengths
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        prefixes = [(i % (w + 3)) - 1 for i in range(b)] if mode == "edge" else [w] * b
        index_dtype = t.int64 if seed % 2 else t.int32
        for name, values in (("IDs", ids), ("CU", offsets), ("P", prefixes)):
            tensor = t.tensor(values, dtype=index_dtype)
            self.cpu_raw[name] = tensor
            self.cpu_views[name] = tensor
        if task["operator"] == "gdn_verify":
            dtype = getattr(t, dtype_name)
            tokens, nk = offsets[-1], max(h // 3 if h % 3 == 0 else h // 2, 1)
            for name in ("Q", "K"):
                self.add(name, (tokens, nk, kd), dtype, padding=9)
                value = t.nn.functional.normalize(
                    t.randn((tokens, nk, kd), generator=gen), dim=-1
                ).to(dtype)
                self.cpu_views[name].copy_(value)
            self.add("V", (tokens, h, vd), dtype, padding=9)
            self.cpu_views["V"].copy_(
                (t.randn((tokens, h, vd), generator=gen) * 0.2).to(dtype)
            )
            for name in ("G", "Beta"):
                self.add(name, (tokens, h), t.float32, padding=9)
            self.cpu_views["G"].copy_(
                -(t.rand((tokens, h), generator=gen) * 0.3 + 0.05)
            )
            self.cpu_views["Beta"].copy_(t.rand((tokens, h), generator=gen) * 0.8)
            size = tokens * h * vd
            raw = t.full((size + 64,), 0.125, dtype=dtype)
            self.cpu_raw["O"] = raw
            self.cpu_views["O"] = raw[32 : 32 + size].view(tokens, h, vd)
        for name, view in self.cpu_views.items():
            self.meta[name] = (
                tuple(view.shape),
                tuple(view.stride()),
                view.storage_offset(),
            )
        self.raw = {name: value.to(device) for name, value in self.cpu_raw.items()}
        self.views = {
            name: self.raw[name].as_strided(*geometry)
            for name, geometry in self.meta.items()
        }
        self.mutable = ("S",) if task["operator"] == "gdn_commit" else ("R", "O")
        self.initial = {name: self.raw[name].clone() for name in self.mutable}
        names = (
            ("S", "R", "IDs", "CU", "P")
            if task["operator"] == "gdn_commit"
            else ("S", "Q", "K", "V", "G", "Beta", "IDs", "CU", "O", "R")
        )
        self.args = [self.views[name] for name in names]

    def add(self, name, shape, dtype, **kwargs):
        self.cpu_raw[name], self.cpu_views[name] = padded(
            self.t, shape, dtype, **kwargs
        )

    def reset(self):
        for name in self.mutable:
            self.raw[name].copy_(self.initial[name])

    def oracle(self):
        t = self.t
        expected = {name: raw.clone() for name, raw in self.cpu_raw.items()}
        views = {
            name: expected[name].as_strided(*geometry)
            for name, geometry in self.meta.items()
        }
        masks = {
            name: t.zeros(self.cpu_raw[name].numel(), dtype=t.bool)
            for name in self.mutable
        }
        _, h, vd, kd, w = self.shape
        if self.task["operator"] == "gdn_commit":
            for row, slot in enumerate(self.ids):
                count = min(max(int(self.cpu_views["P"][row]), 0), self.lengths[row])
                if slot > 0 and count:
                    state = self.cpu_views["S"][slot].clone()
                    for step in range(count):
                        record = self.cpu_views["R"][slot, :, step]
                        key, correction, decay = (
                            record[:, vd : vd + kd],
                            record[:, :vd],
                            record[:, -1],
                        )
                        state = state * decay[:, None, None]
                        state = state + correction[:, :, None] * key[:, None, :]
                    views["S"][slot] = state
                    start = views["S"].storage_offset() + slot * views["S"].stride(0)
                    masks["S"][start : start + h * vd * kd] = True
        else:
            headmap = t.arange(h) // (h // self.cpu_views["K"].shape[1])
            start_token = 0
            for slot, length in zip(self.ids, self.lengths):
                state = self.cpu_views["S"][slot].clone() if slot > 0 else None
                for step in range(length):
                    token = start_token + step
                    if slot > 0:
                        key = self.cpu_views["K"][token, headmap].float()
                        query = self.cpu_views["Q"][token, headmap].float() * (kd**-0.5)
                        decay = self.cpu_views["G"][token].float().exp()
                        state = state * decay[:, None, None]
                        correction = (
                            self.cpu_views["V"][token].float()
                            - (state * key[:, None, :]).sum(-1)
                        ) * self.cpu_views["Beta"][token, :, None].float()
                        state = state + correction[:, :, None] * key[:, None, :]
                        views["O"][token] = (
                            (state * query[:, None, :]).sum(-1).to(views["O"].dtype)
                        )
                        views["R"][slot, :, step, :vd] = correction
                        views["R"][slot, :, step, vd : vd + kd] = key
                        views["R"][slot, :, step, -1] = decay
                        for head in range(h):
                            off = (
                                views["R"].storage_offset()
                                + slot * views["R"].stride(0)
                                + (head * w + step) * (vd + kd + 1)
                            )
                            masks["R"][off : off + vd + kd + 1] = True
                    else:
                        views["O"][token].zero_()
                    begin = 32 + token * h * vd
                    masks["O"][begin : begin + h * vd] = True
                start_token += length
        return expected, masks


def original(task):
    path = Path.cwd() / ("original_" + task["operator"] + ".py")
    source = render(task["operator"], {"value_tile": 0})
    if not path.exists() or path.read_text() != source:
        path.write_text(source)
    return worker.load_module(path)


def validate(module, task, device):
    t = worker.torch_module(device)
    base = original(task)
    cases = []
    shapes = list(
        dict.fromkeys(
            tuple(s) for s in task["validation_shapes"] + task["benchmark_shapes"]
        )
    )
    for shape in shapes:
        for dtype in task["dtypes"]:
            for mode in ("full", "edge"):
                # Two distinct seeds also exercise both int32 and int64 metadata.
                for seed in (task["seed"], task["seed"] + 1):
                    fixture = Fixture(task, list(shape), dtype, device, seed, mode)
                    expected, masks = fixture.oracle()
                    fixture.reset()
                    base.run(*fixture.args)
                    t.npu.synchronize()
                    golden = {name: fixture.raw[name].cpu() for name in fixture.mutable}
                    fixture.reset()
                    module.run(*fixture.args)
                    t.npu.synchronize()
                    for name, raw in fixture.raw.items():
                        actual = raw.cpu()
                        if name not in fixture.mutable:
                            if not t.equal(
                                actual.view(t.uint8),
                                fixture.cpu_raw[name].view(t.uint8),
                            ):
                                raise AssertionError(
                                    f"GDN modified immutable input {name}"
                                )
                            continue
                        if not t.equal(
                            actual.view(t.uint8), golden[name].view(t.uint8)
                        ):
                            raise AssertionError(
                                f"GDN candidate differs bitwise from original NPU kernel: {name}, shape={shape},dtype={dtype},mode={mode}"
                            )
                        mask = masks[name]
                        if not t.equal(
                            actual[~mask].view(t.uint8),
                            fixture.cpu_raw[name][~mask].view(t.uint8),
                        ):
                            raise AssertionError(
                                f"GDN padding/null/unselected region modified: {name}"
                            )
                        rtol, atol = (
                            (0.002, 0.002)
                            if dtype == "bfloat16" and name == "O"
                            else (5e-5, 2e-5)
                        )
                        t.testing.assert_close(
                            actual,
                            expected[name],
                            rtol=rtol,
                            atol=atol,
                            equal_nan=False,
                        )
                    cases.append(
                        {
                            "shape": list(shape),
                            "dtype": dtype,
                            "mode": mode,
                            "seed": seed,
                            "cpu_oracle": True,
                            "original_npu_bitwise": True,
                        }
                    )
    return {
        "passed": True,
        "cases": cases,
        "checks": [
            "independent CPU recurrence",
            "original NPU bitwise",
            "all backing storage guards",
            "immutable inputs",
            "mixed clipped prefixes",
            "null/reordered IDs",
            "int32/int64 metadata",
        ],
    }


def compile_kernel(module, task, device):
    fixture = Fixture(
        task, task["benchmark_shapes"][0], task["dtypes"][0], device, task["seed"]
    )
    module.run(*fixture.args)
    worker.torch_module(device).npu.synchronize()
    return {
        "compiled_and_launched": True,
        "source": "Frozen project kernel, real NPU launch",
    }


def benchmark(base, candidate, task, device):
    t = worker.torch_module(device)
    checked = {
        "baseline": validate(base, task, device),
        "candidate": validate(candidate, task, device),
    }
    m = task["measurement"]
    cases = []
    for shape in task["benchmark_shapes"]:
        for dtype in task["dtypes"]:
            fixture = Fixture(task, shape, dtype, device, task["seed"])
            for _ in range(m["warmup"]):
                fixture.reset()
                base.run(*fixture.args)
                fixture.reset()
                candidate.run(*fixture.args)
            t.npu.synchronize()
            pairs = []
            for r in range(m["rounds"]):
                order = (
                    ["baseline", "candidate", "candidate", "baseline"]
                    if r % 2 == 0
                    else ["candidate", "baseline", "baseline", "candidate"]
                )
                samples = {"baseline": [], "candidate": []}
                for label in order:
                    fixture.reset()  # Not inside timing; stateful versions start from identical checkpoint.
                    module = base if label == "baseline" else candidate
                    samples[label].append(
                        worker.measure(
                            lambda: module.run(*fixture.args), m["iterations"]
                        )
                    )
                pairs.append(
                    {
                        "baseline_us": sum(samples["baseline"]) / 2,
                        "candidate_us": sum(samples["candidate"]) / 2,
                        "order": "ABBA" if r % 2 == 0 else "BAAB",
                    }
                )
            # Ensure timing did not silently produce divergent/invalid state.
            for name in fixture.mutable:
                if not bool(t.isfinite(fixture.raw[name]).all()):
                    raise AssertionError(
                        "Nonfinite state during repeated GDN measurement"
                    )
            cases.append({"case": {"shape": shape, "dtype": dtype}, "pairs": pairs})
    return {
        "correctness": checked,
        "cases": cases,
        "timing_scope": "Original project NPU kernel vs tuned tile; reset identical backing state before every event segment outside timing; allocations, CPU oracle, compile, warmup excluded",
    }


def profile(module, task, device, directory):
    import torch_npu

    t = worker.torch_module(device)
    fixture = Fixture(
        task, task["benchmark_shapes"][-1], task["dtypes"][0], device, task["seed"]
    )
    for _ in range(5):
        fixture.reset()
        module.run(*fixture.args)
    t.npu.synchronize()
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    p = torch_npu.profiler
    with p.profile(
        activities=[p.ProfilerActivity.CPU, p.ProfilerActivity.NPU], record_shapes=True
    ) as prof:
        for _ in range(5):
            module.run(*fixture.args)
        t.npu.synchronize()
    trace = out / "trace.json"
    prof.export_chrome_trace(str(trace))
    summary = worker.summarize_trace(trace)
    (out / "trace_summary.json").write_text(json.dumps(summary, indent=2))
    return {
        "trace": str(trace),
        "summary": summary,
        "invocations": 5,
        "note": "Profiler is diagnostic, not acceptance timing",
    }


def handle(request):
    task, device = request["task"], request.get("device", "npu:0")
    module = worker.load_module(request["candidate"])
    action = request["action"]
    if action == "compile":
        return compile_kernel(module, task, device)
    if action == "validate":
        return validate(module, task, device)
    if action == "benchmark":
        return benchmark(worker.load_module(request["baseline"]), module, task, device)
    if action == "profile":
        return profile(module, task, device, request["directory"])
    raise ValueError("Unknown GDN action")
