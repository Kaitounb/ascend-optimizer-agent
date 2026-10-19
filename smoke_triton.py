"""Compile and validate a minimal Triton-Ascend operator on the selected NPU."""

import json

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def vector_add(x, y, out, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    a = tl.load(x + offsets, mask=mask, other=0)
    b = tl.load(y + offsets, mask=mask, other=0)
    tl.store(out + offsets, a + b, mask=mask)


def main():
    assert torch.npu.is_available(), "NPU unavailable"
    results = []
    for n in (1, 1023, 1024, 65537):
        x_cpu = torch.arange(n, dtype=torch.float32) / 1024
        y_cpu = -x_cpu / 2
        x, y = x_cpu.npu(), y_cpu.npu()
        out = torch.empty_like(x)
        vector_add[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), x_cpu + y_cpu, rtol=0, atol=0)
        results.append({"n": n, "passed": True})
    print(
        json.dumps(
            {
                "status": "TRITON_SMOKE_PASS",
                "device": torch.npu.get_device_name(0),
                "cases": results,
            }
        )
    )


if __name__ == "__main__":
    main()
