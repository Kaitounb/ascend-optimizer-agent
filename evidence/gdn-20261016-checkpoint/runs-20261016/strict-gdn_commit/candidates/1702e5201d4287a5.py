# SPDX-License-Identifier: Apache-2.0
# Kernel extracted from frozen project source; origin.json records commit and SHA256.
import triton
import triton.language as tl

@triton.jit
def _commit_kernel(
    S,
    R,
    IDs,
    CU,
    Prefix,
    SS: tl.constexpr,
    RS: tl.constexpr,
    W: tl.constexpr,
    HV: tl.constexpr,
    KD: tl.constexpr,
    VD: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    row, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot = tl.load(IDs + row).to(tl.int64)
    available = tl.load(CU + row + 1) - tl.load(CU + row)
    length = tl.minimum(tl.maximum(tl.load(Prefix + row), 0), available)
    kk = tl.arange(0, BK)
    vv = tile * BV + tl.arange(0, BV)
    mask = (vv[:, None] < VD) & (kk[None, :] < KD)
    if slot > 0 and length > 0:
        offset = slot * SS + head * VD * KD + vv[:, None] * KD + kk[None, :]
        state = tl.load(S + offset, mask=mask, other=0).to(tl.float32)
        for step in range(W):
            if step < length:
                record = slot * RS + (head * W + step) * (VD + KD + 1)
                key = tl.load(R + record + VD + kk, mask=kk < KD, other=0)
                correction = tl.load(R + record + vv, mask=vv < VD, other=0)
                decay = tl.load(R + record + VD + KD)
                state = state * decay
                state = state + correction[:, None] * key[None, :]
        tl.store(S + offset, state, mask=mask)

def run(state, records, ids, offsets, prefixes):
    _, heads, vd, kd = state.shape
    window = records.shape[2]
    tile = (32 if kd > 128 else 64) if 16 == 0 else (32 if kd > 128 else 16)
    _commit_kernel[(ids.numel(), heads, triton.cdiv(vd, tile))](
        state, records, ids, offsets, prefixes,
        SS=state.stride(0), RS=records.stride(0), W=window, HV=heads, KD=kd, VD=vd,
        BK=triton.next_power_of_2(kd), BV=tile, enable_fp_fusion=False)
