# SPDX-License-Identifier: Apache-2.0
# Kernel extracted from frozen project source; origin.json records commit and SHA256.
import triton
import triton.language as tl

@triton.jit
def _verify_kernel(
    S,
    Q,
    K,
    V,
    G,
    Beta,
    IDs,
    CU,
    Out,
    R,
    SS: tl.constexpr,
    RS: tl.constexpr,
    QT: tl.constexpr,
    KT: tl.constexpr,
    VT: tl.constexpr,
    GT: tl.constexpr,
    BT: tl.constexpr,
    W: tl.constexpr,
    NK: tl.constexpr,
    HV: tl.constexpr,
    KD: tl.constexpr,
    VD: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SCALE: tl.constexpr,
):
    row, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot = tl.load(IDs + row).to(tl.int64)
    start = tl.load(CU + row).to(tl.int64)
    end = tl.load(CU + row + 1).to(tl.int64)
    kk = tl.arange(0, BK)
    vv = tile * BV + tl.arange(0, BV)
    mask = (vv[:, None] < VD) & (kk[None, :] < KD)
    if slot > 0 and end > start:
        state_offset = slot * SS + head * VD * KD + vv[:, None] * KD + kk[None, :]
        state = tl.load(S + state_offset, mask=mask, other=0).to(tl.float32)
        qk_head = head // (HV // NK)
        for step in range(W):
            token = start + step
            if token < end:
                key = tl.load(K + token * KT + qk_head * KD + kk, mask=kk < KD, other=0).to(tl.float32)
                query = tl.load(Q + token * QT + qk_head * KD + kk, mask=kk < KD, other=0).to(tl.float32)
                query = query * SCALE
                value = tl.load(V + token * VT + head * VD + vv, mask=vv < VD, other=0).to(tl.float32)
                decay = tl.exp(tl.load(G + token * GT + head).to(tl.float32))
                beta = tl.load(Beta + token * BT + head).to(tl.float32)
                state = state * decay
                correction = (value - tl.sum(state * key[None, :], 1)) * beta
                state = state + correction[:, None] * key[None, :]
                output = tl.sum(state * query[None, :], 1)
                tl.store(Out + (token * HV + head) * VD + vv, output, mask=vv < VD)
                record = slot * RS + (head * W + step) * (VD + KD + 1)
                tl.store(R + record + vv, correction, mask=vv < VD)
                if tile == 0:
                    tl.store(R + record + VD + kk, key, mask=kk < KD)
                    tl.store(R + record + VD + KD, decay)
    elif end > start:
        for step in range(W):
            token = start + step
            if token < end:
                tl.store(Out + (token * HV + head) * VD + vv, 0, mask=vv < VD)

def run(state, q, k, v, g, beta, ids, offsets, out, records):
    _, heads, vd, kd = state.shape
    window = records.shape[2]
    tile = (32 if kd > 128 else 64) if 16 == 0 else (32 if kd > 128 else 16)
    _verify_kernel[(ids.numel(), heads, triton.cdiv(vd, tile))](
        state,q,k,v,g,beta,ids,offsets,out,records,
        SS=state.stride(0),RS=records.stride(0),QT=q.stride(0),KT=k.stride(0),
        VT=v.stride(0),GT=g.stride(0),BT=beta.stride(0),W=window,NK=q.shape[1],HV=heads,
        KD=kd,VD=vd,BK=triton.next_power_of_2(kd),BV=tile,SCALE=kd**-0.5,enable_fp_fusion=False)
