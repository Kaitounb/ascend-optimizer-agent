"""Export measured winners as a reviewable original-project module and diff.

This does not edit the user's checkout. Public API validation stays unchanged;
the tuned path is limited to the production geometry measured by this project.
"""

import difflib
import json
from pathlib import Path

from ascend_optimizer.engine import atomic_json
from ascend_optimizer.gdn_templates import SOURCE_ROOT, kernel_source, validate_config


def build(configs):
    original = (SOURCE_ROOT / "gdn_recover.py").read_text()
    content = original
    for operator, config in configs.items():
        if operator not in ("gdn_verify", "gdn_commit"):
            raise ValueError("Unsupported project operator")
        validate_config(config)
        name = "_commit_kernel" if operator == "gdn_commit" else "_verify_kernel"
        if config.get("unroll"):
            source = kernel_source(name)
            unrolled = source.replace(
                "def " + name + "(", "def " + name + "_unrolled("
            ).replace("range(W)", "tl.static_range(W)")
            content = content.replace(source, source + "\n\n" + unrolled, 1)
        function = (
            "gdn_recover_commit" if operator == "gdn_commit" else "gdn_recover_verify"
        )
        start = content.index("def " + function + "(")
        end = content.find("\ndef ", start + 1)
        end = len(content) if end < 0 else end
        section = content[start:end]
        policy = (
            "    # Measured 910B4 short-window path; other geometry keeps the original kernel.\n"
            "    use_tuned = window == 4 and heads == 48 and vd == 128 and kd == 128 and indices.numel() <= 32\n"
        )
        tile = config["value_tile"] or 64
        kernel = name + "_unrolled" if config.get("unroll") else name
        section = section.replace(
            "    " + name + "[(",
            policy + f"    value_tile = {tile} if use_tuned else value_tile\n"
            f"    kernel = {kernel} if use_tuned else {name}\n"
            "    kernel[(",
            1,
        )
        content = content[:start] + section + content[end:]
    return original, content


def adapter(source, operator):
    # The Notebook has Triton but no full vLLM install; only replace its import facade.
    text = source.replace(
        "from vllm.triton_utils import tl, triton",
        "import triton\nimport triton.language as tl",
    )
    if operator == "gdn_commit":
        text += "\ndef run(state, records, ids, offsets, prefixes):\n    gdn_recover_commit(state, records, ids, offsets, prefixes)\n"
    else:
        text += "\ndef run(state,q,k,v,g,beta,ids,offsets,out,records):\n    gdn_recover_verify(q,k,v,g,beta,state,records,ids,offsets,out=out)\n"
    return text


def build_batch_compatibility():
    original = (SOURCE_ROOT / "gdn_recover_batch.py").read_text()
    old = """        if length > 0:
            offset = conv_offset + slot * CS + (length - 1 + hh[:, None]) * CHANNELS + cc[None, :]
        else:
            offset = history_offset + slot * HS + hh[:, None] * CHANNELS + cc[None, :]
        history = tl.load(ConvBase + offset, mask=mask, other=0)
        destination = conv_offset + slot * CS + hh[:, None] * CHANNELS + cc[None, :]
        tl.store(ConvBase + destination, history, mask=mask)"""
    new = """        destination = conv_offset + slot * CS + hh[:, None] * CHANNELS + cc[None, :]
        # CANN 8.5 cannot lower addptr of an scf.if-yielded offset tensor.
        # Keep each address and memory access in its originating branch.
        if length > 0:
            offset = conv_offset + slot * CS + (length - 1 + hh[:, None]) * CHANNELS + cc[None, :]
            history = tl.load(ConvBase + offset, mask=mask, other=0)
            tl.store(ConvBase + destination, history, mask=mask)
        else:
            offset = history_offset + slot * HS + hh[:, None] * CHANNELS + cc[None, :]
            history = tl.load(ConvBase + offset, mask=mask, other=0)
            tl.store(ConvBase + destination, history, mask=mask)"""
    if original.count(old) != 1:
        raise ValueError(
            "Frozen batch source no longer matches compatibility transform"
        )
    return original, original.replace(old, new, 1)


def main():
    project = Path(__file__).resolve().parent
    root = project / "runs-20261016"
    qualification = json.loads((root / "qualification.json").read_text())
    configs = {
        name: row["config"]
        for name, row in qualification.items()
        if row.get("qualified")
    }
    out = root / "project-integration"
    out.mkdir(exist_ok=True)
    original, content = build(configs)
    (out / "gdn_recover.py").write_text(content)
    diff = "".join(
        difflib.unified_diff(
            original.splitlines(True),
            content.splitlines(True),
            fromfile="a/vllm_ascend/ops/triton/mamba/gdn_recover.py",
            tofile="b/vllm_ascend/ops/triton/mamba/gdn_recover.py",
        )
    )
    (out / "gdn_recover.patch").write_text(diff)
    batch_original, batch_content = build_batch_compatibility()
    (out / "gdn_recover_batch.py").write_text(batch_content)
    (out / "gdn_recover_batch.patch").write_text(
        "".join(
            difflib.unified_diff(
                batch_original.splitlines(True),
                batch_content.splitlines(True),
                fromfile="a/vllm_ascend/ops/triton/mamba/gdn_recover_batch.py",
                tofile="b/vllm_ascend/ops/triton/mamba/gdn_recover_batch.py",
            )
        )
    )
    for operator in configs:
        (out / (operator + "_baseline.py")).write_text(adapter(original, operator))
        (out / (operator + "_candidate.py")).write_text(adapter(content, operator))
    atomic_json(
        out / "selection.json",
        {
            "configs": configs,
            "scope": "48 heads, 128x128, window4, batch<=32; public API unchanged",
        },
    )
    print("PROJECT_CANDIDATE", configs, str(out), flush=True)


if __name__ == "__main__":
    main()
