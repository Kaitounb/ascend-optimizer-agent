"""Run the project's original kernel tests without installing the full vLLM engine.

Only the triton import shim and package namespaces are supplied; kernels and tests
are frozen project files. This does not emulate the inference engine.
"""

import importlib.util
import ast
from dataclasses import dataclass
from typing import Any
from pathlib import Path
import sys
import types


def module_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    root = Path(__file__).resolve().parent
    import triton
    import triton.language as tl
    import torch_npu  # noqa: F401 - registers the real NPU backend on torch
    import pytest

    for name in (
        "vllm",
        "vllm_ascend",
        "vllm_ascend.ops",
        "vllm_ascend.ops.triton",
        "vllm_ascend.ops.triton.mamba",
    ):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    utilities = types.ModuleType("vllm.triton_utils")
    utilities.triton, utilities.tl = triton, tl
    sys.modules["vllm.triton_utils"] = utilities
    source = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else root / "ascend_optimizer/project_sources/gdn_recover.py"
    )
    module_file("vllm_ascend.ops.triton.mamba.gdn_recover", source)
    batch_source = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else root / "ascend_optimizer/project_sources/gdn_recover_batch.py"
    )
    module_file("vllm_ascend.ops.triton.mamba.gdn_recover_batch", batch_source)
    # Exercise the exact lifecycle classes, without loading the unrelated vLLM platform/config stack.
    import torch

    torch.set_num_threads(1)
    context = types.ModuleType("vllm_ascend.ops.gdn_recover")
    context.__dict__.update(torch=torch, dataclass=dataclass, Any=Any)
    primitive = sys.modules["vllm_ascend.ops.triton.mamba.gdn_recover"]
    context.gdn_recover_commit = primitive.gdn_recover_commit
    context.gdn_commit_conv_history = primitive.gdn_commit_conv_history
    sys.modules[context.__name__] = context
    tree = ast.parse((root / "project_regression/context_source.py").read_text())
    classes = [
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        and n.name in ("GDNRecoverEntry", "GDNRecoverContext")
    ]
    exec(
        compile(
            ast.Module(body=classes, type_ignores=[]), "frozen_project_context", "exec"
        ),
        context.__dict__,
    )
    code = pytest.main(
        [str(root / "project_regression"), "-q", "--tb=short", "-o", "addopts="]
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
