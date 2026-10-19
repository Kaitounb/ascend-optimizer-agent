# Third-party sources

The frozen GDN kernels in `ascend_optimizer/project_sources/` and project regression sources in `project_regression/` come from Kaitounb/vllm-ascend, commit `622725cd44aa47186dc1f1c9f08485e9013fff7a`. Their original SPDX Apache-2.0 headers are retained. The license text is included in `COPYING.vllm-ascend`; file hashes and original paths are recorded in the corresponding `origin.json` files.

Generated GDN candidate bodies are derived from these frozen kernels and retain the Apache-2.0 SPDX header. The optimizer's orchestration and evaluators are implemented independently for this workspace.
