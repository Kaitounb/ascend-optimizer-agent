# 验证记录（2026-10-15，北京时间）

## 环境

Ascend 910B4 / 32GiB HBM，aarch64，Python 3.11.14，CANN 8.5.0，torch 2.9.0+cpu，torch_npu 2.9.0，triton-ascend 3.2.0。

## 实卡实验

| 算子 | 最佳配置 | 相对初始基线的独立复测 | 探索性 bootstrap 95% 区间 |
|---|---|---:|---|
| vector_add | block=1024, unroll=1 | 1.2307× | [1.2249, 1.2378] |
| rms_norm | triton, rows=1 | 2.0387× | [2.0160, 2.0843] |

RMSNorm 对照为 PyTorch eager 分解表达式，不能将 2.04× 解读为超过厂商最优 RMSNorm 算子。vector add 汇总包含较小和较大 shape，以及 float32/float16；小 shape 的延迟基本不变。这里只是当前这次单卡微基准，不代表整模型收益或硬件极限。

- 两类算子都经过 CPU 参考、两组种子、输入不可变、输出 guard 与 NaN 检查。
- 大 tile/展开和 RMSNorm 多行配置触发真实 UB overflow 时，候选被拒绝，最佳版本保留。
- 故意将加法写成减法的 kernel 被真实正确性评测拒绝。
- 两类算子的实际 CPU/NPU profiler trace 与事件摘要均导出成功。初版 profiler key_averages 兼容错误已修复；原始失败记录保留。
- 修复 profiler 时复用了已有工作区的最佳版本并重新完成预检，没有重置搜索结果。
- 原始配对计时已在 Mac 离线重新计算统计，并检查 task / 候选源码 SHA256 一致。

详细报告：[vector add](examples/verified-910b4/vector_add/REPORT.md)、[RMSNorm](examples/verified-910b4/rms_norm/REPORT.md)。

## Agent 与 API 验证

- 本机 Python 3.14 与远端 Python 3.11：最终均为 **16 项测试通过**，覆盖状态机、过期证据、源码篡改、恢复、超时终止、密钥环境隔离与 API 协议。
- Chat Completions / Responses 均通过本机临时 HTTP 服务的请求/响应测试；这些测试的模型和 worker 为替身。
- 另运行了完整 agent + 本地 HTTP 模型替身 + **真实 NPU worker**：提出配置 → 编译 → 验证 → 配对测速 → 裁决 → 独立复测，输出 `HTTP_AGENT_REAL_NPU_PASS`。
- 10 月 15 日的上述实验没有调用真实模型；10 月 16 日已接入真实 DeepSeek 服务，另见下面的分日记录。

## 原始证据

[完整原始证据 ZIP](evidence/910b4-20261015/ascend-optimizer-evidence-20261015.zip)

`SHA256: a8717d9f9eb8993b308d3dbdb72dacf76a4a418459f3aabd34cb68e125750728`

解压副本保存在 `evidence/910b4-20261015/`：候选源码、所有计时、接受/拒绝轨迹、编译错误、trace、最终测试日志和 API/NPU 集成记录。可运行 `python3 verify_artifacts.py` 离线复核。


## 2026-10-16：真实 API 与投机解码 GDN

API 配置使用 `https://api.deepseek.com` 和请求模型名 `deepseek-v4-flash`，官方服务实际返回 `deepseek-flash`。首个强制工具调用已成功；连接证据保存在 `evidence/real-api-20261016/connectivity.json`。密钥保存在仓库外的 0600 文件，未打包。

冻结的项目内核来自公开提交 `622725cd44aa47186dc1f1c9f08485e9013fff7a`；已从公开 GitHub URL 取回并验证 SHA256 一致，证据见 `evidence/real-api-20261016/public-source-proof.json`。

第一轮真实模型 GDN commit 搜索完成了三个成功候选的编译、正确性、配对计时和拒绝决策。tile=64 无显著收益，tile=32/16 出现退化，tile=128 在真实编译时 UB 溢出；最终保留原实现。独立复测 0.9990x，未建立提速。原始记录保存在 `evidence/gdn-20261016-checkpoint/`，检查点 ZIP SHA256 为 `c9d3444658c01f7f725c424819131cd8a5043f3c2a7abb8b4522c8ea68cbb0dd`。

这一早期检查点的 Agent 单元测试为 21/21 通过；原项目 pytest 当时因未安装 pytest 而未启动，不能计为通过。后续已补齐依赖，在新实例完成以下验证。


### 最终真实模型搜索

新实例为 Ascend 910B4 / 32GiB HBM，CANN 8.5、torch/torch_npu 2.9、triton-ascend 3.2。使用真实 DeepSeek 工具调用驱动两个任务，均返回 `complete`；服务端返回模型名称 `deepseek-flash`。最新一轮共 50 个 API 回合，提供商 usage 累计输入 1,098,932 token、输出 6,513 token（输入包含重复历史与缓存，不表示独立文本量或计费金额）。

| 任务 | 最终配置 | 对原始基线的新鲜配对复测 | 结论 |
|---|---|---:|---|
| gdn_commit | 原始 tile 策略，无静态展开 | 1.0004× | 未建立稳定改进，保留原实现 |
| gdn_verify | 原始 tile 策略，无静态展开 | 0.9971× | 未建立稳定改进，保留原实现 |

模型测试了 value tile 与窗口 <=4 的静态展开。成功编译的每项候选都经过独立 CPU 递推、原 NPU 内核逐位比对、整块 backing storage guard、输入不可变、两组 seed、空/负/重排 slot、截断前缀和 int32/int64 元数据验证。大 tile 或大展开触发的真实 UB 溢出、性能退化、模型工具顺序误用与纠正均保留。最新一轮保存 9 个候选裁决、11 份配对计时；验收阈值和原始基线在任务创建后固定。

报告：[GDN commit](examples/verified-gdn-910b4/gdn_commit/REPORT.md)、[GDN verify](examples/verified-gdn-910b4/gdn_verify/REPORT.md)。无候选通过初始搜索，因此后续收益确认和 held-out 性能实验未启动；不将它们计为已通过。

### 项目兼容性修复

原始冻结项目回归在当前镜像上为 **62 通过、5 失败**。五项失败都来自融合 `_batch_commit` 的 CANN 8.5 编译：卷积分支返回偏移张量，再计算指针时触发 `AddPtrOp produced by unsupported operation`，IR 中对应 `scf.if` 的 yield。

修复把地址对应的 load/store 保留在各自分支内。状态递推、tile、地址宽度、公开函数签名和缓存布局都保持原样。修复后的原项目相关测试为 **67/67 通过**，覆盖 CPU 合同检查、真实 NPU 内核、共享 KV 组、device-side sampled-token 上下文、图回放、卷积历史及 >8GiB 偏移。已经将这份实测通过的改动应用到本地项目 `upstream/vllm-ascend-fia/vllm_ascend/ops/triton/mamba/gdn_recover_batch.py`，尚未提交或推送。

修复后可在当前 CANN 8.5 镜像编译运行，本轮未对这一修复计算性能加速比。源码 SHA256 为 `c7a027e2f764199a956a8384f9a95678f762a471139a8ecd78bb65e6d31a8d26`。可审阅补丁见 [gdn_recover_batch.patch](examples/verified-gdn-910b4/gdn_recover_batch.patch)。

回归运行器使用真实 torch_npu / Triton，只为 vLLM import 路径提供命名空间和 Triton facade，并从冻结源加载原始上下文生命周期类。本轮覆盖 Agent 闭环与相关算子回归；整模型的吞吐和投机接受率尚未测量，也未复测生产 CANN 9.1 / 910B2 环境。

### 原始证据与交付检查

[本轮完整原始证据](evidence/gdn-20261016/ascend-gdn-agent-evidence-20261016.zip) 的 SHA256：`8c995de1e8f65570a91f7a5c5b94f9ad47444c12423c05b745f37fb59cf7dbaf`。

`python3 verify_project_evidence.py` 已在 Mac 离线重新计算所有配对统计，检查任务哈希、所有候选源码哈希、最终 best 与真实模型 usage。早期失败/部分运行检查点与最终完成记录分开保存。构建的 0.2.0 wheel 已在干净的临时目标目录离线安装，验证四个任务及冻结源码资源齐全，记录见 `evidence/package-smoke.json`。


最终交付版本在 Mac Python 3.14 和 Notebook Python 3.11 均为 **24/24 Agent 测试通过**。仓库完整 `bash format.sh ci` 通过（含 Ruff、拼写、Markdown、工作流与 Gitleaks 等检查），仅目标 batch 文件有差异，其 SHA256 与 NPU 实测版本一致。wheel 0.2.0 已在 Notebook 离线安装，从项目目录之外导入并真实编译、启动 GDN NPU 内核成功，证据见 `runs-20261016/installed_package_smoke.json`。
