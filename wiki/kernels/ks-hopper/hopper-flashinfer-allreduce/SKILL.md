---
name: hopper-flashinfer-allreduce
description: 从 NVIDIA Hopper 模型推理 Profiler 中识别高频、小 payload、暴露在 decode 关键路径上的 AllReduce 模式，并为 vLLM 配置、实现或评审 FlashInfer MNNVL AllReduce 显式优化，包括阈值、路由、安全回退和 A/B 验证。不要用于非 Hopper 平台、大 payload 主导或已经充分隐藏的通信。
---

# Hopper FlashInfer AllReduce

## 目标

沉淀一个显式 opt-in 的 Hopper 通信优化方法：使用户配置的 FlashInfer AllReduce 阈值同时约束 fusion 与 standalone 路径，修复小数 MiB 导致的 float token/workspace 问题，并让符合条件的小 AllReduce 在 symmetric-memory 之前进入 FlashInfer MNNVL。

处理推理 Profiler 时，先读 [references/profiler-patterns.md](references/profiler-patterns.md) 做模式匹配；命中后再读 [references/optimization-evidence.md](references/optimization-evidence.md)，核对实现不变量、已知效果和不能外推的结论。

## 模式匹配入口

不要因为 trace 中出现 `AllReduce` 就直接应用本优化。先检查四个硬条件：

1. 目标为 NVIDIA Hopper CUDA，且运行时具备 FlashInfer MNNVL 能力；
2. 热点来自模型推理的 tensor/expert parallel AllReduce，而不是跨节点网络瓶颈；
3. decode 阶段存在二维、contiguous、FP16/BF16/FP32 的小 payload AllReduce；
4. 这些通信未被充分隐藏，对关键路径有可观察贡献。

四个硬条件满足后，以下信号中至少命中两个，才把本 Skill 作为候选优化：

- `multimem_all_reduce_kernel` 或当前 fallback kernel 调用次数高，在多个 decode step 重复出现；
- 单次 kernel 很短但累计时间显著，表现为“高频小通信”而非少量大通信；
- payload 直方图有一批张量落在可配置的 FlashInfer 上限内；
- 通信 kernel 与前后计算基本串行，时间线存在可回收的暴露间隙；
- 显式开启 FlashInfer 后，eligible shape 仍未出现 MNNVL kernel，说明配置传播或路由优先级可能失效。

若硬条件不满足，输出“不匹配”及原因，转向 overlap、大消息通信、NCCL/拓扑、布局转换或其他优化，不要强行套用本规则。

## 触发条件

在以下任务中使用本 Skill：

- 优化或诊断 Hopper 上 vLLM TP/EP 模型的 decode AllReduce 热点；
- 分析 Nsight Systems、Nsight Compute、PyTorch Profiler 或 Chrome trace 中高频小 AllReduce 模式；
- 实现、迁移或评审 FlashInfer MNNVL AllReduce 路由；
- 配置 `VLLM_ALLREDUCE_USE_FLASHINFER`、`VLLM_FLASHINFER_ALLREDUCE_BACKEND` 或 `fi_allreduce_fusion_max_size_mb`；
- 复现 FlashInfer AllReduce 的单算子、Profiler 或端到端 A/B 效果；
- 排查显式 FlashInfer 配置未生效、阈值不一致、float workspace/token 异常或 fallback 失效。

不要用于 ROCm、非 Hopper GPU，或与 vLLM/FlashInfer AllReduce 无关的通用 Kernel 优化。

## 前置条件

- 目标硬件为 NVIDIA Hopper；原验证环境为单机 8×H200、TP8。
- vLLM 构建包含 FlashInfer AllReduce，并支持 `mnnvl` backend。
- 能取得目标 vLLM 源码、启动配置、模型 workload 和稳定的空闲 GPU。
- 记录 GPU、驱动、CUDA、vLLM/FlashInfer 版本、commit、编译参数及拓扑。
- A/B 两侧必须使用同一代码基线、模型、请求集、并发、compilation config 和计时方法。

## 工作流程

1. 建立 baseline。固定真实 decode shape、dtype、hidden size、并发和请求长度；保留服务日志、请求完整性和性能原始数据。
2. 提取 Profiler 特征。按 phase 区分 prefill/decode，记录 AllReduce 的 kernel 名、调用次数、累计时间、单次时长、payload、dtype、shape、contiguous、前后依赖和 overlap。
3. 执行模式匹配。用“硬条件 + 至少两个信号”判定是否进入本优化；按 [references/profiler-patterns.md](references/profiler-patterns.md) 输出命中模式、反证和置信度。
4. 形成假设。若命中，优先假设“高频小 AllReduce 的固定开销/路由成本过高”，再判断是阈值覆盖不足、显式配置未传播，还是 eligible 张量被更早的 fallback 路由消费。
5. 检查实现不变量：
   - MiB 转 bytes 后显式保持整数；由 workspace、hidden size、element size 算出的 token 上限也必须是整数。
   - standalone 从当前 `compilation_config.pass_config` 读取显式 `fi_allreduce_fusion_max_size_mb`，与 fusion 使用同一边界。
   - FlashInfer eligibility 仅接受 CUDA、contiguous、二维、FP16/BF16/FP32 且 payload 不超过阈值的张量。
   - 显式启用且 eligible 的 FlashInfer 必须排在 symmetric-memory 之前；不支持或超阈值输入必须安全回退。
6. 显式启用候选路径。启动服务前设置：

   ```bash
   export VLLM_ALLREDUCE_USE_FLASHINFER=1
   export VLLM_FLASHINFER_ALLREDUCE_BACKEND=mnnvl
   ```

   并在 compilation config 中设置：

   ```json
   {"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","pass_config":{"fi_allreduce_fusion_max_size_mb":2}}
   ```

7. 做公平 A/B。baseline 只把 `VLLM_ALLREDUCE_USE_FLASHINFER` 设为 `0`；candidate 设为 `1`。两侧都保留相同 backend 与 payload pass config，控制唯一变量。
8. 分层验证：先跑静态/单元测试，再测 Profiler 中命中的真实 shape 单算子，再用 trace 证明路由，最后做端到端 ABBA 或交替顺序 paired A/B。
9. 根据 paired 中位数、正收益比例、CV 与 Bootstrap 95% CI 决策。保留回退路径；只有目标机器和固定 workload 证据稳定时才部署。

## 优化思路与原理

- **小消息选择专用路径**：decode 中 AllReduce payload 小、调用密集，固定启动、同步和调度开销占比高。FlashInfer MNNVL 针对节点内 NVLink/NVSwitch 小消息路径，可降低单次通信延迟；收益来自重复调用的累积。
- **按 payload 分流**：用阈值只接管更适合 MNNVL 的小张量；大张量保留 symmetric-memory、custom AllReduce 或 PyNCCL，避免一个 backend 覆盖所有 shape。
- **让配置贯穿 fusion 与 standalone**：同一阈值必须控制两条路径，否则 Profiler 中看似匹配的张量可能因边界不一致走错 backend。
- **显式路由优先**：用户 opt-in 后，eligible 张量必须在 symmetric-memory 之前判断，否则开关存在但 trace 不会出现 MNNVL kernel。
- **不支持输入安全回退**：优化是局部分流，不改变非 CUDA、非 contiguous、非二维、不支持 dtype 或超阈值输入的语义和稳定性。

## 优化原则

- `2 MiB` 是 standalone/fusion FlashInfer 的最大 payload，不表示所有 AllReduce 都切换到 FlashInfer。
- BF16、hidden size 7168 时，146 tokens 为 2,093,056 bytes，可进入 2 MiB 阈值；147 tokens 为 2,107,392 bytes，应回退。
- `mode: 0` 只是显式 compilation/cudagraph/pass 配置，不是 FlashInfer 开关。
- 路由顺序保持：ROCm QuickReduce → 显式且 eligible 的 FlashInfer → symmetric-memory → AITER/vLLM custom AllReduce → PyNCCL。
- 优先覆盖 decode 小张量；大 prefill 张量仍可能由 symmetric-memory 处理。
- 单算子延迟改善不能直接换算为模型吞吐提升。

## 验证要求

### 正确性

- 覆盖 1.5 MiB 小数阈值，确认 workspace/token 参数没有 float 泄漏。
- 覆盖阈值内、边界与阈值外 payload，以及 shape、dtype、device、contiguous 条件。
- 验证 FlashInfer 优先级与所有 fallback；任何不支持输入不得失败或错误路由。
- 端到端检查请求零失败、输入 token 和输出 token 完整，不以仅有 HTTP 成功代替正确性。

### 性能

- 单算子使用真实 decode shape，分别独立 warmup/measure，多进程或多轮交错，报告中位数与离散度。
- trace 中 candidate 应出现 MNNVL kernel；baseline 不应出现。仍有 multimem kernel 是可能且合理的，因为阈值只覆盖部分张量。
- 端到端采用预先固定的 workload 和 ABBA/交替顺序，避免固定 baseline→candidate 顺序偏差。
- 异常慢轮只可做有说明的敏感性分析，不得静默剔除或用其放大收益。

## 常见问题

- 只设置环境变量但 MNNVL kernel 为 0：检查路由优先级、eligibility、初始化时机和 pass config 是否真正传到 standalone。
- 1.5 MiB 配置触发位运算类型错误：检查 bytes 与 max token 的每一步是否显式为整数。
- candidate 仍出现大量 multimem kernel：先判断这些张量是否超阈值；不要误判为优化完全失效。
- 端到端收益波动或反向：检查 GPU 是否被占用、服务启动顺序、异常停顿、请求完整性和顺序偏差；重新做干净的交替 paired 测试。
- 想把 2 MiB 设为默认值：现有证据只支持显式 opt-in，不支持全局默认；应在目标 workload 上重新选取阈值。

## 输出要求

报告必须区分“功能生效”“单算子收益”和“端到端收益”。明确目标机器、workload、样本数、聚合口径、异常轮次、fallback 覆盖与结论边界。最终结论使用 `ACCEPT_EXPLICIT_OPT_IN`、`REJECT` 或“证据不足”，不得宣称固定 4% 或跨 workload 普适收益。

报告开头先给出模式匹配结果：`MATCH`、`PARTIAL_MATCH` 或 `NO_MATCH`；列出命中的 Profiler 信号、反证、拟采用的优化规则和预期作用机制。没有 trace/payload 证据时最多判为 `PARTIAL_MATCH`。
