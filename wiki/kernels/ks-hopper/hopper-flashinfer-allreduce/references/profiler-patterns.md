# Profiler 模式匹配与优化映射

## 使用方法

本参考用于把新的模型推理 Profiler 映射到可复用的 FlashInfer AllReduce 优化规则。先识别 phase 和通信 shape，再匹配模式；kernel 名称只是证据之一，不是唯一触发器。

## 必需输入

尽量收集：

- GPU 架构、单机/跨节点拓扑、TP/EP 规模；
- prefill 与 decode 的时间线或可区分标记；
- AllReduce kernel 名、调用次数、累计 GPU 时间、p50/p95 单次时长；
- 每类调用的 shape、dtype、element size、payload bytes、contiguous；
- kernel 前后依赖、stream、同步点和与计算的 overlap；
- vLLM、FlashInfer、CUDA 版本及 AllReduce 配置。

payload 统一按 `numel × element_size` 计算；二维 `[tokens, hidden_size]` 可写为 `tokens × hidden_size × element_size`。不要只用 token 数判断阈值。

## 模式目录

### 模式 A：decode 高频小 AllReduce 暴露在关键路径

观察特征：

- decode step 中重复出现 AllReduce；
- 单次耗时较短，但调用数高、累计占比不可忽略；
- 多数目标 shape 为 contiguous 2-D FP16/BF16/FP32；
- payload 集中在小消息区间，且与计算串行或 overlap 很少；
- baseline 常见 `multimem_all_reduce_kernel`、vLLM custom AllReduce 或其他 fallback kernel。

匹配结论：`MATCH`。优先评估 FlashInfer MNNVL 小消息路径，并以 payload 阈值局部接管。

原理：小消息的固定启动、同步和调度成本占比高；降低每次固定延迟，能够在大量 decode step 中累积收益。它不是提升大消息带宽的方案。

### 模式 B：显式开关已开，但 eligible 张量没有进入 MNNVL

观察特征：

- 启动前已设置 FlashInfer 与 MNNVL 配置；
- payload、shape、dtype、device 均满足条件；
- trace 中 `trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` 为 0 或明显少于预期；
- 同一批张量仍由 symmetric-memory/custom AllReduce 消费。

匹配结论：`MATCH`。检查 standalone 是否读取 pass config，以及 FlashInfer eligibility 是否排在 symmetric-memory 之前。

原理：配置存在不等于路由生效；如果两个路径阈值不一致，或 fallback 更早返回，专用 kernel 永远不会执行。

### 模式 C：小张量进入 MNNVL，大张量仍走 multimem

观察特征：

- candidate trace 同时出现 MNNVL 与 multimem kernel；
- 按 payload 分组后，MNNVL 主要位于阈值内，multimem 主要位于阈值外；
- 请求正确性正常。

匹配结论：这是预期的分流结果，不是优化失效。下一步围绕真实 payload 分布扫描阈值，比较增量覆盖与性能回归。

原理：backend 的优势依赖消息大小；按阈值分流保留各路径的优势，并限制回归范围。

### 模式 D：小数 MiB 配置后初始化或位运算报类型错误

观察特征：

- 阈值配置为 `1.5` 等小数；
- workspace/token 初始化出现 float；
- 报错含 float 与整数位运算、对齐或 workspace size 不兼容。

匹配结论：`MATCH`。在 MiB→bytes 以及 bytes→max tokens 的边界显式整数化，并补小数阈值单测。

原理：阈值可以是小数，但底层 workspace、对齐和 token capacity 是离散整数；类型应在单位转换边界收敛。

### 模式 E：大 prefill AllReduce 或带宽主导

观察特征：

- 热点主要位于 prefill；
- 单次 payload 大、调用较少，耗时随 bytes 明显增长；
- 问题更像链路带宽、拓扑或大消息算法选择。

匹配结论：`NO_MATCH` 或仅 `PARTIAL_MATCH`。不要因为出现 AllReduce 就扩大 FlashInfer 小消息阈值。优先研究 overlap、分块、通信算法、拓扑、NCCL/symmetric-memory 或计算通信融合。

### 模式 F：通信已被隐藏或占比很低

观察特征：

- AllReduce 与计算高度 overlap；
- 关键路径上暴露时间很小；
- 单算子加速的理论上限不足以改变端到端指标。

匹配结论：`NO_MATCH`。即使单算子更快也不应优先投入；先优化关键路径上的其他热点。

## 判定模板

每次分析都输出：

```text
判定：MATCH | PARTIAL_MATCH | NO_MATCH
硬条件：Hopper / 节点内 / decode 小消息 / 暴露关键路径
命中信号：kernel、次数、累计时间、payload 分布、shape/dtype、overlap
反证：大消息、prefill 主导、跨节点、已隐藏、不支持输入
优化规则：MNNVL 小消息分流 / 配置传播 / 路由优先级 / 整数边界 / 不适用
原理：为什么该规则能作用于当前瓶颈
验证：单测 → 命中 shape 单算子 → 路由 trace → 交替 paired E2E
```

## 示例

若 H200 TP8 decode trace 中 `multimem_all_reduce_kernel` 出现数百次，目标调用多为 `[36,7168]`、`[64,7168]`、`[112,7168]` 的 BF16 contiguous 张量，通信与计算串行，且 MNNVL kernel 为 0，则判为 `MATCH`：这些 payload 分别约为 0.49、0.88、1.53 MiB，适合用约 2 MiB 上限试验 MNNVL 分流。优化后应看到阈值内调用转为 MNNVL、阈值外继续 fallback；再验证端到端收益，而不是由单算子结果直接推断。
