# 指标、正确性与验收门禁

## 目录

1. [E2E 指标](#1-e2e-指标)
2. [GPU kernel 时间](#2-gpu-kernel-时间)
3. [多卡与重叠](#3-多卡与重叠)
4. [收益预测](#4-收益预测)
5. [正确性门禁](#5-正确性门禁)
6. [性能门禁](#6-性能门禁)
7. [审计结果](#7-审计结果)

## 1. E2E 指标

定义请求开始为发送 streaming completion 前的单调时钟，结束为收到 `[DONE]`：

```text
E2E duration = done_time - request_start
TTFT         = first_choice_time - request_start
ITL_i        = choice_time_i - choice_time_(i-1)
TPOT         = mean(all ITL)
output tok/s = actual_output_tokens / duration
total tok/s  = (actual_input_tokens + actual_output_tokens) / duration
```

固定 OSL 的单请求下，output throughput 基本是 duration 的倒数变换，不是独立证据。vLLM 服务日志的 `Avg generation throughput` 是日志窗口内生成 token 数/时间；长 B1 稳态时可接近 bench output throughput，但短窗口峰值不能替代最终 JSON。

明确主指标。延迟优化通常用 duration/TTFT/TPOT；容量测试用 request/output/total throughput。不要混用“越低越好”和“越高越好”的收益公式：

```text
latency_gain_pct    = (B - C) / B * 100
throughput_gain_pct = (C - B) / B * 100
```

## 2. GPU kernel 时间

每个 CUDA event 使用半开区间 `[start,end)`。同 family 的 exclusive GPU 时间为其区间 union，不是 duration 简单求和。

对 decode trace：

```text
rank_family_us_per_replay = union_us / actual_graph_replays
stage_family_us           = median(rank_family_us_per_replay)
```

每个场景：

```text
decode_step_us  = (early + 4*mid + late) / 6
decode_total_us = decode_step_us * OSL
total_us        = prefill_us + decode_total_us
gpu_share       = total_us / total_gpu_activity_union_us
```

保存 min/max/median/CV 或 P90 rank skew。中位数用于 TP 汇总，skew 用于发现负载不均和慢 rank。

## 3. 多卡与重叠

TP8 的八张卡并行执行，不能将八个 rank 的时间相加。若一个请求 wall time 为 10 ms、每个 rank kernel 均约 4 ms，请求不是 32 ms。

parent/child 示例：

```text
MLA wrapper
└── sparse indexer
    └── FlashMLA split/combine leaves
```

parent inclusive 和 child exclusive 可分别展示，但进入总收益排序时只选择一个 replacement group 接入点。不同 stream 上的两个 family 可能重叠；合计占比前先计算跨 family union。

## 4. 收益预测

微基准 speedup 为 `S`、目标 kernel 在请求中的时间为 `T`：

```text
predicted_saved_us = T * (1 - 1/S)
predicted_e2e_pct  = predicted_saved_us / measured_e2e_us * 100
```

若只知道 GPU busy share `p`，`p*(1-1/S)` 是相对于 GPU activity 的粗略 saving，不等于 wall E2E。预测表同时列出：

```text
逐场景 share
micro speedup 和最差轮
predicted saved absolute time
predicted E2E upper bound
coverage
overlap caveat
```

实测 E2E 明显超过上界时，优先怀疑复合二进制、测量协议或归因，不把它包装成超线性收益。

## 5. 正确性门禁

### 5.1 三方比较

至少区分：

```text
candidate vs production baseline
production baseline vs high-precision oracle
candidate vs high-precision oracle
```

对于 reduction/softmax，candidate 可能因归约顺序不同而非 bitwise equal。提前定义容差和模型可见量。

### 5.2 不劣于 baseline

以 FP64 oracle 为例：

```text
baseline_mae  = mean(abs(baseline_fp64 - oracle_fp64))
candidate_mae = mean(abs(candidate_fp64 - oracle_fp64))
pass          = candidate_mae <= baseline_mae + declared_tolerance
```

同时报告 max abs、相对误差、NaN/Inf 和 exact equal rate。MoE hidden 使用均值误差时，保持输入、权重、router 和 accumulation 口径一致。

### 5.3 模型可见边界

若 API 返回 `(output, LSE)`，但 vLLM 只消费 output：

- 以 output 作为生产验收主边界；
- 仍报告 LSE 误差和是否通过全 tuple 门禁；
- 证明 LSE 在上层确实被丢弃；
- 不把“返回值未使用”误写成“kernel 中不计算 LSE”。

### 5.4 自回归输出

生成文本 hash 不同不能单独证明 kernel 错误，也不能证明正确。微小数值差异会改变后续 token。使用相同 prompt hash、token 数和独立 tensor/oracle test 判定正确性。

## 6. 性能门禁

在实验前声明阈值，例如：

```text
三组 paired duration gain 全部 > 0
paired gain 中位数 > 1%
最差场景不低于预设回退阈值
microbenchmark 三轮最差 speedup > 1
```

门禁还要求：

- B/C CUDA Graph、compile、TP、dtype 和请求完全一致；
- timed region 无外部 GPU process；
- timed request 无新增 JIT；
- warmup 不计入正式结果；
- image/extension hash 每个 arm 稳定；
- runtime-hit 在所有 candidate TP workers 出现，baseline 不出现；
- prompt hash、实际 ISL/OSL 一致；
- speculative workload 的 acceptance 可比较；
- 生产代表性与因果控制结果分开报告。

`B1-C1-C2-B2-B3-C3` 可以抵抗单调漂移，但不能消除所有噪声。保存各 pair 原始值和顺序，不只保存中位数。

## 7. 审计结果

最终至少生成以下布尔审计及 failure 列表：

| 审计 | 证明内容 |
| --- | --- |
| artifact | 镜像、源码、extension、日志和结果完整 |
| trace | stage/rank/replay/gzip/CUDA Graph 完整 |
| mapping | 热点 raw symbol 到源码覆盖充分 |
| single-variable | 非目标变量无差异 |
| runtime-hit | candidate 实际执行补丁路径 |
| correctness | 模型可见输出满足预设门禁 |
| performance | 配对数量、方向和阈值通过 |
| causal-bound | 实测收益没有违反可解释上界 |

任何 audit failure 非空时，结论写为 `inconclusive` 或 `failed`，不能仅凭正收益接受 candidate。
