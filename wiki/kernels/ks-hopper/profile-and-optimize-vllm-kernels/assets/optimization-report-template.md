# <优化名称>：从真实 vLLM Profile 到 Hopper Kernel E2E 验收

## 1. 结论摘要

- 目标 kernel/replacement group：
- production hotspot：
- 核心源码修改：
- 正确性结论：
- microbenchmark：
- E2E 主指标：
- 适用边界：
- 尚未证明：

## 2. 冻结环境

| 字段 | Baseline | Candidate |
| --- | --- | --- |
| Host/GPU |  |  |
| Driver/CUDA |  |  |
| Model snapshot |  |  |
| Image digest |  |  |
| vLLM/source commit |  |  |
| Extension SHA256 |  |  |
| TP/DP/PP |  |  |
| dtype/KV dtype |  |  |
| Compile/CUDA Graph |  |  |
| Prefix/speculative |  |  |
| Sampling |  |  |

## 3. Workload 与 E2E baseline

| 场景 | ISL/OSL | 并发 | 请求数 | Temperature | E2E | TTFT | TPOT | Output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|  |  |  |  |  |  |  |  |  |

说明 production 代表性协议与因果控制协议，不混合绝对值。

## 4. Trace 采集和完整性

| 模式 | CUDA Graph | Profiler | Stage/Rank | 完整性 | 用途 |
| --- | --- | --- | --- | --- | --- |
| production |  |  |  |  | E2E |
| mapping |  |  |  |  | source attribution |
| formal |  |  |  |  | GPU timing |

记录服务日志证据、trace 数、gzip、实际 replay 和失败 attempts。

## 5. Leaf kernel 时间与占比

| 排名 | Raw symbol/family | 逐场景绝对时间 | GPU busy share | Rank skew | Coverage | Replacement group |
| ---: | --- | --- | ---: | ---: | ---: | --- |
|  |  |  |  |  |  |  |

说明 interval union、TP rank 中位数、decode 积分和 parent/child 去重。

## 6. 源码映射

```text
raw CUDA symbol
  -> framework/custom op
  -> vLLM file:line
  -> extension binding
  -> CUDA/Triton/C++ source
```

## 7. 优化假设与 Amdahl 预算

- 生产 shape：
- 机制瓶颈：
- 可证伪假设：
- Micro speedup 目标：
- Predicted saved time：
- Predicted E2E upper bound：
- 回退条件：

## 8. Patch 与构建

- Patch/commit：
- 修改文件：
- Dispatch 条件：
- Runtime-hit：
- Build command：
- Image/extension identity：
- Kernel resource：

## 9. 正确性

| 模型可见量 | Production vs oracle | Candidate vs oracle | Candidate 变化 | 门禁 |
| --- | ---: | ---: | ---: | --- |
|  |  |  |  |  |

另列未消费输出、跨 binary 差异、NaN/Inf、max abs 和 exact rate。

## 10. Microbenchmark

| Shape | B R1/R2/R3 | C R1/R2/R3 | 最差 speedup | 正确性 | Runtime path |
| --- | --- | --- | ---: | --- | --- |
|  |  |  |  |  |  |

## 11. 正式交错 E2E

顺序：`B1-C1-C2-B2-B3-C3`。

| 场景 | Pair 1 | Pair 2 | Pair 3 | E2E gain median | TTFT | TPOT | ITL | Throughput | Acceptance |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
|  |  |  |  |  |  |  |  |  |  |

## 12. 审计

| Audit | 结果 | Failure/证据 |
| --- | --- | --- |
| Artifact |  |  |
| Trace |  |  |
| Mapping |  |  |
| Single variable |  |  |
| Runtime hit |  |  |
| Correctness |  |  |
| Performance |  |  |
| Causal bound |  |  |

## 13. 结论边界

### 已证明

- <填写已证明结论>

### 推断

- <填写仍需实验支持的推断>

### 未完成/下一步

- <填写剩余实验或下一步>

## 14. 原始证据索引

- Manifest：
- Server logs：
- Raw traces：
- Analysis CSV/JSON：
- Patch/build：
- Correctness：
- Microbenchmark：
- E2E/audits：
