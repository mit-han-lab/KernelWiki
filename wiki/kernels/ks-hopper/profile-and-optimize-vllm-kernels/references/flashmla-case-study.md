# FlashMLA：从 production trace 到 Hopper kernel 优化

## 目录

1. [发现热点](#1-发现热点)
2. [确认源码边界](#2-确认源码边界)
3. [C04 调度优化](#3-c04-调度优化)
4. [active-head 优化](#4-active-head-优化)
5. [正确性](#5-正确性)
6. [微基准和 E2E](#6-微基准和-e2e)
7. [可迁移经验](#7-可迁移经验)

## 1. 发现热点

在 GLM-5.2-FP8、TP8、8×H200 的四个真实场景中，先完成无 profiler E2E，再用 CUDA Graph formal trace 全量枚举 leaf。两个 FlashMLA sparse-decode family 排名靠前：

| family | 四场景 macro GPU busy share |
| --- | ---: |
| `mla_sparse_attention_splitkv` | 12.504% |
| `mla_attention_combine` | 6.940% |
| 算术合计 | 19.444% |

逐场景合计约为：1k32k 15.320%、32k1k 30.355%、4k4k 16.062%、8k8k 16.039%。这些是 formal per-rank GPU activity share，不是 E2E wall time share；两 family 的算术和用于热点筛选，严格上界仍需检查重叠。

这个步骤比“看到 MLA wrapper 很慢”更可靠，因为它定位到真实 CUDA leaves，而不是 Python inclusive wrapper。

## 2. 确认源码边界

eager mapping trace 将 raw symbol 映射为：

```text
flash_fwd_splitkv_mla_fp8_sparse_kernel
  -> mla_sparse_attention_splitkv

flash_fwd_mla_combine_kernel
  -> mla_attention_combine
```

调用链：

```text
vLLM FlashMLASparseImpl
  -> _forward_fp8_kv_mixed_batch
  -> _fp8_flash_mla_kernel
  -> flash_mla_with_kvcache
  -> _flashmla_C::sparse_decode_fwd
```

FlashMLA 把 sparse indexer 选出的 top-k KV 索引划分给多个 split。每个 split 计算局部 score、softmax max/LSE、exp sum 和 partial output；combine 使用局部 LSE 做稳定归一化并合并最终 output。

## 3. C04 调度优化

### 3.1 从 production shape 推导问题

旧 Hopper scheduler 的核心关系为：

```text
num_sm_parts = max(num_sms / s_q / (h_q/64), 1)
```

H200 production shape：

```text
num_sms=132, s_q=1, padded h_q=64
num_sm_parts=132
topk=2048, page_size=64
有效 sparse KV blocks=32
```

132 partitions 对 32 个有效 sparse blocks 可能产生空 CTA、过细切分、partial workspace 和 combine 开销。优化假设是将 parts 限制到有效工作规模附近。

### 3.2 实现和运行证据

补丁在 FlashMLA scheduler metadata 中加入可控 cap，正式候选选择 32，并添加一次性 runtime-hit：

```text
h_q=64 s_q=1 uncapped=132 requested_cap=32 selected=32
```

这不是打开 FlashMLA 已有配置，而是修改并重新编译 `_flashmla_C.abi3.so`。不满足条件时保留原调度。

### 3.3 sweep 而不是猜常数

同 binary、CUDA Graph boundary sweep：

```text
cap132: 17.6647 us
cap32:  16.6244 us
speedup=1.0626x，saving=5.89%
```

cap16/8 反而退化，证明“parts 越少越快”是错误规则。32 只适用于该 top-k/page/head/batch/GPU shape。

## 4. active-head 优化

继续分析发现 WGMMA 主体按 64 padded heads 运行，但 TP8 下模型只消费部分 heads：

| 模型 | padded heads | active heads |
| --- | ---: | ---: |
| GLM-5.2 | 64 | 8 |
| DS-V4 | 64 | 16 |

上游矩阵主体仍需保留 64-row tile，但 split epilogue 和 combine 不必缩放、写入、读取或归约 padded rows。后续优化包括：

```text
active-head combine
small-split fast combine
active-head split epilogue
```

安全边界：

- `active_h_q <= h_q` 且满足对齐，否则回退；
- 只对已验证的 H200/64-head 模板特化；
- 不改变 QK/PV、softmax 和 active heads 的归约顺序；
- active output 必须通过精度门禁；
- 保持无 local memory spill，并记录寄存器/shared memory。

低 query-row 并发的同 binary CUDA Graph microbenchmark 中，GLM 和 DS-V4 的 split/combine boundary 获得约 1.16x～1.35x 的收益；接近饱和时收益收敛到约 1。因此必须测 `batch * seq_q` 边界，不能把 B1 收益外推到所有 continuous batching。

## 5. 正确性

C04 改变 partition 时会改变浮点归约顺序。验收采用 production BF16 baseline 与 FP64 oracle，而非强制逐 bit 相同：

```text
candidate active output MAE <= production active output MAE
```

active-head 版本还区分：

- production `.so` 与 candidate `.so` 的跨 binary 差异；
- candidate binary 内 baseline policy 与 fast policy 的纯策略差异；
- 模型可见 active output 与上层丢弃的 LSE。

在 GLM/DS-V4 的对应 vLLM 路径中，LSE 仍在 FlashMLA 内参与 softmax/combine，但返回值未进入后续 Transformer。因而报告同时给出 output 门禁和全 tuple 结果，不能概括为“LSE 不计算”或“所有结果 exact”。

## 6. 微基准和 E2E

### 6.1 微基准边界

固定相同 q、packed FP8 KV、indices 和 metadata，先 eager，再 capture CUDA Graph。交错 policy 顺序，覆盖 cache length、batch、`seq_q` 和回退边界。记录 output/LSE hash、MAE、kernel symbol、资源和最差轮 speedup。

`query_rows_per_graph = batch * seq_q`，不总等于客户端并发。`seq_q=4` 可来自 speculative verification；相同 rows 的 `(batch,seq_q)` 仍可能具有不同 shape 和性能。

### 6.2 正式 E2E

C04 采用四场景 `B1-C1-C2-B2-B3-C3`。旧复合 candidate 的四场景 E2E duration 中位收益为 1.798%～3.537%，12 个 pair 全正且超过 1%。TPOT 四场景方向一致。

但正式 baseline 使用 production binary，candidate 使用重编译 binary+cap32；因此全部 E2E 只能声明为“可部署复合 candidate 收益”。纯 cap 的最严格归因还需要同 candidate binary 的 cap132/cap32 E2E。

这条限制是案例最重要的审计经验：正收益、runtime-hit 和合理 Amdahl 上界仍不足以消除二进制构建差异。

## 7. 可迁移经验

1. 从 production leaf trace 选目标，不从函数名或单算子榜单猜热点。
2. 用 mapping trace 证明源码边界，用 formal trace 计时；两者不可互换。
3. 把真实 shape 代入 scheduler/layout 公式，寻找“硬件规模”和“有效工作”不匹配。
4. 先 sweep 找拐点，再编码 guarded policy；不要把 H200 的 32 parts 当通用常数。
5. 优先删除 padded/无消费者工作，但证明模型可见边界。
6. 同 binary 隔离 patch，跨 binary 验证部署物。
7. 用 runtime-hit 证明生产调用实际命中，但 marker 本身不是性能证据。
8. 把 micro speedup 乘真实占比估计可测收益，再用交错 E2E 验证。
9. 报告并发边界、回退路径、失败点和未完成的严格归因。
10. 对 MoE/speculative workload 固定或记录 sampling、专家路由相关输入和 acceptance，避免把工作量变化当 kernel 收益。
