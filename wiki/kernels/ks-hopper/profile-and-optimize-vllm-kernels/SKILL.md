---
name: profile-and-optimize-vllm-kernels
description: 面向 NVIDIA Hopper（H100/H200）上的 vLLM 推理，建立无 profiler 的生产 E2E 基线，采集 eager 源码映射 trace 与 CUDA Graph production formal trace，统计 TP 多卡 leaf GPU kernel 耗时和占比，选择并修改 CUDA/Triton/CUTLASS/FlashMLA/FlashInfer kernel，再完成微基准、数值精度、镜像替换、交错 E2E 和因果审计。用于抓 trace、分析算子或 kernel 占用、定位 vLLM 热点、判断算子替换优先级、实施 Hopper kernel 优化、排查 profile 映射/重叠/多卡统计问题，或撰写可复现优化报告。
---

# Profile and Optimize vLLM Kernels

## 目标

把“真实负载发现热点 → 源码归因 → kernel 修改 → 正确性与 E2E 验收”做成可审计闭环。始终从真实 vLLM workload 出发，不用孤立 microbenchmark 热点替代生产热点。

## 读取资源

开始任务前完整读取 [references/workflow.md](references/workflow.md)。按任务继续读取：

- 计算占比、预测收益、判断正确性或验收结果时，读取 [references/metrics-and-gates.md](references/metrics-and-gates.md)。
- 优化 attention、sparse decode、split/combine、padding 或调度参数时，或需要完整案例时，读取 [references/flashmla-case-study.md](references/flashmla-case-study.md)。
- 输出正式报告时，复制并填写 [assets/optimization-report-template.md](assets/optimization-report-template.md)，不要直接改模板。
- 检查 trace 矩阵时，运行 `scripts/validate_trace_matrix.py`；它只读实际 Run，不修复或删除数据。

## 不可破坏的测量边界

始终分开三种服务，不混用计时：

| 服务 | CUDA Graph | Profiler | 作用 |
| --- | --- | --- | --- |
| production | 开 | 关 | 完整 E2E、TTFT、TPOT、吞吐 |
| mapping | 关（eager） | stack/shape 开 | raw kernel 到源码归因，不计性能 |
| formal | 开 | stack/shape/memory 关 | production graph 下的 GPU leaf 时间 |

强制遵守：

1. 固定模型 snapshot、镜像 digest、GPU、TP/DP、dtype、KV cache、编译配置、CUDA Graph、请求和采样参数。
2. 先证明 CUDA Graph 状态和实际运行路径，再接受 trace 或 E2E。
3. 按 TP rank 分析并取 rank 中位数；绝不把并行 rank 时间相加。
4. 对同一 family 的 GPU interval 求 union；绝不重复累计多 stream 重叠。
5. 区分 parent inclusive 与 leaf exclusive；按 replacement group 去重。
6. 把 `not-observed`、`fused-into-parent`、`CPU-only` 和真实 `0` 分开。
7. 把 GPU busy 占比与 wall-clock E2E 占比的分母分开。
8. 在改代码前保存 baseline、trace、源码/二进制身份和优化假设。
9. 未经用户授权只做分析时，在候选选择和设计处停止，不部署、不改远端服务。

## 执行流程

### 1. 冻结任务与环境

读取仓库约束和现有 runner。记录 hostname、GPU/driver/topology、容器与 image digest、vLLM/依赖版本、模型 config hash、源码 commit、编译参数和挂载证据。检查 GPU 排他性和结果盘容量。

定义场景矩阵，至少覆盖长 decode、长 prefill、均衡短上下文和均衡长上下文。把并发、采样、prefix cache、speculative decoding 和 acceptance 纳入 workload 身份。

### 2. 建立无 profiler 的生产基线

启动 production 服务，保存 vLLM 最终解析配置和 CUDA Graph capture/replay 日志。显式设置采样参数，不依赖版本默认值。

分别运行：

- **生产代表性协议**：使用真实 prompt/采样/并发；若使用 random dataset，显式设置 temperature、top-p 等。
- **因果控制协议**：使用固定 token-ID prompt、固定 seed、精确 ISL/OSL 和 prompt hash。

不要混合两套协议的绝对 token/s。对 MoE 检查长 greedy 输出是否退化重复；对 speculative decoding 保存 acceptance、draft/accepted token 数和输出 hash。

### 3. 采集 mapping trace

以 eager 启动 profiler，打开 stack 和 shape。抓短 prefill/decode 窗口，只建立 raw symbol → framework op → vLLM source → extension source 的证据链。若新 family 无法映射，补抓最小对应场景；禁止把 mapping duration 放入性能表。

### 4. 采集 formal trace

重启为 CUDA Graph production 配置，关闭 stack、shape 和 memory profiling。每个场景抓 `prefill`、`decode-early`、`decode-mid`、`decode-late`；长 decode 使用真实 KV 长度的短窗口，不全程 profile。

验证每个 stage 的所有 TP rank、gzip、完成标记、probe 元数据和 graph replay。失败 stage 移入 attempts 留证，只重跑失败项。

### 5. 统计 leaf kernel

全量枚举 CUDA/CUTLASS/Triton/Inductor/NCCL/ATen leaf，不从预设 Python 算子列表倒推热点。执行：

1. canonicalize raw symbol，同时保留完整原名；
2. family 内 interval union；
3. decode 按 trace 中实际 replay 数归一化；
4. 每 stage 取 TP rank 中位数并报告 rank skew；
5. 用 early/mid/late 积分估计完整 decode；
6. 计算逐场景 GPU busy share、绝对时间、调用次数和 coverage；
7. 生成四场景 macro、suite saved、worst scene 和 replacement group 表。

### 6. 选择优化目标

优先选择同时满足以下条件的 leaf/replacement group：

- production formal trace 占比较高或绝对时间高；
- 多场景覆盖稳定，或明确针对目标场景特化；
- 源码边界可修改，runtime path 已证明；
- 不是只有 Python wrapper inclusive 时间；
- 不是无法在当前任务中替换的 NCCL/驱动库；
- 未已经被 compile/CUDA Graph 融合到另一 kernel；
- micro speedup 乘真实占比后仍可能产生可测 E2E 收益。

为每个候选写一条可证伪假设：生产 shape、瓶颈机制、拟修改变量、适用边界、回退条件、预期 kernel saving 和预期 E2E 上界。

### 7. 修改与隔离验证

一次只实现一个核心机制。对 shape 特化使用 fail-closed dispatch，不满足 dtype、架构、layout、head、batch 或 alignment 时回退 baseline。增加低频 runtime-hit 标记，确保标记发生在 warmup/graph capture，不污染 timed region。

建立同 binary microbenchmark，复用完全相同输入，覆盖真实 shape、边界 shape 和回退 shape。交错 benchmark 顺序并报告最差轮，不只报告最佳点。检查寄存器、shared memory、spill、occupancy 和生成 symbol。

### 8. 通过正确性门禁

先比较 candidate 与 production baseline，再比较两者与高精度 oracle。按模型真正消费的张量定义验收边界，但同时披露被丢弃输出的误差。归约顺序改变时不强求 bitwise equal；要求 candidate 的模型可见输出误差不劣于 production baseline。

### 9. 构建并证明替换生效

从冻结源码构建可复现 candidate image，只替换目标 extension/kernel。保存 patch、源码 commit、构建日志、image digest、extension SHA256、符号和 runtime-hit。baseline/candidate 的非目标配置必须一致。

### 10. 执行交错 E2E

每个 arm 使用独立服务和相同 warmup。推荐 `B1-C1-C2-B2-B3-C3`，按 `B1/C1`、`B2/C2`、`B3/C3` 配对。持续检查 GPU 排他性、运行期 JIT、token 数、prompt hash、采样参数、prefix cache 和 speculative acceptance。

主指标预先声明；单请求优化默认使用完整 E2E duration，TPOT/ITL/TTFT 和吞吐用于解释。不要在结果出来后挑选最有利指标。

### 11. 做因果审计并迭代

同时检查 artifact、single-variable、runtime-hit、correctness、performance 和 Amdahl causal bound。若 candidate image 与 baseline 的重编译方式也不同，补做同 candidate binary 内 feature-off/feature-on；否则只声明“复合 candidate 收益”，不要归因给单个 patch。

若 E2E 不符合预期，依次核查运行路径、shape 覆盖、kernel saving、热点占比、同步/CPU 空洞、sampling/acceptance、并发饱和和二进制差异，再决定迭代或放弃。

## 完成条件

只有下列证据同时存在时才宣称优化完成：

- 可复现环境 manifest 和无 profiler production baseline；
- mapping 与 formal trace 的配置、完整性和 CUDA Graph 证据；
- 逐场景 leaf 时间、占比、rank skew、coverage 和去重后的融合排序；
- raw symbol 到可修改源码的映射；
- patch、构建身份、runtime-hit 与回退边界；
- production shape microbenchmark 和数值正确性结果；
- 至少三组交错 E2E 配对及所有原始指标；
- single-variable 与 causal-bound 审计；
- 明确区分已证明结论、推断和剩余工作。
