# Hopper vLLM trace 到 kernel 优化工作流

## 目录

1. [输入和产物契约](#1-输入和产物契约)
2. [冻结环境](#2-冻结环境)
3. [生产 E2E](#3-生产-e2e)
4. [mapping trace](#4-mapping-trace)
5. [formal trace](#5-formal-trace)
6. [trace 完整性](#6-trace-完整性)
7. [leaf 时间统计](#7-leaf-时间统计)
8. [热点到源码](#8-热点到源码)
9. [优化与构建](#9-优化与构建)
10. [正确性和微基准](#10-正确性和微基准)
11. [正式 E2E](#11-正式-e2e)
12. [失败诊断](#12-失败诊断)

## 1. 输入和产物契约

开始前取得以下输入；缺失时先用只读检查补齐：

```text
目标 Hopper GPU 和可用主机
baseline image digest 与 candidate 构建入口
模型 snapshot、dtype、KV layout
vLLM 源码/版本和目标 extension 源码
生产启动参数
生产请求分布：ISL/OSL、batch、并发、sampling、prefix、spec decode
允许的正确性误差
```

为每次实验创建不可复用的 `RUN_ROOT`：

```text
RUN_ROOT/
├── manifest/
├── server/{production,mapping,formal}.log
├── e2e/
├── profiles/{mapping,formal}/captures/<scenario>/<stage>/
├── analysis/{leaf-kernel-timing,operator-timing}/
├── candidate/{patches,build,correctness,microbench}/
└── audits/
```

不要覆盖旧 Run。失败 capture 放入带时间戳的 `attempts/`。

## 2. 冻结环境

记录并保存输出：

```bash
hostname -f
nvidia-smi -L
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
  --format=csv
nvidia-smi topo -m
docker image inspect "$IMAGE"
docker inspect "$CONTAINER"
sha256sum "$MODEL_PATH/config.json"
python3 -m pip show vllm torch triton flashinfer-python
```

正式计时前检查所有 GPU 无外部 compute process，并在整个 timed region 持续轮询。不要只在开始前检查一次。

保存 vLLM 最终解析配置，而不只保存 shell 参数。确认：

```text
enforce_eager=False
cudagraph mode/capture sizes
TP/DP/PP
dtype/quantization/KV dtype
prefix caching
speculative config
compiler pass
kernel backend
```

## 3. 生产 E2E

定义代表性场景，例如：

| 场景 | ISL | OSL | 目的 |
| --- | ---: | ---: | --- |
| long-decode | 1024 | 32768 | decode 主导 |
| long-prefill | 32768 | 1024 | prefill 主导 |
| balanced-short | 4096 | 4096 | 常规均衡 |
| balanced-long | 8192 | 8192 | 长上下文均衡 |

先运行无 profiler E2E。不同 vLLM 版本的 random dataset 参数名称可能是 `--input-len/--output-len` 或 `--random-input-len/--random-output-len`，先执行：

```bash
vllm bench serve --help | rg 'random-input-len|input-len|temperature|num-warmups'
```

生产代表性模板：

```bash
vllm bench serve \
  --backend openai \
  --base-url "$BASE_URL" \
  --endpoint /v1/completions \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL" \
  --tokenizer "$MODEL_PATH" \
  --dataset-name random \
  --random-input-len "$ISL" \
  --random-output-len "$OSL" \
  --random-range-ratio 0 \
  --ignore-eos \
  --temperature "$TEMPERATURE" \
  --seed "$SEED" \
  --max-concurrency "$CONCURRENCY" \
  --request-rate inf \
  --num-warmups "$WARMUPS" \
  --num-prompts "$PROMPTS" \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,99 \
  --save-result --save-detailed \
  --result-dir "$RESULT_DIR"
```

始终显式传 `--temperature`。没有统一正确值：使用真实生产参数测生产能力；使用固定 token-ID 和确定性设置做严格控制。若生产没有 greedy，不要把 `temperature=0` 的退化长输出当成生产吞吐。

每个场景至少三轮。保存逐请求原始结果，不只保存平均值。验证成功数、实际输入/输出 token 数、错误列表和 warmup 是否排除在正式统计之外。

## 4. mapping trace

独立重启服务并使用：

```text
--enforce-eager
torch_profiler_with_stack=true
torch_profiler_record_shapes=true
torch_profiler_with_memory=false
torch_profiler_use_gzip=true
ignore_frontend=true
```

只抓足以覆盖主要 prefill/decode 路径的短窗口。通过 `/start_profile` 与 `/stop_profile` 控制采集边界。decode 时先接收首个 token，确认 prefill 已结束，再开始捕获后续 token。

mapping 产物至少保存：

```text
raw CUDA symbol
launching ATen/custom op
Python stack
input shape/dtype
vLLM source file:line
extension API/source
mapping coverage/confidence
```

mapping 是源码证据，不是性能证据。

## 5. formal trace

再次重启服务，保持 production CUDA Graph，使用轻量 profiler：

```text
torch_profiler_with_stack=false
torch_profiler_record_shapes=false
torch_profiler_with_memory=false
torch_profiler_use_gzip=true
```

每个场景采集：

```text
prefill:      prompt=ISL，捕获完整 prefill + 1 token
decode-early: prompt=ISL
decode-mid:   prompt=ISL + floor(OSL/2)
decode-late:  prompt=ISL + OSL - capture_margin
```

mid/late 直接使用更长的固定 token-ID prompt 建立对应 KV 长度。这样无需先生成数万 token，也不改变 decode kernel 看到的 context-length shape。

客户端请求的 capture token 数不等于 trace 内 graph replay 数。start/stop 与 GPU 异步，必须从每个 rank trace 统计真实 replay。

## 6. trace 完整性

通用检查：

```bash
python3 scripts/validate_trace_matrix.py \
  "$RUN_ROOT" \
  --mode formal \
  --tp-size 8
```

或手工检查：

```bash
find "$RUN_ROOT/profiles/formal/captures" -name '*.pt.trace.json.gz' -print0 \
  | xargs -0 -n1 gzip -t
find "$RUN_ROOT/profiles/formal/captures" -name capture.complete | wc -l
```

四场景、四 stage、TP8 的期望矩阵为 16 个 stage、128 个 trace。还要检查：

- `probe.json` 与 stage 的 prompt/capture 口径一致；
- 服务日志证明 FULL decode 和 PIECEWISE mixed graph capture/replay；
- trace 时间覆盖目标 graph replay，不是启动/JIT；
- 没有 rank 缺失、空文件、截断 gzip 或重复 attempt。

## 7. leaf 时间统计

优先复用项目已有 analyzer。若需实现新 analyzer，遵守以下算法：

1. 读取所有 rank 的 GPU events。
2. 保留 raw symbol，另生成稳定 canonical family。
3. 对 family 内 `[start,end)` interval 排序并求 union。
4. decode family union time 除以该 rank 的实际 graph replay 数。
5. 对 TP ranks 取中位数，并保存 min/max/CV。
6. prefill 使用完整阶段时间。
7. 对 decode early/mid/late 做 Simpson 积分：

```text
decode_us_per_step = (early_us + 4 * mid_us + late_us) / 6
decode_total_us    = decode_us_per_step * OSL
family_total_us    = prefill_us + decode_total_us
```

若三点不平滑，补 25%/75% 两点后改用五点积分，不要隐藏非线性。

占比：

```text
family_share = family_interval_union / all_gpu_activity_interval_union
```

不要用 E2E wall time 当 GPU share 分母。不同 family 之间仍可能 overlap；算术和仅用于候选筛选，严格上界需检查跨 family union。

输出至少包含：

```text
scenario, stage, rank
raw symbol, canonical family
calls/replays
prefill time, decode per-step, integrated decode, total
GPU busy share
rank skew
mapping coverage/confidence
parent/leaf/replacement group
status
```

## 8. 热点到源码

对前 N 个 leaf 建立完整调用链：

```text
raw CUDA symbol
  -> registered/custom op
  -> vLLM backend method
  -> extension Python binding
  -> C++/CUDA/Triton source
  -> build target and installed binary
```

逐个判断：

- wrapper：没有独立 kernel，不作为直接替换目标；
- NCCL/driver：先考虑通信拓扑、payload、融合和并行策略，不伪装成自写 kernel；
- compile-fused：在 graph-on 下已融合，修改孤立原函数可能不命中；
- leaf：可定位源文件、生产 shape 和 dispatch，优先进入优化；
- not-observed：写 N/A 并解释，不填 0。

使用以下预测筛选，而不是只看 micro speedup：

```text
saved_kernel_us = baseline_kernel_us * (1 - 1 / micro_speedup)
predicted_e2e_gain_upper = saved_kernel_us / measured_e2e_us
```

该值是筛选或上界，不是 E2E 结果。

## 9. 优化与构建

在修改前写优化设计：

```text
命中 shape 和 runtime 证据
瓶颈：计算/访存/occupancy/launch/sync/load balance/workspace
唯一核心改动
Hopper 特性或限制
dispatch 条件与回退路径
可能改变的归约顺序和数值风险
预期 kernel saving 与 E2E 上界
```

候选构建必须固定源码 commit、CUDA/NVCC、arch、编译 flags 和 job 配置。保存：

```text
patch
source tree hash/commit
build command/log
image inspect/digest
extension SHA256
symbol/strings/line-info
runtime-hit marker
```

若环境变量只是补丁新增的 dispatch 开关，文档必须说明它不是“打开现有功能”。

## 10. 正确性和微基准

同 binary benchmark 优先用于隔离 patch；跨 binary benchmark 用于验证最终部署物。两者不能混为一个结论。

覆盖：

```text
生产主 shape
低/高 batch 或 query rows
context-length 边界
alignment 边界
回退 shape
CUDA Graph replay
反向/交错运行顺序
```

比较完全相同的输入 hash。对模型可见输出执行高精度 oracle 门禁；对未消费输出单独报告，不静默忽略。检查 kernel resource 和 spill。

## 11. 正式 E2E

使用独立容器/服务执行：

```text
B1 -> C1 -> C2 -> B2 -> B3 -> C3
pair: B1/C1, B2/C2, B3/C3
```

每个 arm：

1. 启动并保存最终配置；
2. 完成通用 warmup 和本场景 shape warmup；
3. 确认 runtime-hit；
4. 记录 timed region 前 JIT 数；
5. 执行相同 prompt/config；
6. 全程检查 GPU 排他性；
7. 再次检查 JIT、token、hash、acceptance；
8. 保存日志、结果和二进制身份后销毁容器。

优先在同一主机配对。跨主机结果只做外部一致性证据，不组成严格 pair。

## 12. 失败诊断

| 现象 | 优先排查 |
| --- | --- |
| eager 很快、formal 无收益 | graph 路径已融合或 dispatch 不同 |
| kernel 有收益、E2E 无收益 | 占比低、CPU/sync 空洞、并发饱和、上界过小 |
| 只有某些场景收益 | context、batch、query rows、prefill/decode mix |
| MoE token/s 大幅变化 | prompt、temperature、生成退化、专家路由 |
| speculative 结果波动 | acceptance、draft 长度、输出 hash、调度顺序 |
| candidate 输出不同 | 数值误差、归约顺序、跨 binary 差异；用 oracle 判定 |
| trace 缺 kernel | CUDA Graph、capture 窗口、mapping coverage、被融合 |
| rank 差异大 | expert imbalance、通信、外部进程、时钟/温度 |
| 正收益超过 Amdahl 上界 | 测量口径不同、复合 binary、重叠或归因错误 |
