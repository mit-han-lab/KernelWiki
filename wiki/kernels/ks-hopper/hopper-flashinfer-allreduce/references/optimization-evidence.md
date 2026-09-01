# FlashInfer AllReduce 实现与效果基线

## 实现范围

经验实现涉及以下模块：

- `vllm/distributed/device_communicators/flashinfer_all_reduce.py`
- `vllm/distributed/device_communicators/cuda_communicator.py`
- `tests/distributed/test_comm_ops.py`

核心改动：

1. MiB、workspace 与 max token 计算保持整数，兼容 1.5 MiB 等小数阈值。
2. standalone FlashInfer AllReduce 读取 `compilation_config.pass_config.fi_allreduce_fusion_max_size_mb`。
3. CUDA、contiguous、2-D、FP16/BF16/FP32 且阈值内的张量才 eligible。
4. 显式启用的 eligible FlashInfer 路由优先于 symmetric-memory，其余安全回退。

## 原始缺陷

- float token 数进入 FlashInfer workspace API 后可能触发 `TypeError: unsupported operand type(s) for &: 'float' and 'int'`。
- fusion pass 与 standalone 路径可能使用不同阈值。
- 即使 `VLLM_ALLREDUCE_USE_FLASHINFER=1`，符合条件的张量也可能先被 symmetric-memory 消费。

## 已验证结果

静态与单元测试：14 passed、14 deselected；`py_compile` 与 `git diff --check` 通过。

单机 8×H200、TP8、hidden size 7168、BF16，warmup 100、measure 500 的单算子中位数：

| tokens | SymmMem | FI MNNVL | 延迟下降 |
|---:|---:|---:|---:|
| 36 | 10.733 us | 5.728 us | 46.63% |
| 64 | 13.418 us | 7.754 us | 42.21% |
| 112 | 16.230 us | 11.405 us | 29.73% |
| 128 | 17.078 us | 12.352 us | 27.67% |

真实模型 trace：baseline 的 MNNVL kernel 为 0 次、multimem 为 615 次；candidate 的 MNNVL kernel 为 35 次、multimem 仍为 615 次。该结果证明路由生效，也证明 2 MiB 仅覆盖部分小张量。

端到端首次正式 paired 结果为 mixed：1K→1K 为 -1.560%，8K→1K 为 +1.064%。同机 1K→1K 十轮确认得到 candidate mean +4.008%、median +3.219%、paired median +2.534%，7/10 对为正。另一台 H200 的十轮复测中位数为 +4.223%，但存在异常 baseline 慢轮；敏感性分析后的 8 对 mean +2.749%、paired median +2.863%，6/8 为正。

## 结论边界

证据支持：整数缺陷已修复；显式阈值被 standalone 读取；符合条件的小张量优先进入 MNNVL；fallback 有效；目标 H200 上阈值内单算子有优势；真实模型路由生效。

证据不支持：固定 4% 端到端收益；把 2 MiB 设为所有用户默认值；所有 AllReduce 都切换到 FlashInfer；把 27%–47% 单算子延迟下降等同于模型吞吐提升。

因此能力定位为 `ACCEPT_EXPLICIT_OPT_IN`。部署前必须在目标机器、目标 workload 上重新做交替顺序的 paired A/B。

## 经验复用原则

- 把代码提交和实验轮次视为知识来源，不把它们写入 Skill 的对外命名与触发条件。
- 复用实现不变量、验证流程和结论边界；具体性能数字只作为已观测效果，不作为验收承诺。
- 换 GPU 拓扑、模型、shape、并发、版本或阈值后，重新建立 baseline 并做交替顺序 paired A/B。
