---
id: technique-register-budgeting
title: "Register Budgeting for Occupancy"
type: technique
architectures: [sm100, sm90]
tags: [register-budgeting, register-reuse]
confidence: verified
evidence_basis:
  - source_id: doc-cuda-register-controls
    evidence_type: official-doc
reproducibility: concept
prerequisites: []
related: [pattern-memory-bound, pattern-register-pressure, kernel-nvfp4-gemv]
sources: [doc-cuda-register-controls, blog-amandeep-nvfp4, doc-ptx-isa-sm100]
blackwell_relevance: "SM100 TMEM removes a resident tcgen05 D fragment from the general-purpose register file, but operands, control state, and TMEM-to-register epilogue batches still consume registers; occupancy remains a property of the compiled kernel and launch."
---

# Register Budgeting

## Resource model

Registers per thread are one input to residency. Threads per block, static and dynamic shared memory, architectural limits, allocation granularity, barriers, and cluster configuration can impose other limits. Register thresholds therefore produce discrete changes in active blocks or warps; occupancy is not generally the reciprocal of a source-level register count, and higher predicted occupancy does not by itself imply lower latency.

The CUDA occupancy API evaluates the compiled function with an intended block size and dynamic shared-memory size. `cudaOccupancyMaxActiveBlocksPerMultiprocessor` returns the maximum active blocks per SM for that configuration. Convert that result to active warps only after multiplying by the actual warps per block.

## What the controls mean

| Control | Documented effect | What it does not establish |
|---|---|---|
| `--maxrregcount=N` | Sets a maximum for GPU functions, subject to the ABI minimum and compiler-reserved registers | That the compiler will use exactly `N`, or that residency/performance will change |
| `__launch_bounds__(T, B)` | Supplies maximum threads per block and desired minimum blocks per SM; the compiler derives an architecture-dependent register limit `L` | Exactly `B` resident blocks when shared memory or another resource is limiting |
| `--resource-usage` | Reports registers, stack frame, spill loads/stores, and other resources for compiled functions | Runtime occupancy or the cost of those resources |
| Occupancy API | Predicts maximum concurrent blocks for the compiled function and launch inputs | Achieved latency, bandwidth, or throughput |

When launch bounds make initial register use exceed `L`, CUDA documents that the compiler usually trades registers for more local-memory use and/or instructions. When both `T` and `B` are supplied and initial use is below `L`, the compiler may instead increase register use up to `L` to reduce instruction count. A launch bound is therefore compiler guidance and a launch constraint, not an exact register assignment.

## Reproducible sweep

Compile the same source and target first without a cap, then with candidate caps. `--resource-usage` makes a cap that does not change the compiled allocation visible.

```bash
nvcc -arch=sm_100a --resource-usage kernel.cu -o kernel-default
nvcc -arch=sm_100a --maxrregcount=64 --resource-usage kernel.cu -o kernel-r64
```

For each binary and each supported production shape:

1. Record toolkit, target, compiler options, registers per thread, stack-frame bytes, spill loads/stores, static/dynamic shared memory, block size, and cluster shape.
2. Query active blocks with the occupancy API using the exact function, block size, and dynamic shared-memory bytes. Identify which resource is actually limiting residency.
3. Verify that the candidate cap changed generated resources or code before attributing a timing change to it. Inspect PTX/SASS when instruction selection or loop shape may have changed.
4. Run the same correctness oracle, warmup, synchronization, inputs, and repeated-trial statistic for every variant. Profile spill traffic, memory stalls, and achieved warps rather than assuming the predicted occupancy is reached or useful.
5. Keep a cap only for the measured target and workload where it improves the declared metric without violating correctness or a resource contract.

Spill loads and stores use thread-local stack memory backed by the memory hierarchy. They add instructions and traffic; a memory-bound label is not evidence that this cost will be hidden. Conversely, a lower cap can be irrelevant when the uncapped allocation is already below it.

## Blackwell and TMEM

For tcgen05 MMA, the resident D accumulator is in TMEM rather than a per-thread D register vector. This changes one major register consumer, but it does not erase register pressure: descriptors, addresses, pipeline/control state, operands, and epilogue batches still use general-purpose registers. The pinned SM100 DeepGEMM path explicitly loads completed TMEM values into registers before its shared/global-store epilogue.

Treat the removed D vector as shape-specific. For example, an SM90 m64n256 FP32 WGMMA fragment exposes 128 D registers per participating thread, but that number is not a universal Blackwell saving and does not predict a resident-block transition.

## Scope of the cited NVFP4 observation

Amandeep Singh reports that three inspected B200 NVFP4 GEMV solutions clustered around an 18.5 microsecond geometric mean, with a 32-register cap in rank 1 and 45 in rank 3. The same report says those solutions also differed in PTX decode, cache policies, load widths, exact-K specialization, and B reuse; it provides no controlled register-cap ablation or public contestant code. In the author's own kernel, lowering the cap from 80 to 64 had no effect because natural use was already below 64, while extra accumulator chains and software pipelining increased pressure and regressed.

That observation motivates inspecting register allocation; it does not prove that 32 beats 45, that occupancy caused the ranking, or that aggressive caps generally help memory-bound kernels.
