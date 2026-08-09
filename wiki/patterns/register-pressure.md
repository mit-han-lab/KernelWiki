---
id: pattern-register-pressure
title: "Register Pressure and Residency"
type: pattern
tags: [tmem, register-reuse, warp-specialization]
symptoms: [register-pressure, low-occupancy, register-spilling]
candidate_techniques: [hw-tmem, technique-warp-specialization, technique-register-budgeting, migration-register-to-tmem]
related: [pattern-compute-bound, hw-tmem]
sources: [doc-nvidia-tuning-guide, blog-tcgen05-tutorial, doc-ptx-isa-sm100]
confidence: verified
evidence_basis:
  - source_id: doc-nvidia-tuning-guide
    evidence_type: official-doc
reproducibility: concept
---

## Diagnose a Register-Limited Kernel

Register pressure is performance-relevant only when compiled register allocation or spills constrain useful scheduling or add material local-memory traffic. “Occupancy below target” is not enough: theoretical occupancy is a residency limit, achieved occupancy is runtime behavior, and maximum occupancy is not a universal optimum.

For the exact binary and launch, record registers per thread, allocation granularity, spill stores/loads, local-memory traffic, threads and warps per CTA, SMEM, cluster shape, theoretical limiting resource, achieved active/eligible warps, scheduler stalls, and end-to-end time. Inspect generated SASS/source correlation to find the live ranges or spill sites. Then change one source or compiler factor and require the predicted resource/stall movement plus a runtime improvement.

Common contributors include a resident MMA fragment, an epilogue whose temporaries overlap the mainloop, descriptors/addresses and pipeline state, unrolled loops, and values live across branches. These are hypotheses about compiled liveness, not conclusions from source complexity alone.

## Candidate Tests

### Shorten or separate live ranges

Move epilogue work after accumulator lifetime when dependencies allow, reduce unnecessary unrolling, or split a long-lived role. Warp specialization exists on Hopper as well as Blackwell; it can shorten one role's live set, but adds role state and synchronization and does not guarantee fewer registers. Compare compiled resources and time for matched role layouts.

### Test a register cap

Compare an uncapped build with selected `-maxrregcount` values. Record whether the requested cap is effective, the occupancy-limiting resource, spill traffic, instructions, and runtime. ABI minima can constrain the cap, another resource can remain limiting, and spills can erase any benefit.

### Move resident D from registers to TMEM

For Hopper `wgmma.mma_async.m64nNk16` with FP32 D, each warpgroup thread holds `N/2` accumulator registers. At `N=256`, that is 128 32-bit registers (512 bytes) per thread for D. This exact fragment arithmetic does not include other live state or guarantee a particular resident-block count.

On the SM100 tcgen05 path, resident D is in TMEM, organized as 128 lanes by 512 columns of 32-bit cells. That removes the long-lived per-thread WGMMA D fragment, but not all accumulator-related register work: addresses, descriptors, barriers, loop state, `tcgen05.ld` destinations, conversion, and epilogue temporaries still consume registers. Logical capacity and required columns depend on MMA kind, shape, and packing, so a largest-operation or always-double-bufferable rule cannot be inferred from raw cell count.

The migration must implement collective allocation/address publication, tcgen05 MMA completion, cross-thread ordering where applicable, collective TMEM loads and `tcgen05.wait::ld`, consumer completion, and collective deallocation before exit. The described TMEM/tcgen05 path targets SM100-class data-center architectures and is not the SM120 consumer MMA path.

## Evaluate the Tradeoff

TMEM loads are asynchronous operations with explicit completion waits. Whether their issue/wait and epilogue work are repaid by lower long-lived register use is an empirical whole-kernel question. Compare the Hopper and Blackwell designs only with equivalent math and precision; for same-architecture alternatives, compare TMEM region count/layout while holding tile and launch policy fixed.

Report compiled registers and spills, TMEM columns, SMEM, barriers, occupancy limit, achieved warps, pipeline/scoreboard stalls, correctness, and time. Include shapes where register occupancy changes and shapes where it does not. A reduction in register count without a useful residency, issue, traffic, or time improvement is not sufficient.

## Primary References

- [Nsight Compute 2025.3 register and occupancy analysis](https://docs.nvidia.com/nsight-compute/2025.3/ProfilingGuide/index.html)
- [PTX ISA 9.0 WGMMA accumulator fragments](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma)
- [PTX ISA 9.0 Tensor Memory](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#tensor-memory)
- [PTX ISA 9.0 tcgen05 load/wait](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions-tcgen05-ld)
- [Pinned SM100 tutorial implementation](https://github.com/gau-nernst/learn-cuda/tree/3b90ac9b3f624bdf1f6f78d02dcd533675d36573/02e_matmul_sm100)
