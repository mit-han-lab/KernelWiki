---
id: pattern-register-pressure
title: "Register Pressure — Low Occupancy"
type: pattern
tags: [tmem, register-reuse, warp-specialization]
symptoms: [register-pressure, low-occupancy, register-spilling]
candidate_techniques: [hw-tmem, technique-warp-specialization, migration-register-to-tmem]
related: [pattern-compute-bound, hw-tmem]
sources: [doc-ptx-isa-sm100, pr-vllm-16032]
---

# Register pressure

High live-register demand can reduce residency or cause local-memory spills. Confirm it with compiler resource output and profiler metrics before changing the kernel; occupancy alone is not a performance objective.

Common contributors include register-resident accumulators, epilogue intermediates, unrolled loop state, and long live ranges across asynchronous stages.

## Candidate responses

| Technique | Scope | Tradeoff |
| --- | --- | --- |
| TMEM accumulators | Fifth-generation Tensor Core path | Moves D out of per-thread registers, but adds explicit allocation, synchronization, and TMEM-to-register epilogue work. |
| Warp specialization | Kernel-specific | Gives roles different register budgets, but adds synchronization and may reduce flexibility. |
| Shorter live ranges or less unrolling | General | Can lower register count at the cost of more instructions or less instruction-level parallelism. |
| Smaller tiles | General | Reduces state per CTA but can reduce reuse or Tensor Core efficiency. |

On `sm_100a`/`sm_100f`, PTX exposes each CTA a TMEM view of 512 columns by 128 lanes of 32-bit cells. Treat the required column count as part of the kernel resource design; do not describe that logical CTA view as a documented physical byte capacity per SM.
