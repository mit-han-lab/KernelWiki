---
id: pattern-register-pressure
title: "Register Pressure — Low Occupancy"
type: pattern
tags: [tmem, register-reuse, warp-specialization]
symptoms: [register-pressure, low-occupancy, register-spilling]
candidate_techniques: [hw-tmem, technique-warp-specialization, migration-register-to-tmem]
related: [pattern-compute-bound, hw-tmem]
sources: [doc-nvidia-tuning-guide, doc-ptx-isa-sm100, blog-tcgen05-tutorial, pr-vllm-16032]
---

# Register Pressure

## Symptom

Compiler/resource reports show that register allocation constrains residency, or profiler/compiler output shows local-memory spills that materially affect execution. Low occupancy alone is not proof of a problem; some kernels perform best with one resource-heavy CTA per SM.

## Likely causes

- Large distributed accumulator fragments on SM90 WGMMA paths.
- Unrolled loops, address state, and long live ranges.
- Epilogue conversion, activation, or quantization fragments.
- Vectorized loads and multiple in-flight work units.

## SM90 versus SM100

```text
SM90 wgmma:
  accumulator fragment -> registers distributed across the warpgroup

SM100 tcgen05:
  accumulator layout -> TMEM
  epilogue fragment    -> registers after tcgen05.ld
```

TMEM removes the MMA destination from general registers but does not free a fixed number of registers or eliminate epilogue pressure. Exact register and TMEM-column counts come from the selected instruction shape, datatype, compiler allocation, and surrounding code.

## Candidate actions

- Shorten live ranges or move independent roles to specialized warps.
- Reduce unrolling/vector width when it lowers spills without starving issue.
- On SM100, use the TMEM-based `tcgen05` accumulator lifecycle.
- Sweep launch bounds/register limits and inspect generated resources.

TMEM-to-register transfer and completion synchronization have costs. Choose the fastest validated configuration, not the smallest register count or highest nominal occupancy.
