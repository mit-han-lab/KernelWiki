---
id: hw-pdl-gdc
title: "Programmatic Dependent Launch / Grid Dependency Control"
type: hardware
architectures: [sm100, sm100a, sm90]
tags: [pdl, gdc]
confidence: source-reported
related: [technique-persistent-kernels, hw-clc]
sources: [pr-cutlass-2161, doc-cutlass-changelog-sm100]
aliases: [PDL, GDC, "programmatic dependent launch", "grid dependency control"]
blackwell_relevance: "PDL is available from compute capability 9.0 onward and remains an explicit launch/dependency protocol on SM100."
---

# Programmatic Dependent Launch

## Overview

PDL allows a secondary kernel in the same stream to begin before its primary kernel has fully completed. The overlap is opportunistic and safe only when the secondary separates independent work from work that consumes primary results.

## Correct roles

```cuda
__global__ void primary_kernel(/* ... */) {
    produce_launch-safe_state();
    cudaTriggerProgrammaticLaunchCompletion();
    finish_work_that_may_overlap();
}

__global__ void secondary_kernel(/* ... */) {
    do_work_independent_of_primary_results();
    cudaGridDependencySynchronize();
    consume_primary_results();
}
```

The host opts the secondary launch into programmatic stream serialization with the extensible launch API (`cudaLaunchAttributeProgrammaticStreamSerialization`). It is not enabled automatically for all back-to-back SM100 launches.

If the primary does not explicitly trigger, launch completion is implicitly triggered after all its blocks exit. Even after an early launch, the secondary must synchronize before consuming dependent data.

## Limits

- Available starting at compute capability 9.0, not Blackwell-only.
- Concurrency is not guaranteed; code must remain correct if execution serializes.
- PDL does not remove ordinary data-dependency synchronization.
- Relying on concurrent progress can deadlock.
- It can reduce an exposed inter-kernel gap only when meaningful independent secondary work exists.

CUTLASS may expose policy defaults or global GDC configuration for particular kernels, but those library choices must not be generalized to CUDA launches as a whole.
