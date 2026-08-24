---
id: hw-pdl-gdc
title: "Programmatic Dependent Launch / Grid Dependency Control"
type: hardware
architectures: [sm90, sm100, sm120]
tags: [pdl, gdc]
confidence: source-reported
related: [technique-persistent-kernels, hw-clc]
sources: [doc-cuda-programming-guide-pdl, pr-cutlass-2161, doc-cutlass-changelog-sm100]
aliases: [PDL, GDC, "programmatic dependent launch", "grid dependency control"]
blackwell_relevance: "Blackwell supports PDL, but the dependent launch still requires an explicit launch attribute and device-side synchronization."
---

## Overview

Programmatic Dependent Launch (PDL), also called Grid Dependency Control in
some libraries, can let a dependent kernel begin its independent prologue before
the preceding kernel in the same stream has finished. The CUDA Programming
Guide makes the feature available starting at compute capability 9.0; it is not
a Blackwell-only mechanism.

## Required protocol

1. Every block of the primary kernel calls
   `cudaTriggerProgrammaticLaunchCompletion()` after producing everything
   needed to permit the secondary launch.
2. The host submits the secondary with
   `cudaLaunchAttributeProgrammaticStreamSerialization` enabled through the
   extensible launch API.
3. The secondary may perform work that does not consume the primary's results,
   then calls `cudaGridDependencySynchronize()` before dependent work.

If a primary block does not call the trigger, its trigger occurs implicitly
when that block exits. The secondary may start before the primary's writes are
visible, which is why the secondary-side dependency synchronization (or another
documented way to verify availability) is required.

## Limits

- The runtime is permitted, but not required, to overlap the launches.
- Ordinary back-to-back launches do not opt into PDL by themselves.
- Correctness and progress must not depend on concurrent execution; resource
  pressure may serialize the kernels.
- A speedup requires useful independent secondary work and enough resources for
  overlap. Measure the actual kernel pair rather than assuming that PDL reduces
  wall-clock time.

## Related
- [persistent-kernels](../techniques/persistent-kernels.md) — Alternative approach to reducing launch overhead
- [clc](clc.md) — A distinct same-grid cluster-cancellation mechanism
