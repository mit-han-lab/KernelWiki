---
id: doc-cuda-programming-guide-pdl
title: "CUDA Programming Guide: Programmatic Dependent Launch"
url: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html
source_category: official-doc
architectures: [sm90, sm100, sm120]
tags: [pdl, gdc]
retrieved_at: 2026-08-18
---

# CUDA Programmatic Dependent Launch

This page maps the wiki's PDL claims to section 4.5 of NVIDIA's rolling CUDA
Programming Guide, accessed on 2026-08-18.

The guide says that PDL is available starting with compute capability 9.0. A
primary and secondary kernel are submitted to the same stream. Every block of
the primary calls `cudaTriggerProgrammaticLaunchCompletion()` when it is ready
to permit the secondary launch. The secondary is submitted through the
extensible launch API with
`cudaLaunchAttributeProgrammaticStreamSerialization` enabled, performs work
that is independent of the primary, and calls
`cudaGridDependencySynchronize()` before consuming the primary's results.

The driver may then start the secondary after all primary blocks have launched
and reached the trigger. If a primary block never calls the trigger, its trigger
occurs implicitly when the block exits. The guide explicitly describes overlap
as opportunistic rather than guaranteed and warns against relying on concurrent
execution for correctness or progress.

Exact locators:

- §4.5.1, “Background”
- §4.5.2, “API Description”
- §4.5.3, “Use in CUDA Graphs”
