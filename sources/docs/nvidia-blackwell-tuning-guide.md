---
id: doc-nvidia-tuning-guide
title: "NVIDIA Blackwell Tuning Guide"
url: https://docs.nvidia.com/cuda/blackwell-tuning-guide/
source_category: official-doc
architectures: [sm100, sm100a]
tags: [tcgen05, tmem, clc, tma, 2sm-cooperative, nvfp4, fp8, fp4, block-scale, pdl, gdc]
retrieved_at: 2026-04-16
---

# NVIDIA Blackwell Tuning Guide

## Scope

Official NVIDIA tuning guidance for Blackwell. Exact tcgen05 instruction grammar, operand locations, descriptor fields, target constraints, and ordering rules should be checked in the version-pinned PTX ISA rather than inferred from this higher-level tuning page.

## Evidence-scoped hardware summary

- PTX describes tcgen05 as the fifth-generation TensorCore family. Dense `tcgen05.mma` is asynchronous and is issued by one thread.
- D resides in TMEM. A can be described in SMEM or addressed in TMEM; B is described in SMEM.
- The dense kinds in PTX ISA 9.0 are `f16`, `tf32`, `f8f6f4`, `i8`, `mxf8f6f4`, `mxf4`, and `mxf4nvf4`. The three MX forms use the block-scaled grammar.
- M and N are encoded by the instruction descriptor and have kind-, layout-, CTA-group-, and target-specific constraints. m128n256k16 and m256n256k16 are maximum-shape examples for common F16/BF16 configurations, not an exhaustive shape list.
- `cta_group::2` cooperates with a peer CTA and can access peer resources as defined by PTX. It does not imply one universal rule that doubles M for every legal configuration.
- `tcgen05.commit` plus an mbarrier provides completion tracking. `tcgen05.fence` provides ordering around execution-ordering operations; it does not replace the completion wait.
- The shared-memory descriptor supports multiple swizzle modes, including none, 128B, 64B, and 32B. The selected layout must satisfy its documented constraints.

## Cluster Launch Control

CLC launches the problem-sized grid. A worker begins with its own `blockIdx`; after that work, it may request cancellation of an unspecified not-yet-started block or cluster and process the returned identifier itself. This work-stealing mechanism can improve utilization in suitable persistent schedules, but it neither deletes application-selected work nor guarantees removal of every last-wave or load-imbalance effect.

## Use of performance claims

Treat quantitative claims as source- and environment-specific. The separate `tcgen05 for dummies` source reports one B200 M=N=K=4096 progression. Its final persistent result uses static scheduling, not CLC.

## Primary references

- [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
- [PTX ISA 9.0, CUDA 13.0.2 archive](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html)
- [CUDA Programming Guide: Cluster Launch Control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html)
