---
id: blog-deepgemm
title: DeepGEMM — Pinned Upstream Project Summary
author: DeepSeek AI
url: https://github.com/deepseek-ai/DeepGEMM/tree/891d57b4db1071624b5c8fa0d1e51cb317fa709f
source_category: benchmark-blog
architectures:
- sm100
- sm90
tags:
- gemm
- fp8
- fine-grained-quantization
- block-scale
- jit-compilation
- tcgen05
- wgmma
retrieved_at: 2026-04-27
artifact_dir: artifacts/kernels/deepgemm/full
---

## Scope

This source entry summarizes DeepGEMM at commit [`891d57b4db1071624b5c8fa0d1e51cb317fa709f`](https://github.com/deepseek-ai/DeepGEMM/tree/891d57b4db1071624b5c8fa0d1e51cb317fa709f). The local SM90 and SM100 FP8 1D1D files are byte-verified copies from that commit; see their [`PROVENANCE.yaml`](../../artifacts/kernels/deepgemm/full/PROVENANCE.yaml).

## Verified Techniques

- SM90 uses FP32 scale factors. Its pinned 1D1D kernel fixes `BLOCK_K == 128`, accumulates WGMMA partial results in `float accum[...]`, and applies A/B scales into a separate `float final_accum[...]` after each K block.
- SM100 uses packed UE8M0 scale factors. Its pinned 1D1D kernel copies factor blocks into TMEM, constructs a block-scaled UMMA descriptor, and accumulates through the selected TMEM operation without the SM90 CUDA-core `final_accum` loop.
- M-grouped contiguous and masked APIs vary M with N/K fixed. `masked_m` contains an integer valid-M length for each group. K-grouped APIs instead vary K while M/N remain fixed.
- Kernels are generated and JIT compiled. NVCC is the default compiler; `DG_JIT_USE_NVRTC=1` opts into NVRTC. Generated code and compiler settings participate in the cache signature.
- The pinned README exposes NT only for SM90 FP8 and NT/TN/NN/TT dense interfaces for SM100.

## Performance Scope

The pinned README contains a source-reported claim of **up to 1550 TFLOPS on H800**. It does not attach that number to `M=N=K=4096`, report approximately 90% utilization, or retain a complete reproduction environment. No more specific measurement is attributed to this source entry.

## Primary References

- [README at the pinned commit](https://github.com/deepseek-ai/DeepGEMM/blob/891d57b4db1071624b5c8fa0d1e51cb317fa709f/README.md)
- [GEMM API at the pinned commit](https://github.com/deepseek-ai/DeepGEMM/blob/891d57b4db1071624b5c8fa0d1e51cb317fa709f/csrc/apis/gemm.hpp)
- [JIT compiler at the pinned commit](https://github.com/deepseek-ai/DeepGEMM/blob/891d57b4db1071624b5c8fa0d1e51cb317fa709f/csrc/jit/compiler.hpp)
- [DeepSeek-V3 Technical Report v2](https://arxiv.org/abs/2412.19437v2)
