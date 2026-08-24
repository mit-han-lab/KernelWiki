---
id: blog-deepgemm
title: DeepGEMM tensor-core kernel library
author: DeepSeek AI
url: https://github.com/deepseek-ai/DeepGEMM
source_category: benchmark-blog
architectures: [sm100, sm90]
tags: [gemm, fp8, fp4, fine-grained-quantization, block-scale, jit-compilation, grouped-gemm]
retrieved_at: 2026-08-18
source_commit: 559d79fb6994a58b8a15b4b93bf13ccc16edf247
---

# DeepGEMM tensor-core kernel library

DeepGEMM's README at commit
`559d79fb6994a58b8a15b4b93bf13ccc16edf247` describes a runtime-JIT CUDA
library for FP8, FP4, BF16, grouped GEMMs, Mega MoE, and MQA-scoring kernels.
It requires an SM90 or SM100 GPU.

The README states these architecture-specific interface constraints:

- SM90 accepts NT layout and FP32 scaling factors.
- SM100 accepts NT, TN, NN, and TT layouts and packed UE8M0 scaling factors.
- M-grouped contiguous GEMMs keep N and K fixed and require each expert segment
  to satisfy the library's queried M-block alignment.
- M-grouped masked GEMMs accept a valid-M mask for graph-friendly decode.
- K-grouped APIs serve weight-gradient workloads with fixed M and N.

The current README says kernels are compiled at runtime through a lightweight
JIT module. NVRTC is optional rather than universal: the documented default
compiler path is NVCC, and the README warns that enabling NVRTC can reduce
performance on some cases.

## Source-reported performance

The 2025-04-18 news entry reports “up to 1550 TFLOPS” on H800 and links the
associated PRs and commit. It does not attach that maximum to one shape in the
README entry, so the value must not be generalized to arbitrary GEMMs.

Earlier local code blocks were removed because they were synthesized sketches,
not excerpts from DeepGEMM. The repository's verbatim kernel bundle is kept
separately under `artifacts/kernels/deepgemm/full/` with its own commit pin and
hashes.
