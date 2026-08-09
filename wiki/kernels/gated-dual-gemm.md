---
id: kernel-gated-dual-gemm
title: GPU Mode NVFP4 Gated Dual GEMM
type: kernel
architectures:
- sm100
- sm100a
tags:
- gated-dual-gemm
- gemm
- fused-kernel
- nvfp4
- block-scale
confidence: source-reported
reproducibility: snippet
kernel_types:
- gated-dual-gemm
- gemm
- fused-kernel
languages:
- python
related:
- kernel-nvfp4-gemm
- kernel-fused-moe
- technique-kernel-fusion
- technique-epilogue-fusion
sources:
- contest-gpumode-p3
performance_claims: []
blackwell_relevance: The official challenge targets NVIDIA B200 and its NVFP4
  block-scaled tensor contract; no winning instruction schedule is asserted.
artifact_dir: artifacts/kernels/gated-dual-gemm
---

# GPU Mode NVFP4 Gated Dual GEMM

## Verified scope

The official NVIDIA rules identify NVFP4 Gated Dual GEMM as Kernel Challenge 3 of the Blackwell NVFP4 Hackathon, open from December 20, 2025 through January 16, 2026. The public GPU Mode problem at challenge-opening commit `c5b2f7c` targets NVIDIA B200.

For each batch index, the required result is two block-scaled matrix products sharing the same left operand, with SiLU applied only to the first branch:

```python
def gated_dual_result(a, b1, b2, scale_a, scale_b1, scale_b2, scaled_mm):
    gate = scaled_mm(a, b1.T, scale_a, scale_b1)
    up = scaled_mm(a, b2.T, scale_a, scale_b2)
    return silu(gate) * up
```

This fixes result semantics and branch order. It does not require the submission entry point to use one GPU launch or forbid intermediate storage.

## Published tensor contract

| Tensor | Published dtype | Logical shape |
| --- | --- | --- |
| `a` | NVFP4 E2M1 | `[M,K,L]`, K-major |
| `b1`, `b2` | NVFP4 E2M1 | `[N,K,L]`, K-major |
| `sfa` | FP8 E4M3FNUZ | `[M,K/16,L]`, K-major |
| `sfb1`, `sfb2` | FP8 E4M3FNUZ | `[N,K/16,L]`, K-major |
| `c` | FP16 | `[M,N,L]` |

The submission tuple also supplies layout-reordered copies of `sfa`, `sfb1`, and `sfb2`, plus preallocated `c`. `K` is divisible by 256; `M` and `N` must be divisible by the selected MMA tile dimensions. The correctness checker uses `rtol=1e-3` and `atol=1e-3`.

The pinned upstream files contain one dtype-label inconsistency: `task.yml` and `template.py` call the scales E4M3FNUZ, while `reference.py` constructs `torch.float8_e4m3fn`. A backend must follow the actual submission objects rather than treating those spellings as interchangeable.

## Published workloads and theoretical bounds

| M | N | K | L | Task speed-of-light time (µs) |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 4096 | 7168 | 1 | 4.708 |
| 512 | 4096 | 7168 | 1 | 8.714 |
| 256 | 3072 | 4096 | 1 | 2.125 |
| 512 | 3072 | 7168 | 1 | 6.535 |

Ranking uses the geometric mean of the four benchmark times. The task explicitly labels the final column a speed-of-light analysis based on the maximum of B200 FP4 Tensor Core math time and DRAM-memory time at a 1.5 GHz clock. These values are theoretical comparison bounds, not measured contestant results. The former `M=1024, N=4096, K=7168, 18.5 µs` record is not one of the published workloads and has been removed.

## Implementation boundary

The shared `a` and `sfa` inputs create an opportunity to reuse data across the two products, and a genuinely fused epilogue can avoid writing both full-precision products to global memory. Those are optimization possibilities, not task guarantees.

The public task/reference does not establish a final leaderboard, winning source, single-launch decomposition, physical load count, TMEM accumulator partition, TMA pipeline, CUTLASS schedule, or compute-/memory-bound classification. If an implementation uses `tcgen05`, TMEM, or TMA, its exact descriptors, scale layouts, synchronization, allocation, and architecture requirements must be verified from that implementation and the version-matched ISA.

Compatibility is correspondingly narrower than “any dual-output operation”: a model/backend must match the two same-shaped NVFP4 products, scale storage and layouts, first-branch SiLU, FP16 output, batching, and divisibility rules. Grouped per-expert MoE execution is a different contract unless an implementation explicitly supplies that grouping layer.

## Local artifacts

The `full/` bundle contains the byte-pinned official `task.yml` from commit `c5b2f7c`; it is the problem specification, not an optimized submission. The unrelated third-party schedule extract and duplicate vLLM MXFP4 MoE patch formerly stored here were removed; the latter remains preserved in its own PR artifact collection.

The `variants/` bundle contains a standard-library semantic reference for two small dense products followed by `SiLU(gate) * up`. Its self-test compares compact and expanded forms and rejects the wrong alternative that gates the second branch. It does not model NVFP4 packing, block scales, GPU execution, or performance.

## Primary sources

- [Official challenge rules](https://developer.download.nvidia.com/licenses/Blackwell-NVFP4-Hackathon-Terms-and-Conditions.pdf)
- [Pinned public task](https://github.com/gpu-mode/reference-kernels/blob/c5b2f7c062d5015f29c3a1043cfd04954397944c/problems/nvidia/nvfp4_dual_gemm/task.yml)
- [Pinned starter template](https://github.com/gpu-mode/reference-kernels/blob/c5b2f7c062d5015f29c3a1043cfd04954397944c/problems/nvidia/nvfp4_dual_gemm/template.py)
- [Pinned correctness reference](https://github.com/gpu-mode/reference-kernels/blob/c5b2f7c062d5015f29c3a1043cfd04954397944c/problems/nvidia/nvfp4_dual_gemm/reference.py)

Query via:

```bash
conda run -n base python scripts/get_page.py kernel-gated-dual-gemm --include-code
```
