---
id: kernel-grouped-gemm
title: Grouped GEMM Contracts for MoE and NVFP4
type: kernel
architectures:
- sm100
- sm100a
- sm90
tags:
- grouped-gemm
- moe
- gemm
- fp8
- nvfp4
- tile-scheduling
confidence: source-reported
reproducibility: snippet
kernel_types:
- grouped-gemm
- gemm
- moe
languages:
- python
related:
- kernel-fused-moe
- kernel-deepgemm
- technique-tile-scheduling
- hw-clc
sources:
- contest-gpumode-p4
- blog-deepgemm
- blog-gpu-mode-reward-hack
performance_claims: []
blackwell_relevance: The official challenge targets B200 with independent
  per-group NVFP4 shapes; no particular CUTLASS, CLC, TMA, or launch topology
  is required by the observable contract.
---

# Grouped GEMM Contracts for MoE and NVFP4

## Verified scope

The official NVIDIA rules identify NVFP4 Grouped GEMM as Kernel Challenge 4 of the Blackwell NVFP4 Hackathon, open from January 17 through February 13, 2026. It carries 40% of the four-problem grand-prize score. The corrected public task at commit `ae67948` targets NVIDIA B200.

“Grouped GEMM” does not imply one universal shape contract. Two relevant interfaces differ materially:

- The GPU Mode challenge accepts a list of independent problems; `M_i`, `N_i`, and `K_i` can all differ by group.
- DeepGEMM's M-grouped MoE APIs vary M while holding N and K fixed. It separately provides a K-grouped interface for MoE weight backward.

Neither interface definition proves that a conforming implementation uses exactly one GPU launch.

## GPU Mode Problem 4 contract

For each group `i`, the correctness reference computes `C_i = A_i @ B_i.T` with block scales and FP16 output.

| Per-group value | Published dtype | Logical shape |
| --- | --- | --- |
| `a_i` | packed NVFP4 E2M1, two values per byte | `[M_i,K_i/2,L_i]` |
| `b_i` | packed NVFP4 E2M1, two values per byte | `[N_i,K_i/2,L_i]` |
| `c_i` | FP16 | `[M_i,N_i,L_i]` |
| `sfa_i` | FP8 E4M3FNUZ in task/template | `[M_i,K_i/16,L_i]` |
| `sfb_i` | FP8 E4M3FNUZ in task/template | `[N_i,K_i/16,L_i]` |
| problem size | integers | `(M_i,N_i,K_i,L_i)` |

The submission object actually contains four lists: logical A/B/C tensors, logical scales, reordered scale copies, and problem sizes. The `task.yml` prose lists only three tuple members, but `task.py`, `template.py`, and `reference.py` expose the reordered scales as the fourth. Each published case has `L=1`; `K_i` is divisible by 256, and `M_i`/`N_i` must satisfy the selected MMA tile divisibility.

The pinned upstream files also disagree on the scale dtype suffix: task/template text says E4M3FNUZ, while `reference.py` constructs `torch.float8_e4m3fn`. A submission must follow the actual tensors it receives rather than silently treating those names as interchangeable. The correctness checker uses `rtol=1e-3` and `atol=1e-3`.

For small non-empty ordinary-number matrices, this CPU-only reference isolates the group semantics. Each `b` is stored as `[N,K]`, so its rows are the columns of the mathematical right operand:

```python
def grouped_reference(groups):
    outputs = []
    for a, b in groups:
        k = len(a[0])
        if any(len(row) != k for row in a) or any(len(row) != k for row in b):
            raise ValueError("ragged or incompatible K")
        outputs.append([
            [sum(x * y for x, y in zip(a_row, b_row)) for b_row in b]
            for a_row in a
        ])
    return outputs
```

It models neither NVFP4 packing nor block scales; those remain part of the challenge ABI above.

## DeepGEMM's different grouped layouts

The following shapes describe the pinned FP8/FP4 grouped APIs; scale tensors and layout conversions are additional required inputs.

| Mode | Main tensors | Group descriptor | Documented role |
| --- | --- | --- | --- |
| M-grouped contiguous | A `[M,K]`, B `[G,N,K]`, D `[M,N]` | `grouped_layout` is either one expert ID per packed row (with `-1` padding) or one prefix-sum end per group | Training forward or inference prefill; N/K fixed and expert segments M-block aligned |
| M-grouped masked | A `[G,Mmax,K]`, B `[G,N,K]`, D `[G,Mmax,N]` | `masked_m` is one int32 valid-row count per group | CUDA-graph decode; fixed allocation while computing valid portions |
| K-grouped contiguous | packed A `[sum(K_i),M]`, B `[sum(K_i),N]`, D `[G,M,N]` | host and device K-size lists; optional C has `[G,M,N]` | MoE weight backward; M/N fixed |

These are concrete library contracts, not generic C++ structs. Compatibility also depends on the documented architecture, dtype/scale format, operand-major mode, alignment, output dtype, and recipe constraints.

## Published workloads and theoretical bounds

| Groups | M values | N | K | L | Task speed-of-light time (µs) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8 | 80, 176, 128, 72, 64, 248, 96, 160 | 4096 | 7168 | 1 | 18.833 |
| 8 | 40, 76, 168, 72, 164, 148, 196, 160 | 7168 | 2048 | 1 | 10.667 |
| 2 | 192, 320 | 3072 | 4096 | 1 | 2.406 |
| 2 | 128, 384 | 4096 | 1536 | 1 | 1.525 |

Ranking uses the geometric mean. The task labels these microsecond values a speed-of-light analysis derived from the maximum of B200 FP4 Tensor Core math time and DRAM-memory time at a 1.5 GHz clock. They are theoretical comparison values, not measured contestant results.

## The scrubbed reward hack

GPU Mode's official postmortem records a submission that temporarily reached the number-one leaderboard position with a reported `11.191 µs`, roughly `2 µs` ahead of the next entry, and was scrubbed minutes after the competition.

During correctness, it ran a real padded 8-group kernel on each of 15 cloned data objects. During timing, the first call launched one merged 120-group kernel covering all 15 objects; calls 2 through 15 returned cached output pointers. The harness then divided the combined timing by 15. The reported number is evidence about the exploit, not a valid per-call performance record.

The official post points to `gpu-mode/reference-kernels` PR #104 as the harness response. It does not attribute a FlashInfer-Bench or MLSys 2026 methodology change to this incident.

## Implementation and performance boundary

The public challenge constrains outputs, tolerances, workloads, and scoring. It does not require CUTLASS, CLC, TMA, TMEM, a persistent kernel, a static schedule, or one launch. The former CUTLASS and CUDA sketches were removed because they were not executable instances of the named APIs.

CUTLASS documents that small M or N can leave threads outside the useful problem bounds, and that a small M/N grid with large K can launch too few threadblocks to use every multiprocessor. This is a possible shape effect, not proof that every grouped workload has the same bottleneck. GPU Mode's postmortem, for example, measured substantial fixed setup cost in its studied implementation.

CLC itself uses an asynchronous cancellation request, shared response, mbarrier completion, and response decoding after a worker's initial block. That is different from a global `atomicAdd` tile queue. Whether CLC improves end-to-end time relative to a static or software-persistent scheduler requires a workload- and implementation-specific measurement.

TMA does not have one universal 128-byte alignment rule from which minimum tile sizes follow. CUDA 13.2.1's tiled tensor-map API generally documents 16-byte global-address and stride alignment, a 64-byte descriptor, and feature-specific 32-byte constraints. An implementation must check the exact descriptor mode and version it uses.

## Primary sources

- [Official challenge rules](https://developer.download.nvidia.com/licenses/Blackwell-NVFP4-Hackathon-Terms-and-Conditions.pdf)
- [Pinned public task](https://github.com/gpu-mode/reference-kernels/blob/ae67948685dfccf54ae8374dc9402addb7aae4f6/problems/nvidia/nvfp4_group_gemm/task.yml)
- [Pinned starter template](https://github.com/gpu-mode/reference-kernels/blob/ae67948685dfccf54ae8374dc9402addb7aae4f6/problems/nvidia/nvfp4_group_gemm/template.py)
- [Pinned correctness reference](https://github.com/gpu-mode/reference-kernels/blob/ae67948685dfccf54ae8374dc9402addb7aae4f6/problems/nvidia/nvfp4_group_gemm/reference.py)
- [Pinned DeepGEMM grouped overview](https://github.com/deepseek-ai/DeepGEMM/blob/891d57b4db1071624b5c8fa0d1e51cb317fa709f/README.md#grouped-gemms-contiguous-layout)
- [Pinned DeepGEMM grouped APIs](https://github.com/deepseek-ai/DeepGEMM/blob/891d57b4db1071624b5c8fa0d1e51cb317fa709f/csrc/apis/gemm.hpp)
- [Official reward-hack postmortem](https://www.gpumode.com/news/reward-hacking-nvfp4)
- [CUDA 13.2.1 tensor-map constraints](https://docs.nvidia.com/cuda/archive/13.2.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
- [CUDA 13.2 CLC programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html)
- [Pinned CUTLASS efficient-GEMM guide](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/media/docs/cpp/efficient_gemm.md)

Query via:

```bash
conda run -n base python scripts/get_page.py kernel-grouped-gemm
```
