---
id: kernel-nvfp4-gemm
title: NVFP4 GEMM — GPU Mode Problem 2 Contract
type: kernel
architectures:
- sm100
- sm100a
tags:
- gemm
- nvfp4
- fp4
- block-scale
- tcgen05
- tmem
- tma
confidence: verified
evidence_basis:
- source_id: doc-transformer-engine-2.13-nvfp4
  evidence_type: official-doc
- source_id: doc-ptx-isa-sm100
  evidence_type: official-doc
reproducibility: snippet
kernel_types:
- gemm
languages:
- python
related:
- hw-nvfp4
- hw-tcgen05-mma
- hw-tmem
- hw-tma
- kernel-nvfp4-gemv
sources:
- contest-gpumode-p2
- doc-transformer-engine-2.13-nvfp4
- doc-ptx-isa-sm100
- doc-cuda-13-0-2-tma
performance_claims: []
blackwell_relevance: The official task targets B200 and exact packed-E2M1 and
  block-scale layouts; it does not require a particular CUTLASS schedule or
  establish that every problem shape is compute-bound.
---

# NVFP4 GEMM — GPU Mode Problem 2 Contract

## Verified scope

The official NVIDIA rules identify NVFP4 GEMM as Kernel Challenge 2 of the Blackwell NVFP4 Hackathon. The contest ran from November 29 through December 19, 2025; Problem 2 targeted NVIDIA B200 and contributed 20% of the four-problem grand-prize score.

The pinned public task defines observable inputs, correctness, test shapes, benchmark shapes, and ranking. It does not publish a canonical optimized kernel, require CUTLASS, or establish which implementation mechanisms any entrant used. “NVFP4 GEMM” also does not determine a bottleneck by itself: shape, layout, staging, launch geometry, and epilogue can change whether math, memory traffic, occupancy, or launch overhead controls runtime.

## Format recipe versus task ABI

The generic one-dimensional NVFP4 recipe reconstructs each value from a signed E2M1 payload, one E4M3 local scale per 16 consecutive payloads, and a per-tensor FP32 global scale. E2M1 represents zero and signed magnitudes 0.5, 1, 1.5, 2, 3, 4, and 6, with two payloads packed per byte. MXFP4 is different: its local groups contain 32 payloads and use power-of-two UE8M0 scales.

The contest ABI is narrower and does not expose the generic recipe's FP32 global scales. Its actual generated input has seven tensors:

| Tensor | Published representation | Physical shape |
| --- | --- | --- |
| `a` | packed E2M1, two values per byte | `[M,K/2,L]` |
| `b` | packed E2M1, two values per byte | `[N,K/2,L]` |
| `sfa` | logical E4M3 scales | `[M,K/16,L]` |
| `sfb` | logical E4M3 scales | `[N,K/16,L]` |
| `sfa_reordered` | MMA-oriented scale copy | `[32,4,ceil(M/128),4,K/64,L]` |
| `sfb_reordered` | MMA-oriented scale copy | `[32,4,ceil(N/128),4,K/64,L]` |
| `c` | preallocated FP16 output | `[M,N,L]` |

For each `L` slice, the correctness reference computes the block-scaled equivalent of `A @ B.T` and stores FP16 output. It uses `rtol=1e-3` and `atol=1e-3`.

The upstream files contain two interface inconsistencies that implementations must not conceal:

- `task.yml` describes a five-member `(a,b,sfa,sfb,c)` tuple, while `task.py`, `template.py`, and `reference.py` expose the seven tensors above.
- Task and template prose label scale tensors E4M3FNUZ, while `reference.py` constructs `torch.float8_e4m3fn` values. A submission must follow the tensors supplied by the actual harness rather than treating those suffixes as interchangeable.

## Host-checkable shape contract

The published task requires `K` divisible by 256. Divisibility of `M` and `N` depends on the submission's selected MMA tile. The following CPU-only helper reproduces the published storage shapes; it does not decode FP4, apply scales, or model a GPU kernel:

```python
def task_storage_shapes(m, n, k, l=1):
    if min(m, n, k, l) <= 0:
        raise ValueError("dimensions must be positive")
    if k % 256:
        raise ValueError("K must be divisible by 256")

    return {
        "a_packed": (m, k // 2, l),
        "b_packed": (n, k // 2, l),
        "sfa_logical": (m, k // 16, l),
        "sfb_logical": (n, k // 16, l),
        "sfa_reordered": (32, 4, (m + 127) // 128, 4, (k + 63) // 64, l),
        "sfb_reordered": (32, 4, (n + 127) // 128, 4, (k + 63) // 64, l),
        "c": (m, n, l),
    }


for valid_k in (256, 512, 1536, 2048, 2304, 7168, 16384):
    task_storage_shapes(128, 256, valid_k)

assert (256 // 16) % 128 != 0  # the former scale-array assertion was invalid
```

Nine of the ten official correctness shapes fail the former `(K/16) % 128 == 0` assertion, including the smallest valid case with `K=256`.

## Blackwell implementation boundary

The native Blackwell instruction path supports block-of-16 NVFP4 through `tcgen05.mma...kind::mxf4nvf4.block_scale.block16`. It uses UE4M3 scale elements, which CUTLASS names `float_ue4m3_t`; UE8M0 is the MXFP4 scale type. Converting NVFP4 scales to UE8M0 would discard fractional scale values and is not a required NVFP4 preprocessing step.

CUTLASS defines `KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100`, but its direct official example is a grouped pointer-array kernel. The symbol's existence is not evidence that the contest entrants used it. CUTLASS's current official NVFP4 examples use `nv_float4_t<float_e2m1_t>`, collective mainloop and epilogue builders, `kernel::GemmUniversal`, and `device::GemmUniversalAdapter`; an old scalar-template `device::GemmUniversal` sketch is not interchangeable with that API.

TMA likewise has no universal 128-byte alignment rule for every operand. CUDA 13.2.1 generally requires a 16-byte-aligned global base and 16-byte-multiple strides for a tiled tensor map, with additional datatype-, interleave-, swizzle-, and mode-specific restrictions. Problem 2's `K % 256 == 0` rule belongs to the task ABI; it does not follow as a universal TMA theorem.

TMEM's 128-lane by 512-column organization is a storage and addressing model, not a universal 128-by-512 logical output-tile limit. Official CUTLASS NVFP4 configurations include a cooperative two-SM MMA tile with logical shape 256 by 256 by 256.

## Published performance records

The task ranks the geometric mean across three benchmark cases. It labels the following values a speed-of-light analysis based on the maximum of B200 FP4 Tensor Core math time and DRAM-memory time at a 1.5 GHz clock:

| M | N | K | L | Theoretical time (µs) |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 7168 | 16384 | 1 | 8.994 |
| 128 | 4096 | 7168 | 1 | 2.354 |
| 128 | 7168 | 2048 | 1 | 1.333 |

These are theoretical comparison rows, not measured contestant latencies and not cuBLAS results.

The public Popcorn endpoint currently gives the following dated snapshot. To make its small floating-point `submission_score` values readable beside the task's microsecond presentation, the table shows `submission_score × 10^6`; it does not relabel the rows as prize placements.

| Current rank | User | API score × 10^6 | Submission timestamp (UTC) |
| ---: | --- | ---: | --- |
| 1 | `gau.nernst` | 9.981889 | 2025-12-21 00:43:03 |
| 2 | `s.am._` | 10.060110 | 2025-12-20 17:45:21 |
| 3 | `billcarson` | 10.137411 | 2025-12-21 03:05:32 |
| 8 | `Simon` | 10.806750 | 2025-12-16 20:18:42 |
| 9 | `yue` | 10.914084 | 2025-12-11 04:36:45 |
| 10 | `currybab` | 10.930623 | 2025-12-19 08:10:18 |

Snapshot fetched August 8, 2026. Because the current first three submissions postdate the official December 19 cutoff, this endpoint alone cannot establish winners or final prize rankings. It also publishes no contestant source, CUTLASS attribution, cuBLAS comparison, raw trials, or variance.

## Applicability

Use this task contract only when the packed E2M1 payloads, logical and reordered scales, transpose convention, FP16 output, tolerances, and B200 target match the workload. “Four-bit weights” alone is insufficient: INT4, MXFP4, other scale granularities, and other layouts are not interchangeable with this ABI.

The documented native tensor-core path is Blackwell-specific; Hopper has no native FP4 tensor-core instruction. This does not exclude software conversion or emulation. Choose TMA, TMEM allocation, CUTLASS schedules, warp specialization, tile sizes, and stage counts only after validating their exact API constraints and profiling the actual shapes.

## Primary sources

- [Official contest rules](https://developer.download.nvidia.com/licenses/Blackwell-NVFP4-Hackathon-Terms-and-Conditions.pdf)
- [Pinned task definition](https://github.com/gpu-mode/reference-kernels/blob/ae67948685dfccf54ae8374dc9402addb7aae4f6/problems/nvidia/nvfp4_gemm/task.yml)
- [Pinned task types](https://github.com/gpu-mode/reference-kernels/blob/ae67948685dfccf54ae8374dc9402addb7aae4f6/problems/nvidia/nvfp4_gemm/task.py)
- [Pinned starter template](https://github.com/gpu-mode/reference-kernels/blob/ae67948685dfccf54ae8374dc9402addb7aae4f6/problems/nvidia/nvfp4_gemm/template.py)
- [Pinned correctness reference](https://github.com/gpu-mode/reference-kernels/blob/ae67948685dfccf54ae8374dc9402addb7aae4f6/problems/nvidia/nvfp4_gemm/reference.py)
- [Public Popcorn leaderboard API](https://site--bot--dxfjds728w5v.code.run/submissions/nvfp4_gemm/NVIDIA?limit=12)
- [Transformer Engine 2.13 NVFP4 recipe](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.13/user-guide/features/low_precision_training/nvfp4/nvfp4.html)
- [cuBLAS block-scaling formats](https://docs.nvidia.com/cuda/cublas/index.html#element-1d-block-scaling-for-fp8-and-fp4-data-types)
- [PTX ISA 9.0 block-scaled `tcgen05.mma`](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions-tcgen05-mma)
- [Pinned CUTLASS NVFP4 example](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu)
- [Pinned CUTLASS grouped NVFP4 example](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/examples/75_blackwell_grouped_gemm/75_blackwell_grouped_gemm_block_scaled.cu)
- [CUDA 13.2.1 tensor-map constraints](https://docs.nvidia.com/cuda/archive/13.2.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)

Query via:

```bash
conda run -n base python scripts/get_page.py kernel-nvfp4-gemm
```
