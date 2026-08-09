---
id: kernel-deepgemm
title: DeepGEMM — FP8 GEMM with Fine-Grained Scaling
type: kernel
architectures:
- sm100
- sm90
tags:
- gemm
- fp8
- fine-grained-quantization
- block-scale
confidence: source-reported
reproducibility: snippet
kernel_types:
- gemm
- grouped-gemm
languages:
- cuda-cpp
- ptx
related:
- technique-fine-grained-quantization
- hw-tcgen05-mma
- hw-nvfp4
sources:
- blog-deepgemm
- pr-deepgemm-304
performance_claims: []
blackwell_relevance: The pinned SM100 kernel uses block-scaled UMMA/tcgen05 with
  packed UE8M0 scale factors in TMEM; the pinned SM90 kernel provides the
  WGMMA plus CUDA-core-promotion comparison.
artifact_dir: artifacts/kernels/deepgemm
---

# DeepGEMM -- FP8 GEMM with Fine-Grained Scaling

## Verified Scope

DeepGEMM is DeepSeek's open-source tensor-core kernel library. This page describes the FP8 GEMM paths at commit [`891d57b4db1071624b5c8fa0d1e51cb317fa709f`](https://github.com/deepseek-ai/DeepGEMM/tree/891d57b4db1071624b5c8fa0d1e51cb317fa709f), with byte-verified SM90 and SM100 kernel files stored locally. The pinned project requires an SM90 or SM100 GPU, CUDA 12.3 or newer for SM90, and CUDA 12.9 or newer for SM100.

DeepGEMM contains multiple kernels and APIs; the local SM90 and SM100 files below are representative FP8 1D1D implementations, not the whole library.

## Fine-Grained Quantization

The [DeepSeek-V3 Technical Report v2](https://arxiv.org/abs/2412.19437v2) defines the training scheme that motivates this path:

- activations are grouped per token and per 128 channels, forming `1 x 128` tiles;
- weights are grouped per 128 input channels and 128 output channels, forming `128 x 128` blocks; and
- smaller groups let scales adapt more locally, which the paper says better accommodates outliers. This is a scoped accuracy motivation, not a guarantee that quantization error disappears.

Scale representation is architecture-specific in the pinned DeepGEMM interface. SM90 consumes FP32 scale factors. SM100 consumes packed UE8M0 factors, four UE8M0 values per `torch.int`. The pinned SM90 1D1D kernel fixes `BLOCK_K == 128`; the SM100 1D1D template accepts K scale granularities of 32 or 128 for each operand.

## SM90: WGMMA and CUDA-Core Promotion

The DeepSeek-V3 report characterizes H800 FP8 Tensor Core accumulation as retaining about 14 bits. Its `Nc=128` strategy accumulates 128 elements of the GEMM inner dimension—four WGMMAs in the described configuration—before moving the partial result to FP32 registers on CUDA cores.

The pinned [`sm90_fp8_gemm_1d1d.cuh`](../../artifacts/kernels/deepgemm/full/sm90_fp8_gemm_1d1d.cuh) implements that structure directly:

1. A math warp-group owns `float accum[...]` for WGMMA output and zero-initialized `float final_accum[...]` for promoted results.
2. For each 128-element K block, it reads the A and B FP32 factors from shared memory, issues `BLOCK_K / WGMMA::K` WGMMA operations, commits the group, and waits for completion.
3. The `Promote with scales` loop multiplies each partial result by its A and B factors and adds it to `final_accum`.
4. After all K blocks, the kernel stages the final FP32 values for the epilogue/store path.

The exact upstream file is the reproducible reference. It should not be replaced by a sketch using half accumulators or by deriving the promotion interval from WGMMA's output-N tile.

## SM100: Native Block-Scaled UMMA

The pinned [`sm100_fp8_gemm_1d1d.cuh`](../../artifacts/kernels/deepgemm/full/sm100_fp8_gemm_1d1d.cuh) uses a different data path:

1. It creates a block-scaled UMMA instruction descriptor with a `float` accumulator type and `cutlass::float_ue8m0_t` scale type.
2. TMA loads packed scale-factor data into shared memory. UTCCP copies selected scale blocks from shared memory into dedicated SFA and SFB columns in TMEM.
3. The elected issuing thread calls the SM100 UMMA wrapper with shared-memory operand descriptors, the TMEM accumulator column, a runtime instruction descriptor containing scale IDs, and the two TMEM scale addresses.
4. The epilogue drains the TMEM accumulator after the kernel's full/empty barrier protocol says it is ready.

This path does not contain the SM90 `final_accum` CUDA-core promotion loop. That difference does not justify a blanket claim that every tcgen05 accumulator mode or datatype has "full FP32" behavior; the accumulator type and instruction form remain part of the selected operation.

## Grouped GEMM Interfaces

The pinned interface distinguishes three workload arrangements rather than treating all of them as M-varying layouts:

| Interface family | Varying group dimension | Fixed dimensions | Work metadata |
|---|---:|---:|---|
| M-grouped contiguous | M | N and K | Either a group index for each packed M row or a prefix-sum M layout, depending on the selected option |
| M-grouped masked | Valid M within each `[G, M, K]` allocation | Maximum M, N, and K tensor extents | An integer `masked_m[G]` vector holding each group's valid M length, not a binary mask |
| K-grouped contiguous | K | M and N | Per-group K lengths plus their device tensor; used for weight-gradient-style grouped GEMM |

Contiguous M grouping is intended for variable per-expert token counts in training forward or inference prefill. Masked M grouping keeps fixed allocations suitable for CUDA-graph decode while limiting work to each group's valid M. The pinned repository provides K-grouped NT on SM90 and K-grouped TN on SM100 for the documented FP8 paths.

## JIT Compilation

DeepGEMM generates and compiles kernel source at runtime. At the pinned commit, the compiler defaults to NVCC. Setting `DG_JIT_USE_NVRTC=1` selects the optional NVRTC path; the project warns that this may reduce performance for some cases.

The cache key includes the kernel name, compiler signature, compiler flags, and generated source. A cache hit reuses the existing kernel runtime; a miss compiles a CUBIN in a temporary directory and then publishes the completed cache entry. Consequently, a new specialization has first-use compilation/loading cost, whereas matching later calls can reuse the artifact. This mechanism does not guarantee globally optimal register allocation or unrolling.

## Operand Layouts

For the pinned FP8 interface, SM90 supports NT only: A is non-transposed and B is supplied in the representation used for `A @ B.T`. SM100 exposes dense `fp8_gemm_{nt, nn, tn, tt}` variants. Its implementation propagates the selected operand-major modes into the UMMA descriptors; the public wrappers transpose the corresponding tensor views for the non-NT variants.

## Performance Evidence

The pinned README reports that DeepGEMM reached **up to 1550 TFLOPS on H800** in an April 2025 news item. It does not bind that peak to `M=N=K=4096`, state approximately 90% utilization, or preserve enough benchmark conditions for reproduction. The unsupported structured performance record and table have therefore been removed; the surviving number is source-reported only and should not be compared across software, clocks, shapes, or GPUs without a controlled rerun.

## Practical Boundaries

- Input transposition, FP8 casting, and scale-layout preparation are separate from the optimized GEMM kernels; the project supplies utilities but warns they may be slower than fusing the work into producers.
- M-grouped contiguous segments must satisfy the configured M/K alignment. Masked mode uses valid-length integers, while K-grouped mode has different shapes and architecture-specific layout variants.
- JIT cache misses add compilation latency, and NVRTC is optional rather than the default.
- Fine-grained scaling's accuracy and cost tradeoffs depend on quantization recipe, target architecture, scale preparation, fusion, and workload. There is no universal per-tensor-scaling penalty or single "use only for outliers" rule.

## Pinned Sources

- [DeepGEMM README at commit `891d57b`](https://github.com/deepseek-ai/DeepGEMM/blob/891d57b4db1071624b5c8fa0d1e51cb317fa709f/README.md)
- [DeepGEMM GEMM API at commit `891d57b`](https://github.com/deepseek-ai/DeepGEMM/blob/891d57b4db1071624b5c8fa0d1e51cb317fa709f/csrc/apis/gemm.hpp)
- [DeepGEMM JIT compiler at commit `891d57b`](https://github.com/deepseek-ai/DeepGEMM/blob/891d57b4db1071624b5c8fa0d1e51cb317fa709f/csrc/jit/compiler.hpp)
- [DeepSeek-V3 Technical Report v2](https://arxiv.org/abs/2412.19437v2)

## Full Reference Implementation

Local verbatim upstream code lives in [`artifacts/kernels/deepgemm/full/`](../../artifacts/kernels/deepgemm/full/) and is pinned by [`PROVENANCE.yaml`](../../artifacts/kernels/deepgemm/full/PROVENANCE.yaml) to commit `891d57b4db1071624b5c8fa0d1e51cb317fa709f`. The recorded SHA-256 values match both files. Labeled derived material lives in [`artifacts/kernels/deepgemm/variants/`](../../artifacts/kernels/deepgemm/variants/); it is teaching material, not upstream code.

Query the page and its attached code with:

```bash
python3 scripts/get_page.py kernel-deepgemm --include-code
```

The following is a verbatim fragment of the SM90 promotion loop; the linked full file supplies its surrounding declarations and loop bounds:

```cpp
const float &scale_b_0 = scales_b[i].x;
const float &scale_b_1 = scales_b[i].y;
final_accum[i * 4 + 0] += scale_a_0 * scale_b_0 * accum[i * 4 + 0];
final_accum[i * 4 + 1] += scale_a_0 * scale_b_1 * accum[i * 4 + 1];
final_accum[i * 4 + 2] += scale_a_1 * scale_b_0 * accum[i * 4 + 2];
final_accum[i * 4 + 3] += scale_a_1 * scale_b_1 * accum[i * 4 + 3];
```
