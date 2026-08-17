---
version_sensitive:
  id: vs-triton-3.6-blackwell-tcgen05
---

# Topic map / primer

Use `python3 scripts/get_page.py <id>` to open a page and `--follow-sources` to inspect its evidence. Treat performance entries as scoped source reports, never as defaults for another shape or software revision.

## Hardware

| Topic | Page | Boundary to preserve |
|---|---|---|
| `tcgen05` MMA | `hw-tcgen05-mma` | One elected thread issues; legal kinds/shapes depend on descriptors and CTA group |
| Tensor Memory | `hw-tmem` | 512 columns by 128 lanes of 32-bit cells; allocation, use, and deallocation follow PTX rules |
| CLC | `hw-clc` | Cancels an unlaunched cluster and returns its launch ID; not an arbitrary tile queue |
| TMA | `hw-tma` | Descriptor-specific asynchronous tensor copies; alignment and multicast rules are operation-specific |
| 2-SM cooperative MMA | `hw-2sm-cooperative` | `cta_group::2` requires the documented cluster/CTA protocol |
| NVFP4 | `hw-nvfp4` | E2M1 data with UE4M3 scales; do not substitute MXFP4's UE8M0 contract |
| mbarrier | `hw-mbarrier` | Phase, arrival count, transaction bytes, scope, and visibility all matter |
| PDL/GDC | `hw-pdl-gdc` | Explicit launch opt-in plus programmatic trigger/wait protocol |

## Techniques

| Problem family | Start with |
|---|---|
| Producer/consumer overlap | `technique-warp-specialization`, `technique-pipeline-stages`, `technique-double-buffering` |
| Irregular/tail work | `technique-persistent-kernels`, `technique-tile-scheduling` |
| Memory traffic/layout | `technique-vectorized-loads`, `technique-cache-policy`, `technique-swizzling` |
| Epilogue/intermediate traffic | `technique-epilogue-fusion`, `technique-kernel-fusion` |
| Low-precision scaling | `technique-fine-grained-quantization` |
| Attention exponential path | `technique-software-exp`, `technique-ping-pong-scheduling` |
| Linear-attention recurrence | `technique-chunk-parallelism` |

None of these is universal. Select a technique only after identifying the measured bottleneck and checking the cited kernel's shape, dtype, resource use, and synchronization contract.

## Kernel case studies

| Kernel | Page | Evidence note |
|---|---|---|
| FlashAttention-4 | `kernel-flash-attention-4` | Paper-reported B200 sweep; maxima are not isolated technique gains |
| DeepGEMM | `kernel-deepgemm` | Current upstream interfaces plus a historical unspecified H800 peak |
| FlashMLA / sparse MLA | `kernel-flashmla`, `kernel-sparse-mla` | README peak headlines lack complete peak-shape context |
| NVFP4 GEMM/GEMV | `kernel-nvfp4-gemm`, `kernel-nvfp4-gemv` | Public tasks define contracts; only public author reports establish participant measurements |
| Fused/grouped MoE | `kernel-fused-moe`, `kernel-grouped-gemm` | Interface and scheduling family, not one universal fusion boundary |
| Native Sparse Attention | `kernel-nsa` | A100 paper comparison against Triton FlashAttention-2 |
| Gated Delta Net | `kernel-gated-delta-net` | Algorithm/interface evidence; no retained universal speedup |

## Diagnostic pages

| Symptom | Page |
|---|---|
| Low SM utilization | `pattern-low-sm-utilization` |
| Memory bound | `pattern-memory-bound` |
| Register pressure | `pattern-register-pressure` |
| Compute bound | `pattern-compute-bound` |
| Tail effect | `pattern-tail-effect` |
| Pipeline stalls | `pattern-pipeline-stalls` |
| MoE load imbalance | `pattern-moe-load-imbalance` |

## Languages and migration

| Topic | Page |
|---|---|
| CUDA C++ | `lang-cuda-cpp` |
| CuTe DSL | `lang-cute-dsl` |
| PTX SM100 | `lang-ptx` |
| Triton 3.6+ Blackwell | `lang-triton` |
| WGMMA to `tcgen05` | `migration-wgmma-to-tcgen05` |
| Register accumulators to TMEM | `migration-register-to-tmem` |

Triton source already contained `tcgen05`/TMEM dialect operations by the 3.3 branch. Version 3.6 materially generalized and hardened the Blackwell paths and 3.7 added follow-on support. That does not imply every `tl.dot` maps to the same instruction or that Triton matches hand-written peak kernels.

## Evidence and reproducibility

- `verified`: the page carries both official-document and upstream-code evidence in `evidence_basis`.
- `source-reported`: claims are attributed to their source and retain its limitations.
- `inferred` or `experimental`: the page must state the inference or experimental boundary.
- `reproducibility: snippet` means the validator found a fenced fragment. Read the surrounding text to determine whether it is executable code, logical reference code, or explicitly labeled pseudocode; validation does not compile it.

Contest pages may have no `submissions` code because a public task contract or winner page does not guarantee a locally pinned contestant artifact.

## Audited PR corpus

These counts are the PR source pages currently present in `sources/prs/`; they
describe corpus coverage, not the number of locally captured code bundles.

| Repository | PR pages | Candidate ledger |
|---|---:|---|
| NVIDIA/cutlass | 44 | `candidates/cutlass.yaml` |
| NVIDIA/TensorRT-LLM | 139 | `candidates/tensorrt-llm.yaml` |
| NVIDIA/cccl | 66 | `candidates/cccl-cub.yaml` |
| Dao-AILab/flash-attention | 45 | `candidates/flash-attention.yaml` |
| sgl-project/sglang | 730 | `candidates/sglang.yaml` |
| vllm-project/vllm | 921 | `candidates/vllm.yaml` |
| flashinfer-ai/flashinfer | 623 | `candidates/flashinfer.yaml` |
| pytorch/pytorch | 85 | `candidates/pytorch.yaml` |
| deepseek-ai/DeepGEMM | 10 | `candidates/deepgemm.yaml` |
| tile-ai/tilelang | 29 | `candidates/tilelang.yaml` |
| **Total** | **2,692** | |
