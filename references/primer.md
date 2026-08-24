---
version_sensitive:
  id: vs-triton-3.6-blackwell-tcgen05
---

# Topic Map / Primer

A compact, authoritative map of the knowledge base. Use this as a fast lookup table when the user's question is broad — each row tells you exactly which page to open.

All page IDs below resolve via `get_page.py <id>`. All paths are relative to the wiki root (the directory containing `data/`, `wiki/`, `sources/`, `queries/`).

---

## Hardware Features (SM100)

| Feature | Page ID | Path | Notes |
|---|---|---|---|
| tcgen05 MMA instruction | `hw-tcgen05-mma` | `wiki/hardware/tcgen05-mma.md` | Blackwell tensor core instruction; replaces wgmma. CTA-scope (`cta_group::1`) or cluster-scope (`cta_group::2`). |
| Tensor Memory (TMEM) | `hw-tmem` | `wiki/hardware/tmem.md` | CTA-visible 128-lane × 512-column view of 32-bit cells; allocated in columns by a fully active warp. |
| Cluster Launch Control (CLC) | `hw-clc` | `wiki/hardware/clc.md` | A running cluster can acquire the ID of a successfully canceled, not-yet-launched cluster from its grid. |
| Tensor Memory Accelerator (TMA) | `hw-tma` | `wiki/hardware/tma.md` | Async bulk load/store (`cp.async.bulk.tensor`), multicast across cluster. |
| 2-SM Cooperative MMA | `hw-2sm-cooperative` | `wiki/hardware/2sm-cooperative.md` | `cta_group::2` — two CTAs in one cluster cooperate on a single MMA. |
| NVFP4 / block-scaled FP | `hw-nvfp4` | `wiki/hardware/nvfp4.md` | E2M1 data + per-16 FP8 scale; task prose says E4M3FNUZ, current reference code uses `torch.float8_e4m3fn`. PTX `UE4M3` is a distinct 7-bit unsigned encoding; MXFP4's `UE8M0` is a distinct block-32 format. |
| PDL / GDC | `hw-pdl-gdc` | `wiki/hardware/pdl-gdc.md` | Programmatic Dependent Launch and Grid Dependency Control — overlap successive kernel launches. |
| mbarrier primitives | `hw-mbarrier` | `wiki/hardware/mbarrier.md` | Shared-memory barriers with phase tracking; the glue between TMA/tcgen05/warps. |

---

## Optimization Techniques

| Technique | Page ID | When to use |
|---|---|---|
| Warp specialization | `technique-warp-specialization` | Separate producer/consumer roles when profiling justifies the added synchronization and resource partitioning. |
| Persistent kernels with CLC | `technique-persistent-kernels` | Reuse resident CTAs and acquire work dynamically when measured launch or tail costs justify it. |
| Ping-pong scheduling | `technique-ping-pong-scheduling` | Attention / back-to-back GEMMs; overlap two tiles by alternating SMEM buffers. |
| Epilogue fusion | `technique-epilogue-fusion` | Fuse scale/bias/activation/quantize into the same kernel using TMEM-to-register epilogue warps. |
| Software pipelining | `technique-pipeline-stages` | Multi-stage TMA→MMA overlap; choose the stage count from measured timing and resource budgets. |
| Shared memory swizzling | `technique-swizzling` | Eliminate SMEM bank conflicts on A/B tile loads. |
| Fine-grained FP8/FP4 quantization | `technique-fine-grained-quantization` | Per-tile or per-block scaling to preserve accuracy under aggressive quantization. |
| Tile scheduling | `technique-tile-scheduling` | L2 locality, cluster reordering, CLC swizzle patterns. |
| Double/multi-buffering | `technique-double-buffering` | Classical approach; overlaps load with compute, extends to 3+ stages. |
| Software-emulated exp | `technique-software-exp` | FlashAttention-4 distributes exponential work across MUFU and an FMA approximation. |
| Register budgeting | `technique-register-budgeting` | Raise occupancy when MMA warp counts are tight. |
| Cache policy differentiation | `technique-cache-policy` | PTX `evict_first` / `no_allocate` on streaming data. |
| Chunk-based parallelism | `technique-chunk-parallelism` | Linear-attention kernels (GatedDeltaNet, Mamba). |
| Kernel fusion | `technique-kernel-fusion` | Combine elementwise + GEMM + reduction into a single launch. |
| Wide vectorized loads | `technique-vectorized-loads` | Saturate HBM bandwidth; cp.async or 128-bit LDG. |

---

## Kernel Case Studies

| Kernel | Page ID | Headline perf | Key techniques |
|---|---|---|---|
| FlashAttention-4 | `kernel-flash-attention-4` | Up to 1613 TFLOPS B200 BF16 (71%) in the paper's sweep | Ping-pong scheduling, software exp, 2-CTA backward |
| DeepGEMM (FP8) | `kernel-deepgemm` | ~1550 TFLOPS H800 FP8 | Fine-grained scaling, CUDA-core promotion (Nc=128) |
| NVFP4 GEMM | `kernel-nvfp4-gemm` | — | tcgen05 + per-16 FP8 scales (task prose: E4M3FNUZ; reference code: `torch.float8_e4m3fn`); PTX `UE4M3` is distinct, as is MXFP4's block-32 `UE8M0` |
| NVFP4 batched GEMV | `kernel-nvfp4-gemv` | Author reports a 22.392 µs final aggregate after a multi-step tuning sequence | Memory access, PTX, cache policy, instruction-level parallelism |
| FP8 block-scale GEMM | `kernel-fp8-block-scale-gemm` | — | 1×128 / 128×128 block scaling scheme |
| Fused MoE | `kernel-fused-moe` | — | Gate-up fused with SwiGLU; FP8 block scale routing |
| Gated Dual GEMM | `kernel-gated-dual-gemm` | — | Gate × Up → SiLU fused in epilogue |
| Grouped GEMM for MoE | `kernel-grouped-gemm` | — | Variable-sized expert GEMMs in one launch |
| FlashMLA | `kernel-flashmla` | DeepSeek V3 decode | MLA-specific TMA + tcgen05 layout |
| Sparse MLA | `kernel-sparse-mla` | DeepSeek V3.2 | Sparse KV retrieval before MLA core |
| Native Sparse Attention (NSA) | `kernel-nsa` | 9× fwd speedup | Block-sparse + compressed attention |
| Gated Delta Net | `kernel-gated-delta-net` | — | Chunk parallelism; linear attention |

---

## Problem → Pattern (Diagnosis)

| Symptom | Pattern page | Candidate techniques |
|---|---|---|
| Low SM utilization | `pattern-low-sm-utilization` | Persistent kernels, CLC, tile scheduling |
| Memory bandwidth bound | `pattern-memory-bound` | Vectorized loads, cache policy, register budgeting |
| Register pressure / low occupancy | `pattern-register-pressure` | Register budgeting, TMEM offload, kernel split |
| Not reaching peak FLOPS | `pattern-compute-bound` | Warp specialization, epilogue fusion, pipeline stages |
| Last wave underutilizes GPU | `pattern-tail-effect` | CLC, persistent kernels, tile-M/N aspect tuning |
| Pipeline stalls | `pattern-pipeline-stalls` | Deeper pipeline stages, mbarrier phase management |
| MoE expert load imbalance | `pattern-moe-load-imbalance` | Grouped GEMM, dynamic routing, overflow spill |

---

## Languages / DSLs

| DSL | Page ID | Notes |
|---|---|---|
| CuTe DSL | `lang-cute-dsl` | Preferred high-level path on SM100; native tcgen05/TMEM/CLC bindings. |
| CUDA C++ | `lang-cuda-cpp` | PTX inline is common; used by CUTLASS, vLLM, SGLang custom kernels. |
| PTX (SM100) | `lang-ptx` | `tcgen05.*`, `clusterlaunchcontrol.*`, `cp.async.bulk.tensor.*` — low-level control. |
| Triton | `lang-triton` | On Blackwell: Triton 3.6+ ships native tcgen05 + TMEM lowering through descriptor/TMA + `tl.range(warp_specialize=True)`, `tl.dot_scaled`, and Gluon multi-CTA / 2CTA. Pre-3.6 the framing was "no tcgen05/TMEM exposure"; that historical context is preserved on the page. Cite via `version_sensitive: vs-triton-3.6-blackwell-tcgen05`. |

---

## Migration (Hopper → Blackwell)

| Page ID | Notes |
|---|---|
| `migration-wgmma-to-tcgen05` | wgmma (Hopper) → tcgen05 (Blackwell); CTA-scope vs cluster-scope MMA; descriptor changes. |
| `migration-register-to-tmem` | Register accumulator (Hopper) → TMEM (Blackwell); epilogue restructuring required. |

---

## Source Repositories (PR coverage)

| Repo | PR pages | Ledger |
|---|---:|---|
| NVIDIA/cutlass | 36 | `candidates/cutlass.yaml` |
| sgl-project/sglang | 169 | `candidates/sglang.yaml` |
| vllm-project/vllm | 205 | `candidates/vllm.yaml` |
| flashinfer-ai/flashinfer | 322 | `candidates/flashinfer.yaml` |
| pytorch/pytorch | 4 | `candidates/pytorch.yaml` |
| deepseek-ai/DeepGEMM | 8 | `candidates/deepgemm.yaml` |
| NVIDIA/cccl | 60 | `candidates/cccl-cub.yaml` |
| Dao-AILab/flash-attention | 33 | `candidates/flash-attention.yaml` |
| NVIDIA/TensorRT-LLM | 88 | `candidates/tensorrt-llm.yaml` |
| tile-ai/tilelang | 19 | `candidates/tilelang.yaml` |

Query by repo: `python3 scripts/query.py --repo <name>`.

---

## Competitions

| Contest | Path | Problems/Tracks |
|---|---|---|
| GPU Mode NVFP4 Hackathon | `sources/contests/gpu-mode-nvfp4/` | `problem-1-gemv`, `problem-2-gemm`, `problem-3-gated-dual-gemm`, `problem-4-grouped-gemm` |
| FlashInfer MLSys 2026 | `sources/contests/flashinfer-mlsys26/` | `track-a-fused-moe`, `track-b-sparse-attention`, `track-c-gated-delta-net` |

Each contest page records the organizer's task definition, speed-of-light data when published, and a dated leaderboard or results snapshot. Reconstructed per-submission rankings were removed; no page declares a `submissions:` block.

---

## Canonical Aliases (normalization cheat sheet)

When the user types one of these, match to the canonical term shown:

| User-typed | Canonical |
|---|---|
| UMMA, tensor_core_gen05, tcgen05.mma | `tcgen05` |
| tensor memory, TMEM | `tmem` |
| Cluster Launch Control, CLC | `clc` |
| Tensor Memory Accelerator, TMA, cp.async.bulk | `tma` |
| 2-SM cooperative, two-SM cooperative, 2CTA, cta_group::2 | `2sm-cooperative` |
| NVFP4, nv_float4, E2M1, FP4 E2M1 | `nvfp4` |
| block scaling, UE8M0, microscaling, MX | `block-scale` |
| Blackwell | `blackwell` family |
| B200, GB200, SM100 | `sm100` |
| B300, GB300, SM103 | `sm103` |
| Hopper, H100, H200, H800, SM90 | `sm90` |
| MoE, mixture of experts, expert parallelism | `moe` |
| MLA, multi-head latent attention | `mla` |
| GDN, GatedDeltaNet, gated delta rule | `gated-delta-net` |
| NSA, native sparse attention | `sparse-attention` |

The `query.py` tool applies these automatically when scoring and when using `--tag`.

---

## Confidence & Evidence Quick Reference

- **`verified`** — page carries an `evidence_basis` list with at least one `official-doc` AND one `upstream-code` entry. Safe to quote as authoritative.
- **`source-reported`** — claims come from ≥ 1 authoritative source (paper, official blog, major repo). Most wiki pages live here.
- **`inferred`** — synthesized from multiple sources; flag when quoting.
- **`experimental`** — undocumented / version-sensitive / PTX tricks. Always caveat.

Minimum reproducibility for technique, kernel, and language pages: **`snippet`** (compilable fragment) — validator-enforced.
