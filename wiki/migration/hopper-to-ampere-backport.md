---
id: migration-hopper-to-ampere
title: "Backporting Hopper/Blackwell Kernels to Ampere (sm_80/sm_86)"
type: migration
from_arch: sm90
to_arch: sm86
tags: [cp-async, mma-sync, tma, wgmma, pipeline-stages, warp-specialization]
related: [hw-cp-async, hw-mma-sync-ampere, hw-ampere-memory-model, hw-tma, hw-tcgen05-mma, hw-mbarrier, technique-warp-specialization, technique-pipeline-stages, technique-double-buffering, technique-register-budgeting, technique-tile-scheduling]
sources: [doc-ptx-isa-ampere, doc-ampere-tuning-guide, doc-ga102-whitepaper, doc-cutlass-ampere]
blackwell_relevance: "Reverse-direction guide: maps every Hopper/Blackwell-only feature referenced across this wiki to its sm_8x replacement, so Blackwell-first pages remain actionable on Ampere hardware (A100, RTX 3090-class rigs)."
confidence: source-reported
reproducibility: pseudocode
---

# Backporting Hopper/Blackwell Kernels to Ampere (sm_80/sm_86)

## Overview

This wiki is Blackwell-first, but a large fleet of real hardware is Ampere — A100s in clouds and GA10x (RTX 3090/4-GPU rigs, A6000, A40) on desks. This page is the systematic reverse map: given a kernel design expressed in Hopper/Blackwell idioms (TMA + mbarrier + warp specialization + wgmma/tcgen05), produce the equivalent sm_8x design without rediscovering Ampere lore.

A backport is three separate problems, in order:

1. **Instructions** — mechanical: every sm_90+/sm_100 instruction has a defined replacement or workaround (table below). ptxas failures enumerate this list for you.
2. **Capacity** — arithmetic: SMEM 99 KB/block (sm_86), 1536 threads/SM, small L2, register-resident accumulators. Tile/stage configs must be re-derived, not copied.
3. **Scheduling paradigm** — conceptual: async-everything producer/consumer designs collapse back to synchronous multi-stage pipelines where *all* warps compute. Blindly keeping warp specialization usually loses.

## 1. Instruction replacement table

| Hopper/Blackwell construct | sm_8x replacement | Notes |
|---|---|---|
| `cp.async.bulk.tensor` (TMA load) | `cp.async.cg` per-thread 16 B copies | Address math + tile iteration in-kernel; see hw-cp-async |
| TMA OOB clamping | `cp.async` `src-size` zero-fill per copy | Clamp per 16 B transaction, not per tensor |
| TMA swizzle-on-store (32/64/128 B) | XOR-swizzle in destination index math | Same swizzle functions, applied manually |
| TMA multicast (cluster) | None — each CTA loads its own tile | L2 absorbs most of the duplicate traffic; schedule tiles for L2 reuse |
| `cp.async.bulk` SMEM→GMEM store | Plain `st.global` epilogue | No async store path |
| `wgmma.mma_async` (sm_90) | `mma.sync.m16n8k16` + `ldmatrix` | Warp-scope, synchronous, operands via registers; see hw-mma-sync-ampere |
| `tcgen05.mma` + TMEM (sm_100) | Same as above; accumulators live in registers | Register budget becomes the binding constraint |
| `stmatrix` | `st.shared` + layout shuffle in epilogue | Costs SMEM roundtrip + cycles |
| `mbarrier.arrive.expect_tx` (byte counting) | Commit-group counting (`cp.async.wait_group N`) or mbarrier arrival counting | Count *arrivals*, not bytes; `cuda::barrier` works on sm_80 |
| `setmaxnreg` (register realloc) | None — static `__launch_bounds__` / `-maxrregcount` | All warps keep equal register budgets |
| Thread block clusters / DSMEM | None — single-CTA designs + L2 | Cross-CTA sharing only via L2/global atomics |
| CLC (cluster launch control) | Persistent kernels + global atomic work queue | technique-persistent-kernels works fine on Ampere |
| PDL / GDC (launch overlap) | Streams, CUDA Graphs, kernel fusion | No intra-launch overlap primitive |
| FP8 (E4M3/E5M2) tensor ops | INT8 (`mma.sync` s8→s32, 284 TOPS on 3090) or FP16 | No FP8 tensor cores on Ampere; Marlin-style weight-only INT4/INT8 is the idiomatic substitute |
| NVFP4 / block-scaled MMA | Dequant-to-FP16 in-kernel + FP16 MMA | Scales applied in registers before/after MMA |
| TMEM double-buffering | Register double-buffering is impossible at Blackwell tile sizes — shrink the warp tile | 64×64/warp is the practical ceiling |

## 2. Capacity re-planning (sm_86 numbers)

Re-derive, don't copy, using the hw-ampere-memory-model card:

- **Pipeline depth:** SMEM/block ceiling is 99 KB. A bf16 128×64 A-tile + 64×128 B-tile stage = 32 KB → 3 stages fit (96 KB); a Hopper 5-stage 200 KB config does not. Formula: `stages ≤ 99KB / stage_bytes`, keep ≥2.
- **Occupancy:** target divisors of 1536 (3×512, 6×256, 12×128). Hopper 2×1024 configs strand 25%.
- **Registers:** M×N warp-tile accumulator costs `M*N/32` regs/thread (64×64 f32 = 128). Budget: accums + A/B fragments + addresses ≤ 255, ideally ≤ 168 for 3-block residency. Watch `LDL/STL` in SASS — any spill in the mainloop is disqualifying.
- **L2 (GA10x only, 6 MB):** output-tile rasterization order (technique-tile-scheduling) decides operand re-reads; grid-stride swizzled tile order that keeps a K-panel resident is worth more on 3090 than on A100/H100.
- **GA10x FP32-accumulate half-rate:** ceiling for bf16 GEMM on 3090 is 71 TFLOPS, not the 142 the FP16-acc spec suggests. Set expectations (and roofline math) accordingly before profiling.

## 3. Scheduling paradigm: async producer/consumer → synchronous multistage

Hopper/Blackwell designs split warps into producers (issue TMA) and consumers (issue wgmma/tcgen05, which are *asynchronous*), with mbarrier handoff and `setmaxnreg` shifting registers to consumers. On Ampere all three enablers are missing:

1. `mma.sync` **stalls the issuing warp** — a "consumer" warp can't overlap its own MMAs with anything;
2. accumulators are **pinned to the warp's registers** — you can't hand a tile's accumulation to another warp without an SMEM roundtrip;
3. no register reallocation — idle producer warps waste a full register share.

The idiomatic Ampere shape (CUTLASS `MmaMultistage`, FlashAttention-2) is therefore: **every warp both loads and computes**. Each iteration: issue `cp.async` for stage `t+STAGES-1`, `wait_group` for stage `t`, `__syncthreads()`, run the warp-tile `ldmatrix`+`mma.sync` schedule, repeat. Latency is hidden by (a) 2–4 SMEM stages in flight and (b) other resident warps, not by intra-CTA role split.

Warp specialization on Ampere is justified only when the "producer" work is *not* plain operand streaming — e.g., decompression/dequant of weights, sparse index gather, or top-k selection feeding a dense compute stage. Even then, measure against the multistage baseline first.

## 4. Worked example: attention mainloop, both dialects

Hopper/Blackwell (as written across this wiki):

```text
producer warps:                     consumer warpgroups:
  tma.load K[i] -> smem, mbar.expect_tx     wait mbar(K[i])
  tma.load V[i] -> smem, mbar.expect_tx     S = wgmma(Q, K[i])        # async, regs
                                            softmax_update(S)          # overlap with next wgmma
                                            O += wgmma(P, V[i])
```

Ampere backport (FlashAttention-2 shape):

```text
all warps, per iteration i:
  cp.async K[i+2] -> smem_stage[(i+2)%3]; commit_group      # prefetch 2 ahead
  cp.async.wait_group(1); __syncthreads()                    # K[i] resident
  S_frag = mma.sync(Q_frag, ldmatrix(K[i]))                  # warp stalls per-issue;
  m,l update; P_frag = exp(S_frag - m)                       #   hidden by other warps
  O_frag += mma.sync(P_frag, ldmatrix(V[i]))                 # f16 P·V may use f16 acc (GA10x full rate)
  __syncthreads()                                            # stage consumed
```

Deltas to notice: prefetch distance is explicit (`STAGES-1`); softmax can't overlap MMA within a warp (rely on occupancy); Q stays in registers both ways; the f16-accumulate escape hatch for P·V is a GA10x-specific win worth its numerics review.

## 5. Backport verification checklist (ncu)

After a port, confirm the Ampere idiom actually engaged:

- `sm__inst_executed_pipe_tensor` active — tensor pipe busy % is the primary KPI; on 3090 bf16, 60–75% of the 71 TFLOPS ceiling is a strong result.
- `smsp__sass_inst_executed_op_ldsm` present (ldmatrix in use), zero `LDL/STL` in the mainloop (no spills).
- `l1tex__data_bank_conflicts_pipe_lsu` ≈ 0 — swizzle survived the port.
- `sm__warps_active.avg.pct_of_peak` ≥ ~50% — occupancy is your latency-hiding now; a Hopper-style 1-block config that was fine with async engines will crater here.
- Long-scoreboard stall % dominant → raise stages/occupancy; `selected`/`math pipe throttle` dominant → you're compute-bound, stop tuning loads.

## Reference backport in the wild

FlashAttention-2 (sm_80-era) vs FlashAttention-3/4 (sm_90/sm_100) is the canonical public before/after of this exact migration, in reverse: same algorithm, the FA3→FA2 deltas are precisely this page's tables. When porting an attention-family kernel, diff your plan against FA2's choices first.
