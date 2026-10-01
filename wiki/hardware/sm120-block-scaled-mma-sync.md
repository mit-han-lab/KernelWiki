---
id: hw-sm120-block-scaled-mma-sync
title: "SM120 Block-Scaled mma.sync (mxf4nvf4 / mxf8f6f4) Operand Layouts"
type: hardware
architectures: [sm120]
tags: [block-scale, nvfp4, fp4, fp8, ptx]
confidence: experimental
related: [hw-nvfp4, hw-tcgen05-mma, technique-fine-grained-quantization, kernel-nvfp4-gemm]
sources: [doc-ptx-isa-sm100]
aliases: ["mma.sync kind::mxf4nvf4", "mma.sync kind::mxf8f6f4", "scale_vec::4X", "scale_vec::1X", "SM120 block scale", "RTX PRO 6000 FP4"]
artifact_dir: artifacts/kernels/sm120-block-scaled-mma-sync
---

## Overview

SM120 (RTX PRO 6000 Blackwell, consumer Blackwell) has no `tcgen05`, no TMEM and no CLC. Block-scaled
FP4 / FP8 tensor-core math is reached through the warp-level `mma.sync` with `.kind::mxf4nvf4` or
`.kind::mxf8f6f4` and `.block_scale`: register operands, 32 lanes, fp32 accumulators in registers. TMA
(`cp.async.bulk.tensor`, `cuTensorMapEncodeTiled`) and `mbarrier` work, but the MMA itself never reads
shared memory.

The PTX ISA states the scale-selector semantics (`thread-id`, `byte-id`) tersely, and the two layouts below
were established with a lane-poisoning probe: lanes a hypothesis says are *not* read hold poisoned scales,
so a PASS proves the selector, not only the data path. Probe source and complete output are in
[`artifacts/kernels/sm120-block-scaled-mma-sync/variants/`](../../artifacts/kernels/sm120-block-scaled-mma-sync/variants/). Measured with CUDA 13.0 (`nvcc -gencode arch=compute_120a,code=sm_120a`; plain `sm_120`
rejects `.kind::mxf4nvf4`) on an RTX PRO 6000 Blackwell, 2026-10-01.

Notation: `L` = lane id, `g = L / 4` (0..7), `q = L % 4`. `.row.col`: A is row-major M x K, B is `B[n][k]`.
C/D fragments are the usual m16n8: `d0,d1` -> row `g`, cols `2q, 2q+1`; `d2,d3` -> row `g+8`.

## `m16n8k64 .kind::mxf4nvf4 .scale_vec::4X`

e2m1 x e2m1 -> f32, ue4m3 scales, one scale per 16 k (four per k-step of 64).

```ptx
mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3
    {d0,d1,d2,d3}, {a0,a1,a2,a3}, {b0,b1}, {c0,c1,c2,c3}, {sa}, {0, thread-id-a}, {sb}, {0, thread-id-b};
```

| operand | register | holds |
|---|---|---|
| A | `a0` | row `g`, k `q*8 .. q*8+7` (8 nibbles, low nibble = lower k) |
| A | `a1` | row `g+8`, same k |
| A | `a2` | row `g`, k `32+q*8 ..` |
| A | `a3` | row `g+8`, k `32+q*8 ..` |
| B | `b0` | col `g`, k `q*8 ..`; `b1`: k `32+q*8 ..` |
| scale A | `sa` | bytes 0..3 = k-groups 0..3 of row `g + 8*(q&1)`; **read only from lanes with `(q>>1) == thread-id-a`** |
| scale B | `sb` | bytes 0..3 = k-groups 0..3 of col `g`; **read only from lanes with `q == thread-id-b`** |

`byte-id` must be 0 for 4X. Two consequences a register-only kernel can exploit: lanes `q in {0,1}` serve
m16 tile 0 and `q in {2,3}` tile 1 of a 32-row warp slab, selected by `thread-id-a = tile`, so one 32-bit
scale load per lane per k-step covers a 32-row slab; and one `sb` register per lane can hold the scales of
four different n8 tiles (one per `q`), selected with `thread-id-b = tile % 4`.

## `m16n8k32 .kind::mxf8f6f4 .scale_vec::1X`

e4m3 x e4m3 -> f32, ue8m0 scales, one per 32 k (MXFP8).

```ptx
mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0
    {d0..d3}, {a0..a3}, {b0,b1}, {c0..c3}, {sa}, {byte-id-a, thread-id-a}, {sb}, {byte-id-b, thread-id-b};
```

| operand | register | holds |
|---|---|---|
| A | `a0` | row `g`, k `q*4 .. q*4+3` (4 bytes); `a1`: row `g+8` |
| A | `a2` | row `g`, k `16+q*4 ..`; `a3`: row `g+8` |
| B | `b0` | col `g`, k `q*4 ..`; `b1`: k `16+q*4 ..` |
| scale A | `sa` | byte `byte-id-a` = scale of row `g + 8*(q&1)`; read only from lanes with `(q>>1) == thread-id-a` |
| scale B | `sb` | byte `byte-id-b` = scale of col `g`; read only from lanes with `q == thread-id-b` |

Same lane -> row/col map as 4X; `byte-id` (0..3) picks the byte, so one register carries four consecutive
k-steps' scales for a row, or the scales of four n8 tiles for B (`byte-id-b = tile % 4`). The k-step is
32 bytes for both instructions (64 x 4 bit == 32 x 8 bit): an implicit-GEMM A/B loader written for one is
byte-for-byte reusable for the other.

## Probe (excerpt)

```cuda
// lanes the hypothesis says are not read hold poison; PASS => selector semantics proven
uint32_t sa = 0xFEFEFEFEu, sb = 0xFEFEFEFEu;              // ue8m0 0xFE = 2^127
if ((q >> 1) == tid_a) sa = scales_of_row(g + 8 * (q & 1)); // 4 bytes = 4 k-steps, byte-id-a selects
if (q == tid_b)        sb = scales_of_col(g);
asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0,0}, {%11}, {0,0};"
             : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
```

All 8 (4X) and 16 (1X, with and without B-lane poisoning) selector combinations pass against a CPU reference
(`maxrel <= 1.2e-7`); see `sm120_block_scale_probe.out`.

## Throughput reference points (same GPU)

| kernel | TFLOP/s | notes |
|---|---|---|
| nunchaku `gemm_w4a4_fp4_kernel`, M=65536 K=6912 N=1024 | 1356 | 250 regs, 8 warps/SM, register double-buffer, no smem |
| NVFP4 implicit-GEMM conv3d (register-only, A 3-stage) | 1340-1436 | 254 regs, 117 SASS instructions per warp k-step |
| same conv, TMA + smem ring, 2 blocks/SM | 1090 | parity with the register-only kernel it replaced |
| MXFP8 implicit-GEMM conv3d, first candidate | 617-653 | 255 regs, unprofiled |
| cuDNN fp8 graph conv3d (`conv_fprop`, pre/post padding) | 340-610 | |
| cuDNN fp16 channels_last conv3d | 190-290 | |

Two practical notes from those kernels. Every block-scaled `OMMA` is followed by a `NOP` in SASS (also in
nunchaku's kernel): budget two issue slots per MMA. With ~250 registers a block is 8 warps = 2 warps per
SMSP, and that occupancy reaches 85% tensor-pipe active in nunchaku's GEMM, so "2 warps per SMSP cannot feed
the pipe" is not a valid diagnosis on SM120; compare the instruction count per k-step against such a
reference kernel before theorising about the pipe.
