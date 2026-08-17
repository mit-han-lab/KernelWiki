---
id: contest-flashinfer-track-b
title: 'FlashInfer MLSys 2026 - Track B: Sparse Attention'
source_category: contest-report
architectures: [sm100]
tags: [sparse-attention, mla, fp8, block-scale, decode]
techniques: [kernel-fusion, fine-grained-quantization]
hardware_features: [fp8, block-scale]
kernel_types: [sparse-attention, mla, attention, decode]
url: https://github.com/flashinfer-ai/mlsys26-contest/blob/a523107f1ff793871a04bd535f921a0a4bd44173/index.html
submissions:
  - rank: agent-assisted 1
    participant: Dogacel
    submission_truth: unavailable
    code_unavailable_reason: The official page links a public submission repository, but this repository has not pinned it into a local contest-submission bundle.
  - rank: agent-assisted 2
    participant: Cong
    submission_truth: unavailable
    code_unavailable_reason: The official page links a public submission repository, but this repository has not pinned it into a local contest-submission bundle.
  - rank: agent-assisted 3
    participant: Team Wombat
    submission_truth: unavailable
    code_unavailable_reason: The official page links public indexer and attention repositories, but this repository has not pinned them into a local contest-submission bundle.
  - rank: full-agent 1
    participant: Dogacel
    submission_truth: unavailable
    code_unavailable_reason: The official page links a public submission repository, but this repository has not pinned it into a local contest-submission bundle.
  - rank: full-agent 2
    participant: HAN Lab Kernel Mafia
    submission_truth: unavailable
    code_unavailable_reason: The official page links a public submission repository, but this repository has not pinned it into a local contest-submission bundle.
  - rank: full-agent 3
    participant: UW SyFI
    submission_truth: unavailable
    code_unavailable_reason: The official page links a public submission repository, but this repository has not pinned it into a local contest-submission bundle.
---

# Track B: Sparse Attention

The official contest page describes DeepSeek Sparse Attention on NVIDIA B200 and links two benchmark definitions:

- indexer: `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`;
- sparse attention: `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`.

These identifiers establish the contest workload boundary. They do not by themselves establish a particular internal pipeline or a performance result.

## Official placements

| Approach | Place | Team | Public repository linked by organizer |
|---|---:|---|---|
| Agent-assisted | 1 | Dogacel | [Dogacel/DeepSeek-Sparse-Attention-Kernels](https://github.com/Dogacel/DeepSeek-Sparse-Attention-Kernels) |
| Agent-assisted | 2 | Cong | [luongthecong123/learn-cutedsl](https://github.com/luongthecong123/learn-cutedsl/tree/main/fused_kernel) |
| Agent-assisted | 3 | Team Wombat | [indexer](https://github.com/ykirpichev/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64) and [attention](https://github.com/ykirpichev/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64) |
| Full-agent | 1 | Dogacel | [Dogacel/auto-gpu-kernel](https://github.com/Dogacel/auto-gpu-kernel) |
| Full-agent | 2 | HAN Lab Kernel Mafia | [mit-han-lab/mlsys2026-flashinfer-contest](https://github.com/mit-han-lab/mlsys2026-flashinfer-contest) |
| Full-agent | 3 | UW SyFI | [kamahori/mlsys-contest-syfi-fully-agent](https://github.com/kamahori/mlsys-contest-syfi-fully-agent) |

The official winner page supplies placements but no numerical scores. The old cross-track 0.628x/0.467x/0.456x model ranking is removed because it was not this track's official result. FlashMLA's reported 1,450-TFLOP/s sparse-prefill measurement is also not a Track B submission score and remains attributed only to the FlashMLA source page.
