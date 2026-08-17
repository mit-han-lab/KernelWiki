---
id: contest-flashinfer-track-a
title: 'FlashInfer MLSys 2026 - Track A: Fused MoE'
source_category: contest-report
architectures: [sm100]
tags: [moe, fp8, block-scale, fused-kernel]
techniques: [kernel-fusion, tile-scheduling]
hardware_features: [fp8, block-scale]
kernel_types: [moe, fused-kernel, gemm, grouped-gemm]
url: https://github.com/flashinfer-ai/mlsys26-contest/blob/a523107f1ff793871a04bd535f921a0a4bd44173/index.html
submissions:
  - rank: agent-assisted 1
    participant: Team Wombat
    submission_truth: unavailable
    code_unavailable_reason: The official page links a public submission repository, but this repository has not pinned it into a local contest-submission bundle.
  - rank: agent-assisted 2
    participant: KernelEvolve
    submission_truth: unavailable
    code_unavailable_reason: The official page links a public submission repository, but this repository has not pinned it into a local contest-submission bundle.
  - rank: agent-assisted 3
    participant: LLM-CUDA
    submission_truth: unavailable
    code_unavailable_reason: The official page links a public submission repository, but this repository has not pinned it into a local contest-submission bundle.
  - rank: full-agent 1
    participant: HAN Lab Kernel Mafia
    submission_truth: unavailable
    code_unavailable_reason: The official page links a public submission repository, but this repository has not pinned it into a local contest-submission bundle.
  - rank: full-agent 2
    participant: GEMM People
    submission_truth: unavailable
    code_unavailable_reason: The official page links a public submission repository, but this repository has not pinned it into a local contest-submission bundle.
  - rank: full-agent 3
    participant: Insider
    submission_truth: unavailable
    code_unavailable_reason: The official page links a public submission repository, but this repository has not pinned it into a local contest-submission bundle.
---

# Track A: Fused MoE

The official contest page describes a fused-MoE task with FP8 support on NVIDIA B200. Its benchmark link identifies the task as `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`. Submissions could use hand-written or agent-generated implementations and were divided into agent-assisted and full-agent approaches.

## Official placements

| Approach | Place | Team | Public repository linked by organizer |
|---|---:|---|---|
| Agent-assisted | 1 | Team Wombat | [cheshire/flashinfer-challenge](https://github.com/cheshire/flashinfer-challenge) |
| Agent-assisted | 2 | KernelEvolve | [QasimKhan5d/fused-moe](https://github.com/QasimKhan5d/fused-moe) |
| Agent-assisted | 3 | LLM-CUDA | [syhya/mlsys26-flashinfer-contest](https://github.com/syhya/mlsys26-flashinfer-contest) |
| Full-agent | 1 | HAN Lab Kernel Mafia | [mit-han-lab/mlsys2026-flashinfer-contest](https://github.com/mit-han-lab/mlsys2026-flashinfer-contest) |
| Full-agent | 2 | GEMM People | [Jerry2423/MoE-Kernel-Agent](https://github.com/Jerry2423/MoE-Kernel-Agent) |
| Full-agent | 3 | Insider | [MayankSuthar1/mlsys_2026_contest](https://github.com/MayankSuthar1/mlsys_2026_contest) |

The organizer's winner page does not publish numerical scores. The old 0.628x/0.467x/0.456x model table was a separate AI-agent baseline, not the Track A podium, and has been removed. Likewise, SGLang's 1,262-TFLOP/s and 206.9-us figures are third-party baseline measurements, not a contest submission result; they belong with their original performance source rather than this report.
