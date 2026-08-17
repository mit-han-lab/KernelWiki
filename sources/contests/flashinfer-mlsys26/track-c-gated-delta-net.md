---
id: contest-flashinfer-track-c
title: 'FlashInfer MLSys 2026 - Track C: Gated Delta Net'
source_category: contest-report
architectures: [sm100]
tags: [gated-delta-net, linear-attention, chunk-parallelism]
techniques: [chunk-parallelism, kernel-fusion]
kernel_types: [gated-delta-net, linear-attention, decode, prefill]
url: https://github.com/flashinfer-ai/mlsys26-contest/blob/a523107f1ff793871a04bd535f921a0a4bd44173/index.html
submissions:
  - rank: agent-assisted 1
    participant: Kachua
    submission_truth: unavailable
    code_unavailable_reason: The official page links a public submission repository, but this repository has not pinned it into a local contest-submission bundle.
  - rank: agent-assisted 2
    participant: UW SyFI
    submission_truth: unavailable
    code_unavailable_reason: The official page links a public submission repository, but this repository has not pinned it into a local contest-submission bundle.
  - rank: agent-assisted 3
    participant: LLM-CUDA
    submission_truth: unavailable
    code_unavailable_reason: The official page links a public submission repository, but this repository has not pinned it into a local contest-submission bundle.
  - rank: full-agent 1
    participant: UW SyFI
    submission_truth: unavailable
    code_unavailable_reason: The official page links a public submission repository, but this repository has not pinned it into a local contest-submission bundle.
  - rank: full-agent 2
    participant: LLM-CUDA
    submission_truth: unavailable
    code_unavailable_reason: The official page links a public submission repository, but this repository has not pinned it into a local contest-submission bundle.
  - rank: full-agent 3
    participant: HAN Lab Kernel Mafia
    submission_truth: unavailable
    code_unavailable_reason: The official page links a public submission repository, but this repository has not pinned it into a local contest-submission bundle.
---

# Track C: Gated Delta Net

The official contest page describes Gated Delta Net kernels on NVIDIA B200 and links two benchmark definitions:

- decode: `gdn_decode_qk4_v8_d128_k_last`;
- prefill: `gdn_prefill_qk4_v8_d128_k_last`.

The page identifies GDN as used by Qwen3-Next. It does not support the old page's assertion that Qwen3.5 was part of the task description or that standard attention is replaced in a stated percentage of layers.

## Official placements

| Approach | Place | Team | Public repository linked by organizer |
|---|---:|---|---|
| Agent-assisted | 1 | Kachua | [romitjain/kachua-mlsys](https://github.com/romitjain/kachua-mlsys) |
| Agent-assisted | 2 | UW SyFI | [kamahori/mlsys-contest-syfi-agent-assisted](https://github.com/kamahori/mlsys-contest-syfi-agent-assisted) |
| Agent-assisted | 3 | LLM-CUDA | [syhya/mlsys26-flashinfer-contest](https://github.com/syhya/mlsys26-flashinfer-contest) |
| Full-agent | 1 | UW SyFI | [kamahori/mlsys-contest-syfi-fully-agent](https://github.com/kamahori/mlsys-contest-syfi-fully-agent) |
| Full-agent | 2 | LLM-CUDA | [syhya/mlsys26-flashinfer-contest](https://github.com/syhya/mlsys26-flashinfer-contest) |
| Full-agent | 3 | HAN Lab Kernel Mafia | [mit-han-lab/mlsys2026-flashinfer-contest](https://github.com/mit-han-lab/mlsys2026-flashinfer-contest) |

The organizer's page publishes placements but no score values. The old repeated 0.628x/0.467x/0.456x model table was not the Track C podium, and the “10x+ versus Qwen3-32B” comparison mixed different models and workloads; both have been removed.
