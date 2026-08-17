# Audited contest research index

This index was rechecked on 2026-08-16. It separates public task contracts, participant-authored reports, third-party baselines, and official placements. Those evidence classes are not interchangeable.

## GPU Mode NVIDIA NVFP4 tasks

The pinned `gpu-mode/reference-kernels` task files define four B200 workloads:

| Local source | Public contract |
|---|---|
| `contest-gpumode-p1` | Batched NVFP4 GEMV; three benchmark shapes; geometric-mean latency ranking |
| `contest-gpumode-p2` | NVFP4 GEMM; three benchmark shapes; geometric-mean latency ranking |
| `contest-gpumode-p3` | Dual NVFP4 GEMM with SiLU; four benchmark shapes; geometric-mean latency ranking |
| `contest-gpumode-p4` | Grouped NVFP4 GEMM; four grouped cases; geometric-mean latency ranking |

The task files describe E2M1 data, FP8 E4M3FNUZ scale tensors, FP16 outputs, shape constraints, tests, benchmark cases, and analytical estimates. They do not publish a final contestant leaderboard. The former Problem 2/3/4 podium values and participant-specific implementation descriptions are therefore not retained as public contest facts.

For Problem 1, two public participant retrospectives provide separate author evidence:

- `blog-yue-nvfp4` reports a final 22.392-us suite geometric mean and an optimization progression;
- `blog-amandeep-nvfp4` reports 26.7, 45.1, and 16.4 us for the three task shapes and discusses its own experiments and observations of public solutions.

Neither retrospective establishes the podium asserted by the old contest summary. `blog-simon-nvfp4` is explicitly unresolved because the configured article returns “Not found.” No reconstructed contestant code is retained.

GPU Mode's separate `blog-gpu-mode-reward-hack` post-mortem marks a displayed 11.191-us grouped-GEMM score invalid because of evaluation/timing state reuse. That incident is evaluation-harness evidence, not grouped-kernel performance evidence.

## FlashInfer MLSys 2026

The official contest page defines three NVIDIA B200 tracks and now publishes two podiums per track: agent-assisted and full-agent. The local source pages record the official placements and organizer-linked repositories:

- `contest-flashinfer-track-a`: Fused MoE;
- `contest-flashinfer-track-b`: DeepSeek Sparse Attention indexer plus sparse-attention definitions;
- `contest-flashinfer-track-c`: Gated Delta Net decode plus prefill definitions.

The official winner page does not publish numerical scores. Earlier 0.628x/0.467x/0.456x model values came from a separate agent-baseline evaluation and were incorrectly repeated as each track's podium. SGLang/FlashInfer/vLLM MoE measurements, FlashMLA throughput headlines, and cross-model Qwen throughput comparisons likewise remain with their original sources rather than being relabeled contest results.

## Reusable evidence rules

- Use a task file for the numerical interface, constraints, test cases, and benchmark shapes.
- Use an official winner page for placement only.
- Use a participant post for that participant's reported measurements and methods, with author-report qualification.
- Use a linked submission repository or pinned artifact for code claims.
- Never infer a participant rank, score, or causal optimization from a private channel or an unrelated baseline table.

Primary local records live under `sources/contests/`; participant and benchmark sources live under `sources/blogs/`.
