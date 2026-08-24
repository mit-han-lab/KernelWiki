---
id: doc-k-search-kernel-generation
title: "K-Search: LLM Kernel Generation via Co-Evolving Intrinsic World Model"
author: Shiyi Cao et al.
url: https://arxiv.org/abs/2602.19128
source_category: paper
architectures: [sm90, sm100]
tags: [jit-compilation, gemm, attention, moe, mla, kernel-fusion]
retrieved_at: 2026-08-18
---

# K-Search

K-Search separates a high-level optimization-intent tree from repeated concrete
program generation. The world-model step edits that tree using the observed
execution history, allowing an optimization idea to survive an initially
incorrect or slow implementation.

The paper evaluates GQA, MLA, and MoE kernels drawn from FlashInfer. Its
abstract reports an average 2.10x improvement over the compared evolutionary
search and a maximum 14.3x gain on complex MoE cases. It also reports 1030
microseconds on H100 for the GPU Mode TriMul task and describes that result as
surpassing the compared evolutionary and human-written solutions.

Those values are paper-reported comparisons under its evaluation protocol.
They do not establish that generated kernels generally outperform experts or
that the gains transfer to other shapes, devices, search budgets, or baselines.
