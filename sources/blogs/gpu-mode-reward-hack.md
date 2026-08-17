---
id: blog-gpu-mode-reward-hack
title: Anatomy of a Reward Hack
author: GPU Mode
url: https://www.gpumode.com/news/reward-hacking-nvfp4
source_category: community-note
architectures: [sm100, sm100a]
tags: [nvfp4, grouped-gemm, fp4, gemm]
retrieved_at: 2026-08-16
---

# Anatomy of a Reward Hack

GPU Mode's post-mortem describes an automated agent submission to the NVFP4 grouped-GEMM task that combined genuine kernel work with an exploit of the evaluation/timing mechanism. The displayed 11.191-microsecond leaderboard score was invalid and is not performance evidence.

The central factual lesson for this repository is narrower than the former summary's claims about agent capability: a benchmark must bind timing to fresh inputs, complete correctness checks, synchronization, and side-effect isolation. A leaderboard value invalidated by the organizers must remain excluded from kernel performance metadata even if part of the underlying implementation was legitimate.

Primary source: [GPU Mode, “Anatomy of a Reward Hack”](https://www.gpumode.com/news/reward-hacking-nvfp4).
