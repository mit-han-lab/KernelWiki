---
id: blog-k-search
title: K-Search kernel generation
author: K-Search authors
url: https://arxiv.org/abs/2602.19128
source_category: benchmark-blog
architectures: [sm100, sm100a]
tags: [moe, mla, attention]
retrieved_at: 2026-08-16
---

# K-Search

K-Search uses a separately trained world model to predict candidate-kernel performance during evolutionary search. On the paper's FlashInfer-Bench evaluation, it reports an average final score of 56.13 versus 26.68 for OpenEvolve and 25.37 for ShinkaEvolve: 2.10× and 2.21× respectively. For the selected MoE task it reports 44.1 versus 3.09 for OpenEvolve, a 14.3× ratio.

These ratios compare search systems under the paper's score, trace, task set, budgets, and repeated-run protocol. They are not kernel speedups over FlashInfer and do not mean the generated kernels exceed expert baselines. The paper explicitly analyzes cases where K-Search underperforms other methods on individual workloads.

Primary source: [arXiv:2602.19128](https://arxiv.org/abs/2602.19128).
