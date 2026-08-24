---
id: blog-flash-attention-4
title: FlashAttention-4 Blog
author: Ted Zadouri, Markus Hoehnerbach, Jay Shah, Timmy Liu, Vijay Thakkar, and Tri Dao
url: https://tridao.me/blog/2026/flash4/
source_category: benchmark-blog
architectures: [sm100]
tags: [attention, flash-attention, tcgen05, tmem, 2sm-cooperative, software-exp, ping-pong-scheduling, conditional-rescaling, cute-dsl]
retrieved_at: 2026-08-18
---

# FlashAttention-4 author blog

The FlashAttention-4 authors' article describes the algorithm/kernel co-design of FlashAttention-4 for Blackwell.

## Source-reported design

- Hopper H100 to Blackwell B200 BF16 tensor-core throughput rises from 1 to 2.25 PFLOPS in the article's comparison, while SFU count and shared-memory bandwidth remain unchanged.
- The forward pass overlaps two query tiles per CTA, softmax, matrix multiplication, and memory operations.
- Exponential work is distributed across MUFU and a Cody-Waite/Horner software approximation on FMA units.
- Conditional rescaling and a dedicated correction stage reduce non-matmul work on the critical path.
- The backward path uses TMEM and two-CTA MMA to reduce shared-memory traffic and global atomic reductions.
- The implementation is written in CuTe DSL, and the article reports substantially shorter compile time than the compared C++ template path.

## Source-reported benchmark snapshot

- Up to 1605 TFLOPS/s on B200 BF16, reported as 71% utilization.
- Up to 1.3× faster than cuDNN 9.13 in the article's tests.
- Up to 2.7× faster than the tested Triton implementation.

The companion paper reports a slightly higher peak across its benchmark sweep. The two result sets are kept in their distinct source contexts rather than merged into one shape-specific claim.

The previous local inline code blocks and extracted-code bundle were removed because they were synthesized illustrations, not verbatim code from this article.
