---
id: blog-flash-attention-4
title: FlashAttention-4 Blog
author: Tri Dao
url: https://tridao.me/blog/2026/flash4/
source_category: benchmark-blog
architectures:
- sm100
tags:
- attention
- flash-attention
- tcgen05
- tmem
- 2sm-cooperative
- software-exp
- ping-pong-scheduling
- conditional-rescaling
- cute-dsl
retrieved_at: 2026-08-08
artifact_dir: artifacts/blogs/flash-attention-4/code
---

## Evidence Scope

Tri Dao's first-party explanation of FlashAttention-4's Blackwell design and source-reported performance. The two code blocks below are KernelWiki illustrations derived from formulas and dimensions in the post; they are explicitly not verbatim FA4 source from the post.

## Key Techniques

- A CTA alternates two 128-row output tiles so tensor-core MMA and non-matmul softmax/correction work can overlap.
- FA4 evaluates only a selected fraction of exponentials with FMA polynomial code while retaining hardware `ex2` for the rest.
- The software-selected path uses `n=floor(x)`, a fraction in `[0,1)`, and the post's rounded degree-3 coefficients.
- Conditional rescaling compares running maxima, normally with `tau=8.0` base-2 units, retains the old reference maximum on skipped updates, and renormalizes at the end.
- Two-CTA backward shares operand B for five GEMMs, exchanges the required dS half through distributed shared memory for dQ, and reduces dQ global atomic reductions. It does not halve every dQ/dK/dV shared-memory transfer.

## Performance

The post reports up to 1605 TFLOPS/s on B200 BF16, labeled 71%, plus up to 1.3x over cuDNN 9.13 and up to 2.7x over Triton. These are maxima over the author's evaluated configurations, not one fully specified `seqlen=8192, headdim=128` row.

## Illustrative Code

### Software exp (published range reduction and rounded polynomial)

```cuda
// KernelWiki scalar illustration derived from the FA4 blog equations.
// This is not verbatim upstream FA4 code and omits selection and clamping.
#include <cmath>

__host__ __device__ inline float fa4_blog_exp2_reference(float x) {
    const int n = static_cast<int>(floorf(x));
    const float f = x - static_cast<float>(n);  // f in [0, 1)
    const float p = 1.0f + f * (0.6951f + f * (0.2276f + f * 0.0771f));
    return ldexpf(p, n);
}
```

### 2-CTA cooperative backward

```cuda
// KernelWiki schematic derived from the FA4 paper/blog dimensions.
// This is not upstream inline PTX or a complete kernel.
struct Fa4TwoCtaBackwardShape {
    static constexpr int cta_group = 2;
    static constexpr int mma_m = 256;
    static constexpr int mma_n = 128;
    static constexpr int mma_k = 128;
    static constexpr int backward_gemm_count = 5;
};
```
