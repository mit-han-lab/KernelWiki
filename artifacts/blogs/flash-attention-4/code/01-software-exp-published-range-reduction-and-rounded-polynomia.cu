// Extracted from sources/blogs/flash-attention-4.md by scripts/extract_blog_code.py
// Heading: ## Illustrative Code > ### Software exp (published range reduction and rounded polynomial)
// Original fence language: cuda
// See artifacts/blogs/flash-attention-4/code/PROVENANCE.yaml for origin + license metadata.

// KernelWiki scalar illustration derived from the FA4 blog equations.
// This is not verbatim upstream FA4 code and omits selection and clamping.
#include <cmath>

__host__ __device__ inline float fa4_blog_exp2_reference(float x) {
    const int n = static_cast<int>(floorf(x));
    const float f = x - static_cast<float>(n);  // f in [0, 1)
    const float p = 1.0f + f * (0.6951f + f * (0.2276f + f * 0.0771f));
    return ldexpf(p, n);
}
