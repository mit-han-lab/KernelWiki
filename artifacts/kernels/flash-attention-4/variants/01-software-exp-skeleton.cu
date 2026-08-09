// KernelWiki-derived scalar reference for the range reduction and rounded
// degree-3 polynomial printed in Tri Dao's FlashAttention-4 blog.
// This is not verbatim FA4 source. The real kernel applies software evaluation
// to only a selected fraction of entries and includes range handling/scheduling.

#include <cmath>

__host__ __device__ inline float fa4_blog_exp2_reference(float x) {
    const int n = static_cast<int>(floorf(x));
    const float f = x - static_cast<float>(n);  // f in [0, 1)
    const float p = 1.0f + f * (0.6951f + f * (0.2276f + f * 0.0771f));
    return ldexpf(p, n);
}
