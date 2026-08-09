// Extracted from sources/blogs/flash-attention-4.md by scripts/extract_blog_code.py
// Heading: ## Illustrative Code > ### 2-CTA cooperative backward
// Original fence language: cuda
// See artifacts/blogs/flash-attention-4/code/PROVENANCE.yaml for origin + license metadata.

// KernelWiki schematic derived from the FA4 paper/blog dimensions.
// This is not upstream inline PTX or a complete kernel.
struct Fa4TwoCtaBackwardShape {
    static constexpr int cta_group = 2;
    static constexpr int mma_m = 256;
    static constexpr int mma_n = 128;
    static constexpr int mma_k = 128;
    static constexpr int backward_gemm_count = 5;
};
