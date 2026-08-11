// provenance: derived from blog-flashmla, pr-cutlass-2472; not upstream code
// origin: wiki/kernels/flashmla.md Phase 3 variant (copied from extracted blog bundle)

// Extracted from sources/blogs/flashmla.md by scripts/extract_blog_code.py
// Heading: ## Key Code > ### MLA decode inner loop
// Original fence language: cuda
// See artifacts/blogs/flashmla/code/PROVENANCE.yaml for origin + license metadata.

// MLA collapses K and V into a shared latent matrix of head-dim Dc=128.
// On decode (one query vector against N KV tokens) we alternate TMA load,
// wgmma/tcgen05 into the q@K^T accumulator, online softmax, and the second
// accumulator against V.
constexpr int Dc = 128;              // latent head dim
constexpr int BLOCK_N = 64;          // paged KV block size
float acc[Dc] = {0};
float max_val = -INFINITY;
float l = 0.f;
for (int n0 = 0; n0 < seqlen; n0 += BLOCK_N) {
    tma_load(smem_kv, KV_pages + n0);
    cp_async_wait();
    float scores[BLOCK_N];
    wgmma_or_tcgen05_mma(scores, q, smem_kv);       // q @ K^T
    float new_max = warp_reduce_max(scores, BLOCK_N);
    float scale = expf(max_val - new_max);
    for (int j = 0; j < Dc; j++) acc[j] *= scale;
    l *= scale;
    for (int j = 0; j < BLOCK_N; j++) {
        float p = expf(scores[j] - new_max);
        l += p;
        for (int d = 0; d < Dc; d++) acc[d] += p * smem_kv[j * Dc + d];
    }
    max_val = new_max;
}
for (int d = 0; d < Dc; d++) O[d] = acc[d] / l;
