---
id: kernel-sparse-mla
title: "DeepSeek Sparse Attention / Sparse MLA"
type: kernel
architectures: [sm100, sm90]
tags: [sparse-attention, mla, fp8, attention, decode, prefill]
confidence: source-reported
reproducibility: concept
kernel_types: [sparse-attention, mla, attention, decode, prefill]
languages: [cuda-cpp, cute-dsl]
related: [kernel-flashmla, kernel-nsa, hw-tcgen05-mma]
sources: [blog-flashmla, blog-vllm-deepseek-v3-sparse, blog-nsa]
performance_claims: []
blackwell_relevance: "On SM100, DeepGEMM supplies FP8 indexer-logit kernels while FlashMLA supplies separate sparse-attention kernels; V3-family sparse decode stores FP8 KV but performs BF16 MMA, and sparse prefill accepts BF16 Q/KV."
---

# DeepSeek Sparse Attention / Sparse MLA

## Model mechanism

DeepSeek-V3.2-Exp introduces **DeepSeek Sparse Attention (DSA)**. Its
first-party report defines two components:

1. The lightning indexer computes one score for every query/preceding-token
   pair. A score is a weighted sum across indexer heads of a ReLU-applied query
   and key dot product.
2. Fine-grained token selection retains the KV entries at the top-k index
   scores and applies the main attention only to that selected set.

The released `config_671B_v3.2.json` uses 64 indexer heads of dimension 128 and
`index_topk=2048`. DSA is instantiated under MLA's MQA mode: the selected MLA
latent entry is shared across the query heads. The model has 128 MLA query
heads, a 512-dimensional latent KV component, and a 64-dimensional RoPE
component; a smaller local head count is a parallel-sharding choice rather than
the architecture-wide contract.

For a length-L sequence and fixed selected count k, the report reduces the
**main/core attention** complexity from O(L squared) to O(Lk). The lightning
indexer still scores the preceding context and remains O(L squared) over the
sequence. For decode, measure both the O(L) selector scan for each new query and
the O(k) selected-attention work; do not describe the full pipeline as O(k).

## Separate implementation boundaries

The released high-performance path is not one fused “indexer plus sparse MLA”
kernel:

- DeepGEMM provides non-paged and paged indexer-logit kernels. Its pinned SM100
  FP8 path computes token logits from FP8 query/key inputs, per-token key
  scales, and per-query head weights; top-k selection remains a separate
  operation.
- FlashMLA sparse attention consumes caller-produced token indices. It does not
  run the lightning indexer or top-k selector.

At FlashMLA commit
[`71c7379`](https://github.com/deepseek-ai/FlashMLA/tree/71c737929f2567bd0a094ae140f8f60f390b1232),
prefill and decode have different contracts:

| Stage | Inputs and selected-token encoding | Precision and outputs |
| --- | --- | --- |
| Sparse decode | `q[batch,s_q,h_q,576]`; paged `k_cache`; `indices[batch,s_q,topk]`, where each nonnegative value encodes physical page times page size plus token offset; `-1` is invalid | V3-family sparse mode dequantizes its FP8 cache for BF16 attention and returns BF16 `out` plus FP32 `lse` |
| Sparse prefill | Unbatched BF16 `q[s_q,h_q,d_qk]`, BF16 `kv[s_kv,h_kv,d_qk]`, and INT32 `indices[s_q,h_kv,topk]`; the documented equivalence requires `h_kv=1`; negative or at-least-`s_kv` entries are invalid | Computes attention over gathered token rows and returns BF16 `out`, FP32 `max_logits`, and FP32 `lse` |

These are token indices, not one maximum or one selection per cache block.
The sparse-decode API accepts page size through the cache tensor; pinned
correctness tests exercise multiple values including 2, 53, 61, 64, 69, 256,
and 576. Page size 64 is therefore a deployment configuration, not a universal
FlashMLA requirement.

## V3-family sparse-decode cache

Only the V3/V3.1/V3.2 **FP8 sparse-decode** mode has the documented 656-byte
per-token layout:

- 512 E4M3 NoPE values: 512 bytes;
- four FP32 scales, one for each successive group of 128 NoPE values: 16 bytes;
- 64 BF16 RoPE values used by the attention key: 128 bytes.

The indexer has a separate FP8 K cache and scale cache in the released model.
The 656-byte attention entry is not an indexer cache and is not the layout of
every dense, sparse-prefill, or non-V3 FlashMLA mode.

## Source-reported performance boundary

FlashMLA's pinned README reports the following maxima. They are useful source
claims, not reproducible benchmark tuples: the README does not provide complete
shapes, timed regions, repetitions, samples, or variance.

| Operator and regime | Reported environment | Author-reported maximum and precision scope |
| --- | --- | --- |
| Dense MLA decode, memory-bound configuration | H800 SXM5, CUDA 12.8 | Up to 3000 GB/s with BF16 cache |
| Dense MLA decode, compute-bound configuration | H800 SXM5, CUDA 12.8 | Up to 660 TFLOPS with BF16 cache; separate from the 3000-GB/s case |
| Sparse MLA decode | H800 SXM5, CUDA 12.8 | 410 TFLOPS; FP8 KV storage and BF16 matrix multiplication |
| Sparse MLA decode | B200 | Up to 350 TFLOPS; the source says this path was not really optimized and gives no bandwidth-causality result |
| Sparse MLA prefill | H800 SXM5, CUDA 12.8 | Up to 640 TFLOPS forward with BF16 Q/KV |
| Sparse MLA prefill | B200, CUDA 12.9 | Up to 1450 TFLOPS forward with BF16 Q/KV |

The same README separately reports NVIDIA's dense **MHA** prefill maxima of
1460 TFLOPS forward and 1000 TFLOPS backward on B200. Those values are not a
dense-MLA baseline matched to the sparse-prefill result, so numerical proximity
between 1450 and 1460 does not establish equivalent performance.

## Evaluation procedure

Use this path directly for DeepSeek-V3.2-Exp, or for another model only after
its query dimensions, latent/RoPE layout, cache format, head relationships,
index encoding, invalid-entry rules, and output semantics match the chosen
FlashMLA interface.

For a target deployment:

1. Compare index scores and selected token sets against the released model
   equation and a dense reference, including causal masking and the indexer's
   non-interleaved RoPE layout.
2. Compare sparse prefill/decode outputs and LSE values with the pinned
   FlashMLA reference for identical selected indices, including invalid and
   partially filled top-k cases.
3. Time indexer logits, top-k selection, sparse attention, and the complete
   pipeline separately across context lengths, query batch sizes, page sizes,
   and selected counts. Use a matched dense-attention baseline and report
   synchronization, warmup, repeated trials, statistic, and variation.

There is no verified universal 32K crossover. The relevant threshold is where
the measured selector-plus-sparse-attention pipeline improves the target's
end-to-end latency or cost without violating its accuracy criterion.

## DSA is not Native Sparse Attention

DeepSeek-V3.2-Exp DSA selects individual token positions with a learned
indexer. The ACL Native Sparse Attention architecture instead combines
compressed-block, selected-block, and sliding-window branches with learned
gates. Do not transfer NSA's A100 results or three-branch structure to this
FlashMLA DSA path.

## Primary references

- [DeepSeek-V3.2-Exp report and released model](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/tree/87e509a2e5a100d221c97df52c6e8be7835f0057)
- [FlashMLA at audited commit `71c7379`](https://github.com/deepseek-ai/FlashMLA/tree/71c737929f2567bd0a094ae140f8f60f390b1232)
- [FlashMLA sparse interface at `71c7379`](https://github.com/deepseek-ai/FlashMLA/blob/71c737929f2567bd0a094ae140f8f60f390b1232/flash_mla/flash_mla_interface.py)
- [DeepGEMM indexer-logit implementation at `891d57b4`](https://github.com/deepseek-ai/DeepGEMM/blob/891d57b4db1071624b5c8fa0d1e51cb317fa709f/deep_gemm/include/deep_gemm/impls/sm100_fp8_mqa_logits.cuh)
- [ACL 2025 Native Sparse Attention paper](https://aclanthology.org/2025.acl-long.1126/)
