---
id: kernel-nsa
title: Native Sparse Attention (NSA)
type: kernel
architectures: [sm80]
tags: [sparse-attention, attention, triton]
confidence: source-reported
reproducibility: snippet
kernel_types: [sparse-attention, attention]
languages: [triton]
related: [kernel-flashmla, kernel-sparse-mla, technique-chunk-parallelism]
sources: [blog-nsa, blog-flashmla, blog-vllm-deepseek-v3-sparse]
performance_claims:
  - gpu: A100
    software: "authors' Triton NSA and Triton FlashAttention-2 kernels; Triton/CUDA versions not stated in the retained result"
    dtype: bf16
    shape: seqlen=65536
    workload: "NSA forward attention at 64K sequence length"
    metric: speedup_forward
    value: 9.0
    measurement_method: "paper-reported component benchmark on an eight-GPU A100 system"
    baseline: "authors' Triton FlashAttention-2 forward kernel"
    limitations: "source-reported component result; not end-to-end serving and not an SM100 measurement"
    source_id: blog-nsa
  - gpu: A100
    software: "authors' Triton NSA and Triton FlashAttention-2 kernels; Triton/CUDA versions not stated in the retained result"
    dtype: bf16
    shape: seqlen=65536
    workload: "NSA backward attention at 64K sequence length"
    metric: speedup_backward
    value: 6.0
    measurement_method: "paper-reported component benchmark on an eight-GPU A100 system"
    baseline: "authors' Triton FlashAttention-2 backward kernel"
    limitations: "source-reported component result; not end-to-end training and not an SM100 measurement"
    source_id: blog-nsa
  - gpu: A100
    software: "paper decoding analysis; implementation versions not stated in the retained result"
    dtype: bf16
    shape: context=65536
    workload: "NSA decoding analysis at 64K context"
    metric: speedup_decode
    value: 11.6
    measurement_method: "paper-reported decoding analysis on the stated A100 system"
    baseline: "full-attention decode"
    limitations: "source-reported component analysis; not a universal end-to-end serving speedup and not an SM100 result"
    source_id: blog-nsa
blackwell_relevance: "NSA's block-structured access pattern is portable in principle, but this repository has no cited SM100 benchmark establishing a Blackwell-specific gain."
---

# Native Sparse Attention

NSA is a trainable sparse-attention design with three paths: compressed coarse context, selected fine-grained blocks, and a local sliding window. Its selection is block structured so that the sparse computation can use contiguous tiles rather than arbitrary token gathers.

## Kernel boundary

The model mechanism includes compression and selection as well as sparse attention. A sparse-attention kernel that consumes selected indices is only one part of that system. Implementations must define block layout, causal masking, duplicate/invalid-index behavior, GQA sharing, softmax normalization across the selected set, and how the three paths are combined.

```python
def selected_attention(query, key_blocks, value_blocks, selected_blocks):
    keys = gather_blocks(key_blocks, selected_blocks)
    values = gather_blocks(value_blocks, selected_blocks)
    return softmax(query @ keys.T, axis=-1) @ values
```

This is only the selected-attention contract; it does not implement NSA's compression or sliding-window paths.

The removed Triton snippets did not implement those complete semantics and were not upstream code. They also implied data sharing across programs that Triton does not provide merely by using a “group-centric” index calculation.

## Performance evidence

The paper reports 9× forward and 6× backward speedups at a 64K sequence length against its Triton FlashAttention-2 baseline on the stated eight-GPU A100 system. Its decoding analysis separately reports up to 11.6× at 64K context. All three are represented with their operation and baseline boundaries from the paper's Section 5.

This repository does not have a source establishing that B200 L2 capacity itself improves the sparse fetches, so no architecture-specific speedup is inferred.

## Related implementation

DeepSeek V3.2's indexer-plus-sparse-MLA serving path is documented separately in [Sparse MLA](sparse-mla.md). It should not be conflated with all three NSA training-time paths.
