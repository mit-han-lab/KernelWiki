---
id: kernel-nsa
title: "Native Sparse Attention (NSA)"
type: kernel
architectures: [sm80]
tags: [sparse-attention, attention, triton]
confidence: source-reported
reproducibility: snippet
kernel_types: [sparse-attention, attention]
languages: [triton, python]
related: [kernel-flashmla, technique-pipeline-stages]
sources: [doc-nsa, blog-vllm-deepseek-v3-sparse]
performance_claims:
  - gpu: A100
    dtype: not stated in the paper
    shape: "seqlen=65536"
    metric: speedup
    value: 9.0
    utilization: "versus the paper's Triton FlashAttention-2 forward implementation"
    source_id: doc-nsa
    source_locator: "https://arxiv.org/html/2502.11089#S5.SS1 (Figure 6 discussion)"
blackwell_relevance: "The algorithm is architecture-independent; any Blackwell benefit must be measured for the selected GPU, cache behavior, and workload."
---

# Native Sparse Attention (NSA)

## Algorithm

The DeepSeek-led NSA paper presents a natively trainable sparse-attention design with three
parallel branches:

1. compressed tokens provide coarse global context;
2. a learned selection branch chooses fine-grained tokens; and
3. a sliding-window branch preserves local context.

The branch outputs are combined by the model. This is more than a sparse
attention kernel dropped into an otherwise unchanged model: the compression
and selection machinery is part of the trained architecture.

## Reported performance

In the paper's A100/Triton comparison at 64K context, NSA reaches up to 9.0x
forward and 6.0x backward speedup versus its FlashAttention-2 baseline. Table 4
separately derives an expected decoding speedup of up to 11.6x at that context
length from per-operation memory-access volume; it is not a measured decoding
latency result.
These are maxima for the paper's implementation and workload, not H100 or B200
measurements and not guarantees for another sparse-attention backend.

## Reproduction boundary

The paper does not provide the two purported Triton kernels that an earlier
version of this page printed; those local inventions were removed. For a
concrete adjacent integration, the vLLM DeepSeek-V3.2 post gives this exact
two-line layout excerpt for its separate FP8 indexer cache:

```python
x_fp8[:, : block_size * head_dim] = x_scaled.view(num_blocks, block_size * head_dim).view(dtype=torch.uint8)
x_fp8[:, block_size * head_dim :] = scales.view(num_blocks, block_size).view(dtype=torch.uint8)
```

That excerpt is not an NSA reference implementation and does not reproduce the
paper's benchmark. It is retained only as source-backed sparse-attention
integration code. Use the paper for NSA's algorithm and evaluation, and the
chosen runtime's versioned source for an executable deployment.

## Hardware alignment described by the paper

The paper attributes its kernel efficiency to blockwise memory access,
group-centric loading that shares selected KV data across query heads, and
loop scheduling that avoids redundant KV transfers. Those are design
principles; the legal tile shapes, vector widths, and launch geometry come from
the actual implementation and target GPU.

## Caveats

- Quality depends on training the model with the sparsity mechanism.
- Selection, compression, and branch combination add work outside the sparse
  attention multiply itself.
- DeepSeek-V3.2 deployment material is useful adjacent evidence, but its DSA
  integration must not be presented as the NSA paper's missing source code.

## Sources

- [NSA paper](https://arxiv.org/abs/2502.11089)
- [vLLM DeepSeek-V3.2 integration post](https://blog.vllm.ai/2025/09/29/deepseek-v3-2.html)
