---
id: kernel-nsa
title: "Native Sparse Attention (NSA)"
type: kernel
architectures: []
tags: [sparse-attention, attention, triton]
confidence: source-reported
reproducibility: snippet
kernel_types: [sparse-attention, attention]
languages: [python, triton]
related: [kernel-flashmla, technique-pipeline-stages]
sources: [blog-nsa, blog-flashmla]
performance_claims: []
blackwell_relevance: "The ACL paper benchmarks an A100 Triton implementation. It provides no SM90/SM100 compatibility result or Blackwell performance comparison; porting and benchmarking are separate work."
---

# Native Sparse Attention (NSA)

## Verified scope

Native Sparse Attention is the three-branch, natively trainable sparse-attention design published in the ACL 2025 proceedings. The paper targets long-context training and inference and reports quality comparable to or better than full attention on its evaluated tasks. Those are paper-scoped results, not a guarantee for an arbitrary model or post-training conversion.

The paper's efficiency experiments use an eight-GPU A100 system. The paper does not establish Hopper or Blackwell compatibility for its Triton implementation.

## Exact three-branch mechanism

For token representation \(h_t\), NSA computes three attention outputs and combines them with learned, input-dependent gates:

\[
o_t = g_t^{cmp} o_t^{cmp} + g_t^{slc} o_t^{slc} + g_t^{win} o_t^{win},
\qquad
g_t^c = \operatorname{sigmoid}(\operatorname{MLP}_c(h_t)).
\]

| Branch | Paper-defined role |
|---|---|
| Compression | Learned MLPs with intra-block position encoding compress overlapping KV blocks into coarse-grained representations. |
| Selection | Compression-attention scores are reused and aggregated into fine-grained block-importance scores; top-n blocks are selected and attended at full token resolution. |
| Sliding window | Recent tokens are attended directly to preserve local context. |

The main experimental configuration uses compression block length 32 and stride 16, selected block length 64 with 16 selected blocks, and a 512-token sliding window. These are experiment settings, not universal NSA constants.

### Executable branch-combination reference

This CPU function checks only the learned gated-sum semantics above. It is KernelWiki-derived, uses ordinary numbers, and makes no claim about the paper's GPU layout or performance.

```python
def gated_branch_sum(branch_outputs, gates):
    """Combine equal-width branch vectors with one [0, 1] gate per branch."""
    if len(branch_outputs) != len(gates) or not branch_outputs:
        raise ValueError("one gate is required for each non-empty branch set")
    width = len(branch_outputs[0])
    if any(len(branch) != width for branch in branch_outputs):
        raise ValueError("branch widths must match")
    if any(not 0.0 <= gate <= 1.0 for gate in gates):
        raise ValueError("sigmoid gates must lie in [0, 1]")
    return [
        sum(gate * branch[i] for branch, gate in zip(branch_outputs, gates))
        for i in range(width)
    ]


assert gated_branch_sum([[1.0, 2.0], [10.0, 20.0], [-2.0, 4.0]],
                        [0.5, 0.25, 1.0]) == [1.0, 10.0]
```

Removing the gates is a useful negative control: the raw branch sum for the same vectors is `[9.0, 26.0]`, not `[1.0, 10.0]`.

## Paper-described Triton design

The authors say compression and sliding-window attention can use existing FlashAttention-2-style kernels, while selected attention needs a specialized Triton kernel for training and prefill. Its source-described structure is:

1. Load all query heads in one GQA/MQA group into SRAM so they share selected sparse KV indices.
2. Load selected KV data as contiguous blocks rather than scattered individual tokens.
3. Put the nearly constant query and output loops in Triton grid parallelism; keep the selected-block loop inside a program because its count is approximately constant.

The paper does not publish the former page's kernel listing, specify a grid exactly as `(query_block, head, batch)`, or claim that such a grid eliminates dynamic scheduling overhead. The removed listings were non-executable and mathematically incorrect: the selected-attention sketch omitted within-block softmax normalization, and the sliding-window sketch stepped by a block size while loading only one token.

## Source-reported efficiency

| Quantity | Source scope at sequence/context length 65,536 |
|---|---|
| Training/prefill forward | 9.0x versus the authors' Triton FlashAttention-2 baseline; Figure 5 timing result |
| Training/prefill backward | 6.0x versus the same baseline; Figure 5 timing result |
| Decoding | 11.6x **expected** speedup from Table 4's memory-access volumes: 65,536 full-attention tokens versus 5,632 NSA-equivalent tokens |

The setup states eight A100 GPUs, GQA group count 4, 16 query heads per group, key dimension 192, and value dimension 128. The paper does not provide the benchmark dtype, software versions, batch details, raw samples, or variance. Consequently these values are retained only as explicitly labeled author reports, and `performance_claims` remains empty rather than encoding a falsely reproducible tuple.

## NSA is not the later DSA deployment

DeepSeek-V3.2-Exp later introduced **DeepSeek Sparse Attention (DSA)**. Its pinned first-party inference code uses a learned indexer to select up to 2,048 token positions and masks attention to those positions. Its README points to DeepGEMM for indexer-logit kernels and FlashMLA for sparse-attention kernels. FlashMLA likewise says its sparse kernels power DSA.

That deployed DSA path is related sparse-attention work, but it is not evidence that the ACL paper's gated compression/selection/window NSA architecture was deployed in V3.2-Exp.

## Applicability and boundaries

- NSA requires a model trained for its compression, selection, window, and gate mechanism; do not treat it as a drop-in sparse mask for an arbitrary checkpoint.
- GQA/MQA lets query heads in a group share selected blocks, matching the paper's group-centric kernel design.
- The paper evaluates long contexts through 64K but defines no universal 32K threshold. Compare quality, sequence distribution, memory use, and measured latency or throughput on the target workload.
- CUDA Graphs can reduce repeated-launch CPU overhead, but they are not an NSA-specific default. Profile first: NVIDIA's guidance says the largest gains occur for CPU-bound workflows and that GPU-bound workloads may see little benefit or regress.
- The compression MLP adds learned parameters and computation. The paper does not isolate a standalone cost for that component.
- The paper notes that a Triton implementation can retain abstraction overhead relative to native CUDA; it does not publish an NSA-specific CPU-launch profile.

## Primary sources

- [NSA in the ACL 2025 Anthology](https://aclanthology.org/2025.acl-long.1126/)
- [ACL proceedings PDF](https://aclanthology.org/2025.acl-long.1126.pdf)
- [DeepSeek-V3.2-Exp at pinned commit 87e509a](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/tree/87e509a2e5a100d221c97df52c6e8be7835f0057)
- [FlashMLA at pinned commit 71c7379](https://github.com/deepseek-ai/FlashMLA/tree/71c737929f2567bd0a094ae140f8f60f390b1232)
- [NVIDIA CUDA Graph performance troubleshooting](https://docs.nvidia.com/dl-cuda-graph/troubleshooting/performance-issues.html)
- [Third-party lucidrains PyTorch implementation](https://github.com/lucidrains/native-sparse-attention-pytorch)

## Query

```bash
conda run -n base python scripts/get_page.py kernel-nsa
```
