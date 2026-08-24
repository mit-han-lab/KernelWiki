# DeepSeek and Qwen research disposition

The original scratch research mixed kernel evidence with cluster-level systems topics, repeated secondary performance summaries, and several figures that could not be traced to the cited primary sources. It is superseded by the evidence-checked pages below.

## In-scope kernel material

- [DeepGEMM](wiki/kernels/deepgemm.md) and [FP8 block-scale GEMM](wiki/kernels/fp8-block-scale-gemm.md) retain only the repository's reported H800 throughput claim and identify its unspecified benchmark shape.
- [FlashMLA](wiki/kernels/flashmla.md), [SparseMLA](wiki/kernels/sparse-mla.md), and [Native Sparse Attention](wiki/kernels/nsa.md) distinguish source-reported measurements from explanatory synthesis.
- [Gated DeltaNet](wiki/kernels/gated-delta-net.md) describes the kernel-facing operator and links the official Qwen3-Next architecture source without repeating an unsupported throughput multiplier.
- [FlashAttention-4](wiki/kernels/flash-attention-4.md) and the associated [paper source](sources/docs/flash-attention-4.md) use the paper's reported peak result and benchmark context.
- [tcgen05 tutorial](sources/blogs/tcgen05-tutorial.md), [Tensor Memory](wiki/hardware/tmem.md), and related technique pages record only figures traceable to their cited sources.

## Removed material

DeepEP dispatch bandwidth, EPLB placement gains, DualPipe scheduling, and other multi-node or cluster-level claims are outside this kernel-only repository's scope. They were removed rather than retained as back-links. Untraceable launch-latency, TMEM-latency, energy-efficiency, and derived-utilization claims were also removed from the maintained pages.

The terminal reason for each source-PR exclusion is recorded in [scope disposition](audit/scope-disposition.md); numeric-claim decisions are recorded in [numeric claims ledger](audit/numeric-claims-ledger.md).
