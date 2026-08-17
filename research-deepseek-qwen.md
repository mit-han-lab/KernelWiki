# Audited DeepSeek and Qwen kernel research index

This document is a navigation index, rechecked on 2026-08-16. Detailed claims belong in the cited source and wiki pages, where dtype, shape, version, and evidence boundaries are recorded.

## DeepGEMM

The checked current DeepGEMM README and pinned SM90/SM100 files support these scoped statements:

- the library provides runtime-compiled dense/grouped GEMM and related LLM primitives on SM90 and SM100;
- SM90 interfaces use FP32 scale factors and the checked implementation promotes scaled 128-K-block results into a separate FP32 accumulator;
- SM100 interfaces use packed UE8M0 scale factors and cover NT, TN, NN, and TT layouts in the current README;
- contiguous M-grouped GEMM varies M while N/K remain fixed, masked M-grouped GEMM handles device-side valid counts, and K-grouped weight-gradient interfaces are a separate contract;
- NVCC is the default JIT compiler; `DG_JIT_USE_NVRTC=1` is opt-in and may reduce kernel performance.

The README's historical “up to 1,550 TFLOP/s on H800” item is an unspecified peak. It is not a 4096-cubed benchmark, a universal utilization result, or evidence for SM100 performance. See `blog-deepgemm`, `kernel-deepgemm`, and `technique-fine-grained-quantization`.

## FlashMLA and sparse MLA

The current FlashMLA README lists SM90 dense decode, SM90/SM100 sparse decode, SM100 dense MHA prefill, and SM90/SM100 sparse MLA prefill. Its headline paragraphs report 1,460/1,000 TFLOP/s for B200 dense-MHA-prefill forward/backward “as reported by NVIDIA” and 1,450 TFLOP/s for B200/CUDA 12.9 sparse prefill, but do not identify complete peak shapes or prefill dtypes. The wiki therefore stores them as source-reported unspecified peaks and does not infer utilization or dtype. See `blog-flashmla`, `kernel-flashmla`, and `kernel-sparse-mla`.

## Native Sparse Attention

The primary NSA paper states that its efficiency experiments use an eight-GPU A100 system and compares its Triton implementation with a Triton FlashAttention-2 baseline. At 64K context it reports up to 9x forward and 6x backward speedup; its decode analysis reports up to 11.6x at 64K. These are operation- and baseline-specific A100 paper results, not H100 or B200 measurements. See `blog-nsa` and `kernel-nsa`.

## Gated Delta Net and Qwen

The NVlabs/FLA record establishes the gated delta-rule algorithm and project history. Its README records later adoption in Qwen3-Next, Qwen3.5, and OLMo Hybrid, but does not establish one universal state shape, Blackwell lowering, or kernel speedup. A former “10x+ versus Qwen3-32B” entry mixed model-level serving results with a kernel claim and was removed. See `blog-gated-delta-net`, `doc-tfla`, `kernel-gated-delta-net`, and `technique-chunk-parallelism`.

The Qwen3-Next source record is limited to what the checked official architecture material states. Model-family ratios and adoption claims are kept version/date scoped rather than treated as permanent kernel properties.

## FlashAttention-4 and Blackwell implementation studies

The FlashAttention-4 paper is the primary source for its B200 algorithm and performance claims: pipelined forward, partial software exponential, 2-CTA backward, and maxima within the paper's benchmark sweep. The 1,613-TFLOP/s headline and comparisons are for the complete kernel, not isolated techniques or CuTe DSL generally. See `doc-flash-attention-4`, `kernel-flash-attention-4`, `technique-software-exp`, and `technique-ping-pong-scheduling`.

Community tutorials (`blog-tcgen05-tutorial`, `blog-jax-pallas-blackwell`, and `blog-modular-blackwell`) contain useful source-reported progressions for their exact kernels. They do not define canonical warp maps, instruction constraints, stage counts, or universal per-technique speedups.

## Scope boundary

Distributed-system topics such as DeepEP, DualPipe, and EPLB are outside this kernel-only knowledge base. Their old unpinned performance summaries are not retained here. Use the upstream projects directly when distributed communication or training-pipeline behavior is the actual subject.
