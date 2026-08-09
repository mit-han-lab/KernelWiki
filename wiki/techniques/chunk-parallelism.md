---
id: technique-chunk-parallelism
title: "Chunkwise Parallelism for Linear Recurrences"
type: technique
architectures: [sm100, sm90]
tags: [chunk-parallelism, linear-attention, gated-delta-net, pipeline-stages]
confidence: verified
evidence_basis:
  - source_id: doc-tfla
    evidence_type: paper
reproducibility: concept
prerequisites: []
related: [kernel-gated-delta-net, kernel-nsa]
sources: [blog-gated-delta-net, doc-tfla, blog-qwen3-next-architecture]
blackwell_relevance: "Blackwell can accelerate concrete chunkwise kernels, but neither TMEM capacity nor the architecture alone determines a correct or optimal chunk size."
---

# Chunkwise Parallelism for Linear Recurrences

## Mechanism

Some linear recurrent models admit an algebraically equivalent chunkwise formulation. Work local to a chunk can then be expressed with parallel matrix operations, while the state passed across chunk boundaries preserves the recurrence's sequence order. The exact transform is model-specific: it must be derived from the recurrence, not replaced by a generic attention matrix and additive state update.

A correct implementation separates at least these obligations:

1. Compute the chunk-local quantities required by the model's exact recurrence.
2. Resolve boundary states in sequence order, either with an associative scan supported by the formulation or with explicitly ordered stages or launches.
3. Combine each chunk's local result with its incoming boundary state and emit outputs in the original token order.
4. Validate outputs and final states against a token-by-token reference across variable sequence lengths, chunk tails, batches, heads, dtypes, and gate extremes.

An ordinary GPU grid does not imply increasing program-ID execution order or grid-wide synchronization. Programs for every chunk therefore cannot safely read and overwrite one shared state pointer in a single unordered launch. A staged algorithm must make the boundary-state dependency explicit.

## Verified implementations and scope

The pinned NVlabs GatedDeltaNet repository uses chunkwise Triton kernels for training and a WY representation of the gated delta rule. That implementation is direct evidence for GatedDeltaNet chunking, but it is not equivalent to a generic `scores = Q @ K.T; output = scores @ V` snippet and does not supply a universal chunk-size rule.

Tiled Flash Linear Attention (TFLA) starts from the chunkwise formulation of linear RNNs and adds another level of sequence parallelization within a chunk. The authors state that this permits arbitrarily large chunks, raises arithmetic intensity, and reduces intermediate-state materialization. The paper and pinned official code apply the method to mLSTM and report H100 results; they do not establish a GatedDeltaNet implementation, a Blackwell/TMEM implementation, or a recursive-tiling API.

Qwen3-Next is a hybrid-model example, not evidence that one kernel handles every layer. Its immutable 48-layer configuration repeats three Gated DeltaNet linear-attention layers and one full-attention layer twelve times: 36 GDN and 12 full-attention layers. A chunkwise GDN backend applies only to the GDN recurrence; the full-attention layers retain their distinct attention implementation and context-dependent cache.

## Choosing a chunk configuration

There is no source-backed universal rule that `C=32` is a decode choice or that `C=256-512` is the prefill optimum. Choose only among chunk sizes supported by the exact algorithm and backend, then measure the tradeoff:

- arithmetic intensity and matrix-instruction utilization;
- intermediate-state and workspace traffic;
- registers, shared memory, and occupancy;
- tail handling and variable-length metadata;
- launch count and boundary-scan cost;
- latency and throughput for the intended batch and sequence distribution.

Compare candidates with identical compiler/software revisions, launch inputs, correctness oracles, warmups, synchronization, and repeated-trial statistics. Treat a selected size as scoped to the model dimensions, dtype, GPU, backend, and workload. TMEM availability on SM100 may change an implementation's resource design, but it does not by itself prove that a larger chunk is legal or faster.
