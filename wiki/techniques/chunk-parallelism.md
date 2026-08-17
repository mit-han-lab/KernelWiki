---
id: technique-chunk-parallelism
title: "Chunk-Based Parallelism for Linear Attention"
type: technique
architectures: [sm100, sm90]
tags: [chunk-parallelism, linear-attention, gated-delta-net, pipeline-stages]
confidence: source-reported
reproducibility: snippet
prerequisites: []
related: [kernel-gated-delta-net, kernel-nsa]
sources: [blog-gated-delta-net, blog-nsa, doc-tfla]
blackwell_relevance: "Chunking exposes matrix operations and state-scan structure that can use SM100 tensor cores; the best chunk size is algorithm-, shape-, and resource-dependent."
---

# Chunk-Based Parallelism

## Overview

Some linear-attention/recurrent formulations can be reorganized into parallel work within chunks plus a recurrence or associative scan across chunk summaries. This exposes matrix multiplies without changing the model's recurrence semantics.

The transformation is algorithm-specific. Gated DeltaNet, RetNet, and state-space models do not share one interchangeable update equation, and a kernel cannot safely let independently scheduled Triton programs read/write a single state tensor without a defined inter-program ordering mechanism.

## Dependency structure

```python
def chunked_recurrence(chunks, initial_state):
    summaries = [compute_chunk_summary(chunk) for chunk in chunks]
    incoming_states = associative_prefix_scan(summaries, initial_state)
    return [compute_chunk_output(chunk, state)
            for chunk, state in zip(chunks, incoming_states)]
```

When the recurrence is not associative in the required representation, the cross-chunk pass remains sequential or uses a mathematically derived alternative. The cited algorithm/code determines which case applies.

## Chunk-size tradeoff

Larger chunks can create more tensor-core-friendly intra-chunk work but increase temporary storage, redundant triangular work, and latency. Smaller chunks reduce local work/storage but increase summary/scan and launch overhead. TMEM capacity is only one resource in this tradeoff; shared memory, registers, occupancy, sequence length, head dimension, and numerical stability also matter.

## Verification

- Compare forward and backward results with the exact recurrent reference.
- Test sequence lengths below, equal to, and not divisible by the chunk size.
- Validate state ordering across batches/heads.
- Sweep chunk size under the actual prefill/decode mix.
- Attribute performance numbers to the full algorithm and benchmark configuration, not “chunking” in isolation.
