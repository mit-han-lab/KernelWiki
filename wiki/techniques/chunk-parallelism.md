---
id: technique-chunk-parallelism
title: "Chunk-Based Parallelism for Linear Recurrent Models"
type: technique
architectures: [sm90]
tags: [chunk-parallelism, linear-attention, triton]
confidence: source-reported
reproducibility: snippet
prerequisites: []
related: [kernel-gated-delta-net, kernel-nsa]
sources: [blog-gated-delta-net, doc-tfla]
blackwell_relevance: "The decomposition is relevant to recurrent-model kernels on newer GPUs, but the cited TFLA implementation reports H100 results and does not establish a Blackwell-specific path."
---

# Chunk-Based Parallelism for Linear Recurrent Models

## What the pattern means

A recurrent sequence can be partitioned into chunks while preserving the state
passed from one chunk to the next. Work inside a chunk can expose matrix
operations and sequence parallelism, but the state dependency between chunks
still has to be respected by the algorithm or by separate kernel launches.

TFLA adds a second level of sequence tiling inside each chunk for mLSTM. The
paper and companion repository describe this as removing the earlier fixed
chunk-size limit. They do not prescribe one generally optimal chunk size;
selection depends on the model dimensions, sequence length, implementation,
and target GPU.

## Upstream usage excerpt

This excerpt is contiguous with the companion repository's README at commit
`5b98ff8e2bec189b3d3c249405bab5149564d6f8`. The tensors are created immediately
above it in that README. It selects the published TFLA mLSTM implementation;
it is not presented as a Gated DeltaNet kernel.

```python
from mlstm_kernels.torch.chunkwise.triton_xl_chunk import mlstm_chunkwise__xl_chunk

matH1 = mlstm_chunkwise__xl_chunk(
    q=matQ, k=matK, v=matV, i=vecI, f=vecF, return_last_states=False, chunk_size=256
)
```

## Correctness boundary

Chunks cannot be launched independently while all of them read and overwrite a
single unsynchronized state buffer. A valid implementation must preserve the
model's exact recurrence, normalization and stabilization rules, initial state,
and chunk-boundary handoff. Decode is normally recurrent; chunkwise parallelism
primarily addresses sequence processing such as training or prefill.

Gated DeltaNet and other linear recurrent models may also use chunkwise
formulations, but the TFLA source cited here evaluates mLSTM. Transfer to another
recurrence requires separate mathematical and implementation evidence.
