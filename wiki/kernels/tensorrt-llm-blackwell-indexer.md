---
id: kernel-tensorrt-llm-blackwell-indexer
title: TensorRT-LLM Blackwell FP4 DSA Indexer
type: kernel
architectures:
- sm100
tags:
- attention
- gemm
- fp4
- kernel-fusion
- top-k-selection
- vectorized-loads
confidence: source-reported
reproducibility: concept
kernel_types:
- attention
- gemm
- topk
languages:
- cuda-cpp
- python
related:
- kernel-fused-moe
- kernel-fp8-block-scale-gemm
- technique-vectorized-loads
- technique-fine-grained-quantization
sources:
- pr-TensorRT-LLM-13340
performance_claims: []
blackwell_relevance: At the pinned PR revision, TensorRT-LLM's SM100 DSA path fuses FP4 concatenation and quantization, copies packed cache payloads and scales with separate gather/scatter kernels, computes FP4 indexer logits, and selects top-k separately.
---

## Audited scope

TensorRT-LLM PR 13340 adds an FP4 option to its DeepSeek Sparse Attention
(DSA) indexer path. This page describes the PR's pinned merge revision
`897c4bff`; it is a TensorRT-LLM implementation reference, not a generic DSA
ABI or a drop-in FlashInfer-Bench kernel.

The implementation has four distinct stages:

| Stage | Pinned behavior |
| --- | --- |
| Q/K preparation | `fused_cat_fp4` concatenates BF16 positional and non-positional components, then quantizes the 128-value row to FP4 E2M1 with per-32-value UE8M0 scales |
| Cache update/read | Scatter and gather copy an already-quantized payload and its scale word between contiguous tensors and TensorRT-LLM's possibly non-contiguous paged indexer cache |
| Indexer logits | The FP4 path reinterprets the packed bytes/scales and dispatches TensorRT-LLM's DeepGEMM FP4 MQA-logit implementation |
| Selection | Prefill or decode top-k runs after logit computation in a separate operator |

The gather/scatter kernels do not quantize values, compute logits, or select
indices. Calling them “fused quantized gather/scatter” conflates the first two
stages.

## FP4 row and scale contract

For the fixed 128-value indexer head in this path, one row has:

| Component | Representation | Bytes per row |
| --- | --- | ---: |
| Input | BF16 positional values followed by BF16 non-positional values | 256 before quantization |
| Packed payload | Two FP4 E2M1 codes per byte | 64 |
| Scale word | Four UE8M0 exponent bytes, one per successive group of 32 values, packed little-endian into one `int32` | 4 |
| Cache footprint | Packed payload plus scale word | 68 |

The fused operator requires CUDA BF16 inputs on one device, at least two
dimensions, a contiguous innermost dimension, eight-byte-aligned addresses,
equal row counts, and positional width divisible by four. Positional and
non-positional widths must sum to 128. It returns packed `int8[M,64]` data and
`int32[M,1]` scales.

The scale for each 32-value group is
`2^ceil(log2(max(amax, 1e-12) / 6))`. Quantization uses the FP4 E2M1 magnitude
set `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` and packs the earlier value into the low
nibble.

## Gather and scatter semantics

The cache is viewed as
`[num_blocks, block_size, 1, per_token_size]` and may be non-contiguous. Each
token launch copies four bytes per thread. The FP4 path uses 64 payload bytes
and one four-byte scale word; the legacy FP8 path uses 128 payload bytes and
one four-byte FP32 scale.

Payload and scale use separate contiguous `int64` slot-mapping arrays. If
either mapping for a token is negative, gather and scatter skip that entire
token. The gather wrapper allocates output with `empty`, so a directly gathered
row skipped this way has no kernel-written sentinel value; callers must avoid
consuming it or define initialization in their own reference.

The gather wrapper preserves a historical typed view: payload bytes are
returned as float8 and scale bytes as FP32. The FP4 call site reinterprets them
as packed `int8` data and `int32` scale words before the DeepGEMM call. These
views are byte contracts, not evidence that the FP4 payload became FP8 or that
its UE8M0 scale word became an FP32 numeric scale.

## Validation and profiling procedure

1. Compare `fused_cat_fp4` byte-for-byte with a reference implementation for
   zeros, FP4 decision boundaries, saturation, non-contiguous row strides, and
   multiple positional/non-positional splits whose widths sum to 128.
2. Round-trip valid tokens through scatter and gather using strided cache
   views. Check all 64 payload bytes and the four scale bytes independently;
   test negative payload and scale mappings without reading skipped empty rows.
3. Compare FP4 non-paged and paged indexer logits, then selected indices,
   against an unquantized or higher-precision reference with identical masks,
   weights, and sequence boundaries.
4. Profile fused preparation, scatter, gather, logit computation, top-k, and
   the complete indexer separately. Report shapes, cache layout, mapping mix,
   warmup, synchronization, repetitions, statistic, and variation.

The pinned PR contains no performance result, so use it to define candidate
mechanisms and exact contracts rather than to claim a speedup.

## Primary references

- [TensorRT-LLM PR 13340](https://github.com/NVIDIA/TensorRT-LLM/pull/13340)
- [Pinned `fusedCatFp4.cu` artifact](../../artifacts/prs/tensorrt-llm/PR-13340/key-files/cpp/tensorrt_llm/kernels/fusedCatFp4.cu)
- [Pinned gather kernel artifact](../../artifacts/prs/tensorrt-llm/PR-13340/key-files/cpp/tensorrt_llm/kernels/indexerKCacheGather.cu)
- [Pinned scatter kernel artifact](../../artifacts/prs/tensorrt-llm/PR-13340/key-files/cpp/tensorrt_llm/kernels/indexerKCacheScatter.cu)
- [Pinned DSA integration artifact](../../artifacts/prs/tensorrt-llm/PR-13340/key-files/tensorrt_llm/_torch/attention_backend/sparse/dsa.py)
