---
id: technique-cccl-memory-primitives
title: CCCL CUB SM100 Scan Tuning
type: technique
architectures: [sm100]
tags: [cuda-cpp, parallel-scan, vectorized-loads, tile-scheduling]
confidence: source-reported
reproducibility: snippet
prerequisites: [technique-vectorized-loads]
related: [pattern-memory-bound, technique-tile-scheduling]
sources: [pr-cccl-3559]
blackwell_relevance: "CCCL PR 3559 adds a dedicated SM100 dispatch policy for CUB exclusive sum and reports a before/after B200 performance comparison."
---

# CCCL CUB SM100 Scan Tuning

CUB device primitives select internal policies by architecture and type. PR
3559 extends the scan policy chain with SM100 specializations rather than
assuming that Hopper policy choices remain optimal on B200.

The following contiguous excerpt is from the retained upstream patch. It shows
one specialization for primitive one-byte values and four-byte offsets; the
numbers are a particular CUB policy, not general application-kernel defaults.

```cuda
template <class ValueT, class AccumT, class OffsetT>
struct sm100_tuning<ValueT, AccumT, OffsetT, op_type::plus, primitive_accum::yes, offset_size::_4, value_size::_1>
{
  static constexpr int items                           = 18;
  static constexpr int threads                         = 512;
  using delay_constructor                              = exponential_backon_constructor_t<768, 820>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier     = LOAD_DEFAULT;
};
```

The patch contains different policies for other value and offset sizes and
falls back through the preceding policy chain where no specialization matches.
Applications should normally call the public CUB scan API and benchmark that
dispatch, rather than copy these implementation-detail policy constants.
