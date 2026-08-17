---
id: doc-blackwell-compatibility-guide
title: NVIDIA Blackwell Compatibility Guide
url: https://docs.nvidia.com/cuda/blackwell-compatibility-guide/
source_category: official-doc
architectures: [sm100, sm100a]
tags: [ptx, cuda-cpp]
retrieved_at: 2026-08-16
---

# NVIDIA Blackwell Compatibility Guide

## Compatibility rules

- A cubin runs on the same compute-capability major revision and the same or a higher minor revision. It is not generally forward-compatible across major revisions.
- PTX is forward-compatible to GPUs with a higher supported compute capability and is JIT-compiled at runtime.
- Architecture-conditional `compute_100a`/`sm_100a` code is neither forward nor backward compatible. Likewise, `compute_90a` PTX is not supported on Blackwell.
- CUDA 12.8 is the first toolkit able to generate native compute-capability-10.0 cubins.
- Applications built with CUDA 2.1 through 12.8 can run on Blackwell when they contain suitable PTX; `CUDA_FORCE_PTX_JIT=1` is the guide's compatibility test for that path.

## Build pattern

```bash
nvcc -gencode=arch=compute_100,code=sm_100 \
     -gencode=arch=compute_100,code=compute_100 kernel.cu
```

The native cubin avoids first-use JIT for SM100, while retained PTX provides a forward-compatible path where supported. The exact product-to-target mapping and family-specific targets should be taken from the toolkit/compiler documentation rather than inferred from the Blackwell marketing name.

Primary source: [NVIDIA Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/).
