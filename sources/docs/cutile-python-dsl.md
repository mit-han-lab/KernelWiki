---
id: doc-cutile-python-dsl
title: CUDA Tile Python documentation
url: https://docs.nvidia.com/cuda/cutile-python/
source_category: official-doc
architectures: [sm100, sm100a]
tags: [cutile, python, gemm]
retrieved_at: 2026-08-16
---

# CUDA Tile Python documentation

CUDA Tile Python (`cuda.tile`, conventionally imported as `ct`) is NVIDIA's Python interface to the CUDA Tile programming model. A kernel operates on tiles over a logical block grid, using array arguments in global memory and explicit tile load/store operations. Tiles are immutable values; the API provides elementwise operations, reductions, shape operations, and matrix multiplication.

```python
import cuda.tile as ct

@ct.kernel
def add_kernel(a, b, out, tile_size: ct.Constant[int]):
    block = ct.bid(0)
    result = ct.load(a, index=(block,), shape=(tile_size,)) + ct.load(b, index=(block,), shape=(tile_size,))
    ct.store(out, index=(block,), tile=result)
```

The exact signature and supported shapes must be checked against the installed package/documentation version. The compiler chooses thread-level realization and may select architecture-specific memory or matrix instructions; source code should not promise that every load becomes TMA, every matmul becomes a particular `tcgen05` form, or every accumulator uses TMEM.

The former local page also mixed unpinned requirements and future-architecture claims into the API summary. This record intentionally keeps the portable programming contract separate from version-specific support matrices and profiler requirements.

Primary source: [CUDA Tile Python documentation](https://docs.nvidia.com/cuda/cutile-python/).
