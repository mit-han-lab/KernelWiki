---
id: doc-cutile-python-dsl
title: "cuTile Python Documentation"
url: https://docs.nvidia.com/cuda/cutile-python/
source_category: official-doc
architectures: [sm80, sm90, sm100, sm110, sm120]
tags: [cutile, python, gemm, tma]
retrieved_at: 2026-08-18
---

# cuTile Python documentation

cuTile is NVIDIA's tile-based GPU programming model and Python DSL. A function
decorated with `@ct.kernel` executes across a logical grid of blocks and is
queued from the host with `ct.launch`. Global arrays are mutable, strided
storage; tiles are immutable kernel values without a programmer-defined
physical layout. Tile dimensions must be compile-time powers of two.

The current documentation says cuTile automatically uses hardware capabilities
such as Tensor Cores and tensor memory accelerators and is portable across
supported NVIDIA architectures. That is a compiler/runtime abstraction: it does
not guarantee that every load becomes TMA or that every matrix operation lowers
to a particular PTX instruction.

This contiguous excerpt is from the official vector-add example:

```python
@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    result = a_tile + b_tile
    ct.store(c, index=(pid, ), tile=result)
```

As of the access date, the quickstart requires compute capability 8.x, 9.x,
10.x, 11.x, or 12.x, driver R580 or later, and CUDA Toolkit 13.1 or later (or
matching Python packages for the compiler dependencies). The latest listed
cuTile Python release was 1.5.0, dated 2026-07-08. CUDA Toolkit 13.3 added Hopper
support and block-scaled MMA to the documented Python stack.

Because the API is evolving, examples should be pinned to a cuTile release;
older local claims about compiler-selected worker counts, automatic swizzles,
or guaranteed `tcgen05`/TMEM lowering are not treated as public contracts.
