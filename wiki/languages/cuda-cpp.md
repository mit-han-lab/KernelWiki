---
id: lang-cuda-cpp
title: "CUDA C++ for Blackwell Kernels"
type: language
tags: [cuda-cpp, ptx, tcgen05, tmem]
related: [lang-ptx, hw-tcgen05-mma, hw-tmem, blog-tcgen05-tutorial]
sources: [doc-ptx-isa-sm100, doc-nvidia-tuning-guide, doc-cutlass-cute-dsl]
reproducibility: snippet
architectures: [sm100, sm100a]
confidence: source-reported
---

# CUDA C++ for Blackwell kernels

CUDA C++ kernels can reach Blackwell-specific facilities through compiler abstractions such as CUTLASS/CuTe or through inline PTX. Inline PTX does not relax PTX participation, alignment, descriptor, memory-ordering, or target requirements.

## Query hardware-dependent values

This host fragment is compilable with `nvcc` and avoids embedding a product-wide SM-count constant in a scheduler:

```cpp
#include <cuda_runtime.h>
#include <cstdio>

int main() {
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) return 1;
  std::printf("%s cc=%d.%d sms=%d\n", properties.name,
              properties.major, properties.minor, properties.multiProcessorCount);
}
```

## Inline-PTX review checklist

- Gate target-specific instructions with a compatible compilation target.
- Match C++ operand constraints and widths to the PTX operand types.
- Keep `.sync.aligned` allocation/load/store operations converged across a fully active warp.
- Give `tcgen05.mma` a correctly encoded 32-bit instruction descriptor.
- Preserve async-proxy fences and the documented completion mechanism.
- Make a TMA or matrix descriptor agree with the actual shared-memory layout.

Prefer a maintained CUTLASS/CuTe wrapper when it expresses the operation. If inline PTX is required, copy the form from the current PTX ISA and test compilation plus numerical correctness on the intended architecture.
