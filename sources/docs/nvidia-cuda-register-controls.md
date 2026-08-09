---
id: doc-cuda-register-controls
title: "CUDA 13 Register Controls and Occupancy APIs"
url: https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html
source_category: official-doc
architectures: [sm100, sm100a, sm90, sm90a]
tags: [register-budgeting, occupancy, launch-bounds, maxrregcount, spills]
retrieved_at: 2026-08-09
---

# CUDA 13 Register Controls and Occupancy APIs

## Evidence Scope

This card routes register-budget claims to archived NVIDIA documentation. The CUDA C++ Programming Guide 13.0 defines launch-bounds compiler behavior and the occupancy calculator. The CUDA Compiler Driver 13.0.2 defines `--maxrregcount`, `--resource-usage`, and the assembler's spill warning. The CUDA Runtime API 13.0.2 defines the occupancy function contract.

## Exact Contracts

- `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor, maxBlocksPerCluster)` supplies launch constraints and compiler guidance. The compiler derives a register threshold `L`; it does not make the second argument an exact achieved block count or directly set registers per thread.
- If initial use exceeds `L`, the compiler reduces it, usually at the expense of local-memory use and/or instruction count. If both maximum threads and minimum blocks are present, the compiler may also increase use up to `L` to reduce instructions.
- `--maxrregcount` sets a maximum for GPU functions. A value below the ABI minimum is raised, some registers are compiler-reserved, and NVIDIA describes the option as a tradeoff between individual-thread performance and available parallelism.
- `--resource-usage` reports registers and memory, including stack-frame bytes and spill loads/stores. `ptxas --warn-on-spills` warns when registers spill to local memory.
- `cudaOccupancyMaxActiveBlocksPerMultiprocessor` returns the maximum active blocks per SM for a compiled function, intended block size, and dynamic shared-memory size. It predicts residency, not application performance.

## Primary References

- [CUDA C++ Programming Guide 13.0: Occupancy Calculator and Launch Bounds](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html)
- [CUDA Compiler Driver 13.0.2: `--maxrregcount` and `--resource-usage`](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-compiler-driver-nvcc/index.html)
- [CUDA Runtime API 13.0.2: Occupancy](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-runtime-api/group__CUDART__OCCUPANCY.html)
