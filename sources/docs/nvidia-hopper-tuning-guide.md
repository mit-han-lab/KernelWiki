---
id: doc-nvidia-hopper-tuning-guide
title: "NVIDIA Hopper Tuning Guide (CUDA 12.8.1)"
author: NVIDIA
url: https://docs.nvidia.com/cuda/archive/12.8.1/hopper-tuning-guide/index.html
source_category: official-doc
architectures: [sm90, sm90a]
tags: [tma, cluster, shared-memory-optimization, pipeline-stages]
retrieved_at: 2026-11-11
---

# NVIDIA Hopper Tuning Guide (CUDA 12.8.1)

The archived guide describes Hopper streaming multiprocessors, asynchronous
data movement, thread-block clusters, and shared-memory behavior.

- Section 1.4.1.2 describes TMA asynchronous transfers of 1D through 5D
  tensors, avoiding per-element address calculation and copy instructions in
  the issuing thread. Proper completion synchronization and matching memory
  layouts remain required.
- Section 1.4.1.3 describes thread-block clusters and distributed shared
  memory. The portable maximum cluster size is 8; use
  `cudaOccupancyMaxActiveClusters` for the proposed launch configuration.
- The occupancy and shared-memory sections give **H100** limits, including
  228KB shared memory per SM and 227KB per block. These are not a product
  specification or measured optimum for H20. Query the actual device's
  attributes and occupancy.
- The guide's NVLink bandwidth discussion is also H100-specific; it must not
  be relabeled as an H20 specification.

For H20, this is architectural background supporting candidate TMA,
shared-memory, and pipeline optimizations. It supplies no H20 benchmark or
universally optimal tile, stage count, or warp partition.
