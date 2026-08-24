---
id: doc-blackwell-compatibility-guide
title: "NVIDIA Blackwell Compatibility Guide"
url: https://docs.nvidia.com/cuda/blackwell-compatibility-guide/
source_category: official-doc
architectures: [sm100, sm100a]
tags: [ptx, cuda-cpp]
retrieved_at: 2026-08-18
---

# NVIDIA Blackwell Compatibility Guide

NVIDIA's compatibility guide distinguishes native cubins from virtual PTX.

- A cubin is compatible within the same compute-capability major revision on a
  GPU whose minor revision is the same or higher. It is not compatible across
  major revisions.
- Ordinary PTX is forward-compatible: PTX generated for a compute capability
  can be JIT-compiled for a GPU with that capability or a higher major/minor
  capability, subject to driver support for the PTX version.
- Architecture-conditional PTX or cubins using targets such as `compute_100a`
  or `sm_100a` are not forward- or backward-compatible. The guide gives
  `compute_90a` PTX failing on Blackwell as an example.

The earlier local matrix incorrectly said ordinary `compute_100` PTX could not
JIT to a higher compute capability. That contradicted the guide's general
forward-compatibility rule and has been removed.

CUDA Toolkit 12.8 introduced native compute-capability 10.0 cubin generation.
The guide's Linux example includes both a native cubin and PTX fallback:

```bash
nvcc -gencode=arch=compute_100,code=sm_100 \
     -gencode=arch=compute_100,code=compute_100 \
     -O2 -o mykernel.o -c mykernel.cu
```

For an existing binary, the guide recommends setting
`CUDA_FORCE_PTX_JIT=1`, running the application, and then unsetting the
variable. This ignores embedded cubins and verifies that every launched kernel
has usable PTX. A successful check demonstrates the PTX path; it does not
measure the performance of a native Blackwell rebuild.

The rolling page was labeled Compatibility Guide 13.3 when accessed on
2026-08-18. Its compatibility rules do not provide a product-to-SM catalog or
an instruction-by-instruction list of features requiring the `a` target.
