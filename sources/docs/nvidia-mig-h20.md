---
id: doc-nvidia-mig-h20
title: "NVIDIA MIG User Guide: H20 product identity and requirements"
author: NVIDIA
url: https://docs.nvidia.com/datacenter/tesla/mig-user-guide/pdf/MIG_User_Guide.pdf
source_category: official-doc
architectures: [sm90]
tags: [cuda-cpp]
retrieved_at: 2026-11-11
---

# NVIDIA MIG User Guide: H20

The retrieved PDF is **NVIDIA Multi-Instance GPU User Guide, Release 620**.
Its SHA-256 is
`cb06fb38db8903b6b4c9879c8c6efbcbcea56cd36dc65c4eb967ae8a4f0c4bfc`.
The rolling PDF URL may later serve a newer release.

Chapter 3, Table 1 (printed page 6; PDF page 10) lists:

| Product | Architecture | Microarchitecture | Compute capability | Memory | Maximum MIG instances |
|---|---|---|---|---|---|
| H20 | Hopper | GH100 | 9.0 | 96GB | 7 |

Chapter 7 (printed page 21; PDF page 25), repeated in chapter 10.1
(printed page 47; PDF page 51), gives H20 MIG requirements as **CUDA 12 and
R535 driver >= 535.161.07**. This is a MIG support requirement, not a blanket
minimum for all H20 software.

The table is authoritative for these fields only. It does not establish H20
HBM bandwidth, Tensor Core peak throughput, active SM count, TDP, or
specifications for a differently sized H20 variant. Maximum instance count
does not imply that every instance has full-board resources.

The HTML supported-GPU URL was not usable during retrieval because of a
redirect loop; the NVIDIA-hosted PDF was successfully retrieved instead.
