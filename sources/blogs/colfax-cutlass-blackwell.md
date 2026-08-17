---
id: blog-colfax-cutlass
title: 'Colfax CUTLASS Tutorial: GEMM Kernels Using Tensor Memory for Blackwell'
author: Colfax Research
url: https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tmem-for-nvidia-blackwell-gpus/
source_category: community-note
architectures: [sm100]
tags: [tcgen05, tmem, cute-dsl, warp-specialization, 2sm-cooperative]
retrieved_at: 2026-08-16
---

# Colfax CUTLASS Blackwell tutorial

The Colfax tutorial explains CUTLASS/CuTe abstractions for SM100 MMA and TMEM, including MMA atoms/traits and layout-driven accumulator access. It should be read together with the CUTLASS version it targets: public type names and traits are version-sensitive.

Architectural facts are checked against the PTX ISA in `doc-ptx-isa-sm100`: one thread issues `tcgen05.mma`, the destination is TMEM, A may be shared memory or TMEM for supported forms, B is shared memory, and TMEM register transfers have exact warp participation/layout rules.

The former local page called the operation “register-free” and embedded abbreviated inline PTX with invalid allocation operands and unverified CUTLASS type names. Those snippets were not verbatim article code and are removed.
