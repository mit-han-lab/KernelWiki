---
id: doc-blackwell-microbenchmarking
title: "Microbenchmarking NVIDIA's Blackwell Architecture"
author: Aaron Jarmusch and Sunita Chandrasekaran
url: https://arxiv.org/abs/2512.02189v3
source_category: paper
architectures: [sm100]
tags: [tcgen05, tmem, fp4, fp8, fp6, gemm]
retrieved_at: 2026-08-18
---

# Microbenchmarking NVIDIA's Blackwell Architecture

This paper presents an open-source microbenchmark suite and reports B200
measurements covering the memory hierarchy, Tensor Memory, the decompression
engine, fifth-generation Tensor Cores, dense and sparse GEMM, inference, and
training. Its comparison platform is H200.

Against version 3 of the paper, the supported results and stated properties are:

- 4.141 TB/s and 4.140 TB/s from its STREAM measurements with four- and
  16-gigabyte working sets;
- a stated 16 TB/s TMEM read bandwidth, presented as an architectural property
  rather than a measured microbenchmark result;
- 1.85x ResNet-50 and 1.55x GPT-1.3B mixed-precision training throughput over
  the compared H200 system; and
- 32% better energy efficiency for the comparison summarized in the abstract.

These figures remain tied to the paper's configurations and methodology. They
are not promoted to guaranteed device limits or universal application
speedups.

The paper describes a physical TMEM interpretation, while the PTX ISA defines
the programmer-visible object as a CTA-owned allocation addressed as 512
columns by 128 lanes of 32-bit cells. This wiki uses the ISA wording for
programming rules and treats the paper's capacity and bandwidth descriptions as
microarchitectural observations.

The earlier local page also summarized a separate RTX 5080 paper and asserted
an “optimal” TMEM tile. Those claims were removed because they were not bounded
by this source record.
