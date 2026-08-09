---
id: blog-simveit-load-and-store
title: simveit load_and_store
author: Simon Veitner
url: https://github.com/simveit/load_and_store
source_category: community-note
architectures:
- sm90a
tags:
- cuda-cpp
- ptx
- ldmatrix
- stmatrix
retrieved_at: '2026-05-20'
description: Source-map entry imported from KernelPilot for CuTe load/store and shared-memory movement examples.
---

At commit `05d828cf910dd43f0053ddbbe4744218a06e9d7f`, this repository contains
six inline-PTX examples for `ldmatrix` and `stmatrix` x1/x2/x4 forms. Its
Makefile compiles for SM90a. It contains no CuTe, GEMM, TMA, WGMMA, SM100, or
generic vector-width implementation.
