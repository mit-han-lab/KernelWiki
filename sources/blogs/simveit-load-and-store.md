---
id: blog-simveit-load-and-store
title: simveit load_and_store
author: Simon Veitner
url: https://github.com/simveit/load_and_store
source_category: community-note
architectures:
- sm90
tags:
- shared-memory-optimization
- ldmatrix
- stmatrix
retrieved_at: '2026-05-20'
description: Hopper ldmatrix/stmatrix examples compiled for sm_90a.
---

The repository contains small `ldmatrix` and `stmatrix` CUDA/PTX examples, and
its Makefile targets `sm_90a`. It does not contain CuTe, TMA, WGMMA, or an SM100
kernel in the checked tree.
