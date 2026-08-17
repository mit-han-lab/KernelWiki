---
id: blog-simveit-effective-transpose
title: simveit effective_transpose
author: Simon Veitner
url: https://github.com/simveit/effective_transpose
source_category: community-note
architectures:
- sm90
tags:
- swizzling
- shared-memory-optimization
retrieved_at: '2026-05-20'
description: Hopper CUDA transpose and swizzle examples compiled for sm_90a.
---

The repository provides CUDA transpose and swizzle examples. Its Makefile pins
`-arch=sm_90a`, and its README reports Hopper measurements. It can inform a
layout experiment, but neither those measurements nor the code are direct B200
evidence.
