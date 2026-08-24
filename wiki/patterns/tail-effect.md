---
id: pattern-tail-effect
title: "Tail Effect — Last Wave Underutilization"
type: pattern
tags: [persistent-kernel, clc, tile-scheduling]
symptoms: [tail-effect, low-sm-utilization, wave-quantization]
candidate_techniques: [technique-persistent-kernels, hw-clc, technique-tile-scheduling]
related: [pattern-low-sm-utilization]
sources: [doc-blackwell-microbenchmarking, doc-ptx-isa-sm100]
---

# Tail effect

When a static grid contains a non-multiple of the number of simultaneously resident CTAs, its last wave cannot occupy every available slot. Confirm the effect from the actual grid, occupancy limit, and CTA durations; `num_tiles % num_SMs` is only the one-resident-CTA-per-SM special case.

For example, the locally verified NVIDIA B200 exposes 148 SMs. Under the simplifying assumption of one resident CTA per SM and equal-duration tiles, a 150-CTA grid has a first wave of 148 CTAs and a final wave of two CTAs, leaving 146 SMs without work during that final wave.

Persistent scheduling or CLC can reduce launch overhead and redistribute uneven work. Neither can make 150 independent tile tasks occupy more than 150 tile-time slots, so they do not eliminate the final lack of parallelism in this example.

## What to measure

- resident CTAs per SM for the actual kernel;
- grid and logical tile counts;
- per-CTA duration variance;
- time spent in the final wave;
- changes in cache locality or synchronization introduced by the alternative scheduler.
