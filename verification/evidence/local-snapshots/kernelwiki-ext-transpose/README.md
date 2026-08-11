# Effective matrix transpose

Improve matrix transpose step by step.

Please see [my blogpost](http://veitner.bearblog.dev/making-matrix-transpose-really-fast-on-hopper-gpus/) for a detailed explanation.

## Performance Comparison

| Kernel | Bandwidth (GB/s) | % of Max Bandwidth | Implementation |
|--------|------------------|-------------------|----------------|
| transpose_naive | 875.46 | 26.5291% | Custom |
| transpose_swizzle | 1251.76 | 37.9323% | Custom |
| transpose_swizzle_batched | 2771.35 | 83.9804% | Custom |
