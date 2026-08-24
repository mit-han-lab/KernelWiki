# Triton 3.6 Blackwell evidence boundary

This memo separates compiler capability from downstream kernel adoption.

## Compiler capability

`doc-triton-3.6-blackwell` summarizes the official Triton 3.6 release notes. Those notes directly enumerate TMEM, `tcgen05`, warp-specialization, scaled-MMA, Gluon multi-CTA, and initial two-CTA work for Blackwell.

## Retained downstream code

- `pr-vllm-34597`: inspectable `@triton.jit` MLA decode attention using `tl.dot`; primary downstream code anchor.
- `pr-sglang-22079`: inspectable Triton attention with `tl.dot` on SM100/SM90.
- `pr-sglang-21019`: inspectable Triton memory-rearrangement kernel; adoption evidence only, not MMA-lowering evidence.

## What is not claimed

The evidence does not prove that every plain `tl.dot` shape emits a particular `tcgen05` PTX instruction, that Gluon and `tl.*` expose identical surfaces, or that Triton has universal performance parity with specialized CUDA/CuTe kernels. Excluded dispatcher, test-only, CI, and host-routing PRs are not used as caveat or capability anchors.
