---
id: hw-pdl-gdc
title: "Programmatic Dependent Launch / Grid Dependency Control"
type: hardware
architectures: [sm100, sm100a, sm90, sm90a]
tags: [pdl, gdc]
confidence: verified
evidence_basis:
  - source_id: doc-cuda-13
    evidence_type: official-doc
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
  - source_id: pr-cutlass-2161
    evidence_type: upstream-code
related: [technique-persistent-kernels, hw-clc]
sources: [doc-cuda-13, doc-ptx-isa-sm100, pr-cutlass-2161]
aliases: [PDL, GDC, "programmatic dependent launch", "grid dependency control"]
blackwell_relevance: "PDL remains an explicit CUDA launch-and-kernel protocol on Blackwell; CUTLASS 4.5.0 separately defaults its SM100 GDC compile option on."
---

# Programmatic Dependent Launch

## Contract

Programmatic Dependent Launch (PDL) lets a secondary grid in the same CUDA stream become eligible to start before its prerequisite primary grid completes. It is available on compute capability 9.0 and later, including Hopper and Blackwell.

PDL creates an **opportunity** for overlap; it does not guarantee that the grids execute concurrently. The useful overlap is normally the primary's work after its launch trigger and the secondary's independent preamble before its dependency wait.

The CUDA protocol has three required roles:

1. Every primary CTA either executes `cudaTriggerProgrammaticLaunchCompletion()` or exits. After all CTAs satisfy that condition, the driver may schedule the secondary grid.
2. The host launches the secondary in the same stream with `cudaLaunchAttributeProgrammaticStreamSerialization` and `programmaticStreamSerializationAllowed = 1`.
3. Every secondary thread waits with `cudaGridDependencySynchronize()` before it consumes prerequisite results. The wait completes after the prerequisite grids finish and their memory operations are visible.

If the primary does not explicitly trigger, its CTAs implicitly satisfy the trigger only as they exit. Omitting the explicit trigger is correct but removes the intended primary-tail overlap.

## CUDA role skeleton

This minimal skeleton shows where each operation belongs; real kernels put independent and dependent work at the indicated points and add normal error checking:

```cuda
#include <cuda_runtime.h>

__global__ void primary() {
  // Produce everything needed before the secondary may be launched.
  if (threadIdx.x == 0) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
  // Primary tail that does not change data consumed by the secondary.
}

__global__ void secondary() {
  // Independent preamble may execute early.
  cudaGridDependencySynchronize();
  // Dependent reads begin only after this wait.
}

void launch_pdl(dim3 grid, dim3 block, cudaStream_t stream) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.stream = stream;

  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = &attr;
  cfg.numAttrs = 1;

  primary<<<grid, block, 0, stream>>>();
  cudaLaunchKernelEx(&cfg, secondary);
}
```

Do not place `cudaGridDependencySynchronize()` in the primary: it is the secondary-side wait, not the launch trigger. Do not replace the wait with an ordinary memory fence. PDL's wait supplies both prerequisite-grid completion and visibility for its dependent work.

## PTX mapping

CUDA lowers the two device roles to Grid Dependency Control instructions:

```ptx
// Primary CTA: makes designated dependent grids eligible after every CTA
// has issued this instruction or completed.
griddepcontrol.launch_dependents;

// Secondary thread: waits for in-flight prerequisite grids and visibility.
griddepcontrol.wait;
```

`griddepcontrol` was introduced in PTX ISA 7.8 and requires `sm_90` or newer. Repeating `launch_dependents` within one CTA has no additional effect after that CTA's first invocation. If a prerequisite uses `launch_dependents`, its dependent must use `griddepcontrol.wait` or an equivalent CUDA dependency wait for correct execution.

## CUDA support versus CUTLASS defaults

Blackwell does not make arbitrary back-to-back launches overlap automatically. CUDA applications still opt the secondary launch into programmatic stream serialization and implement the device-side trigger/wait protocol.

CUTLASS has a separate build choice. In CUTLASS 4.5.0, the CMake option `CUTLASS_ENABLE_GDC_FOR_SM100` defaults to `ON`, while the SM90 option is opt-in. That default only enables eligible CUTLASS code to emit its GDC wrappers; it is not a device-wide CUDA default and can be overridden by the build.

## When to use it

PDL can help only when all of these conditions hold:

- the secondary has enough independent preamble to overlap;
- the primary has useful tail work after every CTA reaches the trigger;
- the two grids have enough simultaneous resource headroom; and
- the saved launch/serialization time exceeds the protocol and occupancy costs.

Small-kernel chains, GEMM/epilogue sequences, and pipeline-parallel stages are candidates, not guaranteed wins. Profile an explicit non-PDL baseline and record GPU, clocks, launch shapes, stream/graph configuration, input sizes, warmup, repetitions, and the observed overlap timeline. Never rely on overlap for forward progress; CUDA documents it as opportunistic and warns that such reliance can deadlock.

## References

- [CUDA 13.0.2: Programmatic Dependent Launch](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization)
- [PTX ISA 9.0: `griddepcontrol`](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol)
- [CUTLASS 4.5.0: dependent kernel launch](https://github.com/NVIDIA/cutlass/blob/e406c186f510a15091cce01f782020ceb7ba8eb5/media/docs/cpp/dependent_kernel_launch.md)
- [Persistent kernels](../techniques/persistent-kernels.md)
- [Cluster Launch Control](clc.md)
