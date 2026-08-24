---
id: doc-cutlass-cute-dsl
title: "CUTLASS CuTe DSL Documentation"
url: https://github.com/NVIDIA/cutlass/tree/ad9bd53bdaec27a2e88053d57322ccf74efe525e/media/docs/pythonDSL
source_category: official-doc
architectures: [sm80, sm100, sm100a, sm90, sm90a, sm120]
tags: [cute-dsl, python, jit-compilation, gemm, tcgen05, tmem, tma, wgmma, mbarrier]
retrieved_at: 2026-07-20
---

# CuTe DSL Documentation (single-file compilation)

Compiled from the CUTLASS repo `media/docs/pythonDSL/` rst sources (commit `ad9bd53bdaec27a2e88053d57322ccf74efe525e`, generated 2026-07-20).
Python CuTe DSL docs only; C++ docs and empty API-reference stubs are excluded.

Editorial link repair: the upstream auto-tuning page's stale PTX fragment
`#tensorcore-5th-generation-family-instructions` is retargeted below to the
current PTX ISA 9.3 fifth-generation instruction section, verified 2026-08-19;
the surrounding upstream prose is unchanged.

## License

This page reproduces documentation from the [CUTLASS repository](https://github.com/NVIDIA/cutlass) (`media/docs/pythonDSL/` at commit `ad9bd53bdaec27a2e88053d57322ccf74efe525e`), including its image assets, which are licensed under the BSD 3-Clause License:

> Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
> SPDX-License-Identifier: BSD-3-Clause
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> 1. Redistributions of source code must retain the above copyright notice, this
> list of conditions and the following disclaimer.
>
> 2. Redistributions in binary form must reproduce the above copyright notice,
> this list of conditions and the following disclaimer in the documentation
> and/or other materials provided with the distribution.
>
> 3. Neither the name of the copyright holder nor the names of its
> contributors may be used to endorse or promote products derived from
> this software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
> FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
> DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
> SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
> CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
> OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Note: the CuTe DSL compiler components themselves are subject to a separate NVIDIA EULA (see the FAQ at the end of this page); the notice above covers the reproduced documentation text and images.

## Contents

1. Overview
2. Quick Start
3. Functionality
4. Introduction
5. Code Generation
6. Control Flow
7. JIT Argument Generation
8. JIT Argument: Layouts
9. Struct-like JIT Arguments
10. JIT Caching
11. JIT Compilation Options
12. JIT Types
13. Integration with Frameworks
14. Debugging with the DSL
15. IKET Profiling
16. Autotuning with the DSL
17. Educational Notebooks
18. Deprecation Policy
19. Compile with TVM FFI
20. Ahead-of-Time (AOT) Compilation
21. Talks and Presentations
22. Naming Conventions
23. MMA Programming Guides: WMMA (SM80)
24. MMA Programming Guides: WGMMA (SM90)
25. MMA Programming Guides: tcgen05 (SM100)
26. Limitations
27. FAQs
28. CuTe DSL API Changelog

---

<!-- source: overview.rst -->

## Overview

CUTLASS 4.x bridges the gap between productivity and performance for CUDA kernel development. By providing Python-based DSLs to the powerful CUTLASS C++ template library, it enables faster iteration, easier prototyping, and a gentler learning curve for high-performance linear algebra on NVIDIA GPUs.

Overall we envision CUTLASS DSLs as a family of domain-specific languages (DSLs). With the release of 4.0, we are releasing the first of these in CuTe DSL. This is a low level programming model that is fully consistent with CuTe C++ abstractions — exposing core concepts such as layouts, tensors, hardware atoms, and full control over the hardware thread and data hierarchy.

## Why CUTLASS DSLs?

While CUTLASS offers exceptional performance through its C++ template abstractions, the complexity can present challenges for many developers. CUTLASS 4.x addresses this by:

- **Simplifying metaprogramming**: Metaprogramming in Python is a lot more intuitive than with C++
- **Accelerating Iteration**: Rapid prototyping with familiar Python syntax and blazing fast compile times
- **Lowering Barriers**: Reduced learning curve for GPU programming concepts and consistency between CuTe C++ and DSL
- **Maintaining Performance**: Generated code leverages optimized CUTLASS primitives

Students can learn GPU programming concepts without the complexity of C++ templates. Researchers and performance engineers can rapidly explore algorithms, prototype, and tune kernels before moving to production implementations.

## Key Concepts and Approach

CUTLASS DSLs translate Python code into a custom intermediate representation (IR), which is then Just-In-Time (JIT) compiled into optimized CUDA kernels using MLIR and `ptxas`.

### Core CuTe DSL Abstractions

- **Layouts** – Describe how data is organized in memory and across threads.
- **Tensors** – Combine data pointers or iterators with layout metadata.
- **Atoms** – Represent fundamental hardware operations like matrix multiply-accumulate (MMA) or memory copy.
- **Tiled Operations** – Define how atoms are applied across thread blocks and warps (e.g., `TiledMma`, `TiledCopy`).

For more on CuTe abstractions, refer to the [CuTe C++ library documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/00_quickstart.md).

**Pythonic Kernel Expression**

Developers express kernel logic, data movement, and computation using familiar Python syntax and control flow.

The DSLs simplify expressing loop tiling, threading strategies, and data transformations using concise Python code.

**JIT Compilation**

Python kernels are compiled at runtime into CUDA device code using MLIR infrastructure and NVIDIA’s `ptxas` toolchain, enabling rapid iteration and interactive debugging.

## Relationship to CUTLASS C++

CUTLASS DSLs are not a replacement for the CUTLASS C++ library or its 2.x and 3.x APIs. Instead, it aims to be a high-productivity kernel authoring framework that shares all concepts with CUTLASS 3.x C++ API such as CuTe, pipelines, schedulers etc.

- **Performance**: Generated kernels aim to match CUTLASS C++ kernels in performance; however, some performance gaps may exist due to missing optimizations that have been added over the years to CUTLASS C++ and may be missing in the DSLs examples.
- **Library**: The CUTLASS DSLs do not currently ship with a full GEMM/Conv autotuning profiler or library interface akin to CUTLASS C++. Instead, it focuses on generating and autotuning individual kernel instances (for example: via tile size exploration) and via native integration DL frameworks that support auto-tuning.

## Getting Started

- Quick Start – Initial setup and installation.
- cute dsl – Overview of the typical development and workflow using CuTe DSL.
- cute dsl api – Refer to the full API documentation.
- Limitations – Understand current CuTe DSL constraints and differences from C++.
- FAQs – Common questions and known issues.

## Current Status & Roadmap

CuTe DSL is in public beta and actively evolving. Interfaces and features are subject to change as we improve the system.

### Upcoming Milestones

- Public release targeted for **Summer 2025**
- Expanded support for additional data types and kernel types
- Usability improvements: better error messages, debugging tools, and streamlined APIs
- Broader integration of CUTLASS primitives and features

For known issues and workarounds, please consult the Limitations and FAQs.

## Community & Feedback

We welcome contributions and feedback from the developer community!

You can:

- Submit bug reports or feature requests via our [GitHub Issues page](https://github.com/NVIDIA/cutlass/issues)
- Join the CUTLASS community on [Discord](https://discord.com/channels/1019361803752456192/1150868614921064590) to ask questions and share ideas
- Contribute examples, tutorials, or enhancements to the DSLs
- Report unclear or missing documentation
- Propose support for additional data types or kernel variants
- Help prioritize roadmap features by upvoting GitHub issues

Thank you for helping shape the future of CUTLASS DSLs!

---

<!-- source: quick_start.rst -->

## Quick Start Guide

### Compatibility Requirements

The CUTLASS DSL 4.4 release currently supports **Linux** and **Python 3.10 - 3.14** only.

Only Linux x86_64 and aarch64 are supported. Additional platform support will be added in future releases.

CUTLASS DSL supports the same NVIDIA driver version as the corresponding [CUDA Toolkit](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) (CUDA Toolkit 12.9 or CUDA Toolkit 13.1). Specifically, for 12.9, the driver version must be 575.51.03 or later.

### Installation

To ensure compatibility with the examples and code on [GitHub](https://github.com/NVIDIA/cutlass/tree/main), use the [setup.sh](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/setup.sh) file from the corresponding commit in the repository.

``` bash
git clone https://github.com/NVIDIA/cutlass.git

# For CUDA Toolkit 12.9:
./cutlass/python/CuTeDSL/setup.sh --cu12

# For CUDA Toolkit 13.1:
./cutlass/python/CuTeDSL/setup.sh --cu13
```

If you just want to try out the last known stable release of the CUTLASS DSL (may not be compatible with the latest examples and code), run:

``` bash
# For CUDA Toolkit 12.9:
pip install nvidia-cutlass-dsl

# For CUDA Toolkit 13.1:
pip install "nvidia-cutlass-dsl[cu13]"
```

The `nvidia-cutlass-dsl` wheel includes everything needed to generate GPU kernels.

### Recommended Dependencies

To run examples and begin development, we recommend installing:

``` bash
pip install torch jupyter mypy==1.19.1
```

### Recommended Python environment variables for jupyter notebooks

We recommend setting the following environment variable when running jupyter notebooks.

``` bash
export PYTHONUNBUFFERED=1
```

---

<!-- source: functionality.rst -->

## Functionality

For dependency version requirements, refer to the Quick Start section.

### Supported MMA Operations

**NVIDIA Ampere Architecture:**

- FP16 / BF16 tensor core instructions

**NVIDIA Hopper Architecture:**

- FP16 / BF16
- FP8

**NVIDIA Blackwell Architecture:**

- FP16 / BF16
- TF32
- I8
- F8

### Notable Limitations

For current constraints and unsupported features, refer to the Limitations section.

---

<!-- source: cute_dsl_general/dsl_introduction.rst -->

## Introduction

### Overview

CuTe DSL is a Python-based domain-specific language (DSL) designed for dynamic compilation of high-performance GPU kernels. It evolved from the C++ CUTLASS library and is now available as a decorator-based DSL.

Its primary goals are:

- **Zero-cost abstraction**, DSL is a zero-cost abstraction thanks to Hybrid DSL approach.
- **Consistent with CuTe C++**, allowing users to express GPU kernels with full control of the hardware.
- **JIT compilation** for both host and GPU execution.
- [DLPack](https://github.com/dmlc/dlpack) **integration**, enabling seamless interop with frameworks (e.g., PyTorch, JAX).
- **JIT caching**, so that repeated calls to the same function benefit from cached IR modules.
- **Native types and type inference** to reduce boilerplate and improve performance.
- **Optional lower-level control**, offering direct access to GPU backends or specialized IR dialects.

### Decorators

CuTe DSL provides two main Python decorators for generating optimized code via dynamic compilation:

1.  `@jit` — Host-side JIT-compiled functions
2.  `@kernel` — GPU kernel functions

Both decorators can optionally use a **preprocessor** that automatically expands Python control flow (loops, conditionals) into operations consumable by the underlying IR.

#### `@jit`

Declares JIT-compiled functions that can be invoked from Python or from other CuTe DSL functions.

**Decorator Parameters**:

- `preprocessor`:
  - `True` (default) — Automatically translate Python flow control (e.g., loops, if-statements) into IR operations.
  - `False` — No automatic expansion; Python flow control must be handled manually or avoided.

**Call-site Parameters**:

- `no_cache`:
  - `True` — Disables JIT caching, forcing a fresh compilation each call.
  - `False` (default) — Enables caching for faster subsequent calls.

#### `@kernel`

Defines GPU kernel functions, compiled as specialized GPU symbols through dynamic compilation.

**Decorator Parameters**:

- `preprocessor`:
  - `True` (default) — Automatically expands Python loops/ifs into GPU-compatible IR operations.
  - `False` — Expects manual or simplified kernel implementations.

**Kernel Launch Parameters**:

- `grid` Specifies the grid size as a list of integers.
- `block` Specifies the block size as a list of integers.
- `cluster` Specifies the cluster size as a list of integers.
- `smem` Specifies the size of shared memory in bytes (integer).
  - `None` (default) — Automatically calculates the kernel's shared memory usage via **utils.SmemAllocator**. Recommended unless manual control is required.
  - `int` — Manually specifies the size of shared memory in bytes.

**Additional Kernel Launch Parameters**:

- `fallback_cluster` Specifies the minimum-guaranteed cluster size. When set, `cluster` becomes the **preferred** size, enabling graceful degradation when hardware cannot satisfy the preferred dimensions.
  - `None` (default) — No fallback; `cluster` is used directly.
  - `list[int]` — Three-element list \[x, y, z\].
- `max_number_threads` Specifies the maximum thread count per block (**maxntid**).
  - `[0, 0, 0]` (default) — Auto-generate **reqntid** from `block`.
  - `list[int]` — Three-element list \[x, y, z\].
- `min_blocks_per_mp` Specifies the minimum blocks per multiprocessor (**minctasm**).
  - `0` (default) — No minimum occupancy hint.
  - `int` — Minimum number of blocks per multiprocessor.
- `use_pdl` Enables Programmatic Dependent Launch (PDL) to overlap dependent kernel launches in the same stream.
  - `False` (default) — PDL disabled.
  - `True` — PDL enabled.
- `cooperative` Enables cooperative kernel launch; all thread blocks launch cooperatively with grid-wide synchronization support.
  - `False` (default) — Standard kernel launch.
  - `True` — Cooperative kernel launch.
- `smem_merge_branch_allocs` Enables mutually exclusive control flow branches (sequentially executed if-else) to reuse the same shared memory.
  - `False` (default) — Shared memory is allocated additively across all branches (default CUDA C++ behavior).
  - `True` — Merge shared-memory allocations across branches (experimental feature, recommended for mega-kernels).
- `preferred_smem_carveout` Set per-kernel hint specifying what percentage of SM on-chip memory to reserve for shared memory vs. L1 cache.
  - `None` (default) — Auto calculate the percentage using formula `ceil_div(min_blocks_per_mp * smem * 100, max_smem_per_mp)` when **min_blocks_per_mp** is greater than 1
  - `int` — Override the auto-calculated percentage and manually set hint.

### Calling Conventions

| **Caller** | **Callee** | **Allowed** | **Compilation/Runtime** |
|----|----|----|----|
| Python function | `@jit` | ✅ | DSL runtime |
| Python function | `@kernel` | ❌ | N/A (error raised) |
| `@jit` | `@jit` | ✅ | Compile-time call, inlined |
| `@jit` | Python function | ✅ | Compile-time call, inlined |
| `@jit` | `@kernel` | ✅ | Dynamic call via GPU driver or runtime |
| `@kernel` | `@jit` | ✅ | Compile-time call, inlined |
| `@kernel` | Python function | ✅ | Compile-time call, inlined |
| `@kernel` | `@kernel` | ❌ | N/A (error raised) |

---

<!-- source: cute_dsl_general/dsl_code_generation.rst -->

## End-to-End Code Generation

### 1. Hybrid DSL: Python Metaprogramming, Structured GPU Code

CuTe DSL is a **hybrid DSL** that combines two compilation techniques: *AST rewrite* and *tracing*. This combination gives you the best of both worlds:

- **Program structure is preserved** — control flow (loops, branches) is captured via AST rewrite, compiling to proper structured code instead of flattened traces.
- **Python stays Python** — arithmetic and tensor operations are captured via tracing, so dynamic shapes, metaprogramming, and Python's rich expression language work naturally.

To understand why this matters, let's look at each technique.

#### 1.1 AST Rewrite

The function’s abstract-syntax tree is analysed **before** execution. Python control-flow (`for`/`while`, `if`/`else`) and built-ins are converted to structured intermediate representation (IR) constructs. Computation inside each region is left untouched at this stage.

*Advantages*

- Sees the entire program, so every branch and loop is preserved.
- Keeps loop structure intact for optimization such as tiling, vectorisation or GPU thread mapping.

*Disadvantages*

- Requires a well-defined Python subset that the rewriter understands.

#### 1.2 Tracing

The decorated function is executed once with *proxy* arguments; overloaded operators record every tensor operation that actually runs and produce a flat trace that is lowered to intermediate representation (IR).

*Advantages*

- Near-zero compile latency, ideal for straight-line arithmetic.
- No need to parse Python source, so it supports many dynamic Python features, and Python has many features.

*Disadvantages*

- Untaken branches vanish, so the generated kernel may be wrong for other inputs.
- Loops are flattened to the iteration count observed during tracing.
- Data-dependent control-flow freezes to a single execution path.

#### 1.3 The Hybrid Solution

As shown above, neither technique alone is sufficient—but together they complement each other perfectly.

**Why this works: GPU kernels are simple at runtime**

High-performance GPU kernels are structurally simple at runtime: they avoid deep call hierarchies, complex branching, and dynamic dispatch. However, *authoring* such kernels benefits greatly from Python's abstractions—classes, metaprogramming, and polymorphic patterns improve readability and maintainability.

The hybrid approach resolves this tension by evaluating Python abstractions at compile time while emitting simple, optimized code for runtime execution.

**How CuTe DSL divides the work:**

1.  **AST rewrite handles structure** — loops (`for`, `while`) and branches (`if`/`else`) are converted to structured intermediate representation (IR) *before* execution. This solves tracing's control-flow problem.
2.  **Tracing handles arithmetic** — inside each structured region, the tracer records tensor operations exactly as they execute. No need to model Python's complex semantics—just run Python and record what happens. This solves AST rewriting's complexity problem.

The result:

- Loops compile to real loops, not unrolled traces.
- All branches are preserved, even if not taken during tracing.
- Dynamic shapes, metaprogramming, and Python idioms work naturally.
- The rewriter only needs to understand control flow, not all of Python.

2\. CuTe DSL Compilation Flow: Meta-Stage to Object-Stage ------------------------------------------------------

CuTe DSL bridges Python and GPU hardware through a three-stage pipeline.

<figure class="align-center">
<img src="images/dsl_compilation.png" width="600" alt="The CuTe DSL compilation pipeline: Python source flows through AST preprocessing and interpreter-driven tracing to produce intermediate representation (IR), which is then lowered and compiled to device code." />
<figcaption aria-hidden="true">The CuTe DSL compilation pipeline: Python source flows through AST preprocessing and interpreter-driven tracing to produce <span>intermediate representation (IR)</span>, which is then lowered and compiled to device code.</figcaption>
</figure>

**Stage 1: Pre-Staging (Python AST)**

Before any code executes, the AST preprocessor rewrites the decorated function. It inserts *callbacks* around control-flow constructs—loops, branches, and function boundaries—so that program structure is captured explicitly rather than lost during execution.

**Stage 2: Meta-Stage (Python Interpreter)**

The rewritten function runs in the Python interpreter with proxy tensor arguments. As execution proceeds:

- Callbacks fire at control-flow boundaries, emitting structured intermediate representation (IR) (loops, branches, etc.).
- Tensor operations are traced: each operator invocation records the corresponding operation.
- Compile-time constants are *partially evaluated*—values known at JIT time fold directly into the intermediate representation (IR), enabling aggressive specialization.

The result is a complete representation of the kernel, with both high-level structure and low-level arithmetic intact.

**Stage 3: Object-Stage (Compiler Backend)**

The internal representation passes through a lowering pipeline:

1.  High-level operations are progressively lowered toward hardware-specific representations.
2.  Optimization passes (tiling, vectorization, memory promotion) reshape the code for the target architecture.
3.  The final code is translated to PTX/SASS (for NVIDIA GPUs) and assembled into a device binary.

At runtime, the compiled kernel is loaded and launched on the accelerator.

### 3. Meta-Programming vs Runtime: Two Worlds in One Function

A key insight for understanding CuTe DSL is that **your Python code runs twice**, in two very different contexts:

1.  **Meta-programming time (compilation)** — Python executes to *build* the kernel. This happens on the host CPU when you call a `@jit` function.
2.  **Runtime (execution)** — The compiled kernel runs on the GPU with actual tensor data.

This distinction determines what you can observe and when.

#### `print()` vs `cute.printf()`: Meta-Stage vs Object-Stage Output

CuTe DSL provides two ways to print values, each operating at a different stage:

- **Python's** `print()` — executes during the **meta-stage** (compilation). Use it to inspect what the compiler sees.
- `cute.printf()` — compiles into the kernel and executes at **runtime** on the GPU. Use it to observe actual tensor values during execution.

The following examples demonstrate how the same `result` variable appears differently depending on when and how you print it.

**Example 1: Dynamic variables (both** `a` **and** `b` **are runtime values)**

``` python
@cute.jit
def add_dynamicexpr(b: cutlass.Float32):
    a = cutlass.Float32(2.0)
    result = a + b
    print("[meta-stage] result =", result)          # runs at compile time
    cute.printf("[object-stage] result = %f\n", result)  # runs on GPU

add_dynamicexpr(5.0)
```

``` text
$> python myprogram.py
[meta-stage] result = <Float32 proxy>
[object-stage] result = 7.000000
```

At meta-stage, `result` is a proxy—its value is unknown until the kernel runs. At runtime, `cute.printf()` prints the actual GPU-computed value.

**Example 2: Compile-time constants (both** `a` **and** `b` **are Constexpr)**

``` python
@cute.jit
def add_constexpr(b: cutlass.Constexpr):
    a = 2.0
    result = a + b
    print("[meta-stage] result =", result)          # runs at compile time
    cute.printf("[object-stage] result = %f\n", result)  # runs on GPU

add_constexpr(5.0)
```

``` text
$> python myprogram.py
[meta-stage] result = 7.0
[object-stage] result = 7.000000
```

Both values are known at compile time, so Python evaluates `2.0 + 5.0 = 7.0` during tracing. The constant is baked into the compiled kernel.

**Example 3: Hybrid (** `a` **is dynamic,** `b` **is Constexpr)**

``` python
@cute.jit
def add_hybrid(b: cutlass.Constexpr):
    a = cutlass.Float32(2.0)
    result = a + b
    print("[meta-stage] result =", result)          # runs at compile time
    cute.printf("[object-stage] result = %f\n", result)  # runs on GPU

add_hybrid(5.0)
```

``` text
$> python myprogram.py
[meta-stage] result = <Float32 proxy>
[object-stage] result = 7.000000
```

The constant `b = 5.0` is folded in, but since `a` is dynamic, the result remains a proxy at meta-stage. The GPU computes the final answer at runtime.

#### Practical Implications

- **Use** `print()` **to debug your meta-program** — inspect shapes, strides, tile sizes, and compile-time decisions.
- **Constexpr parameters enable specialization** — the compiler can generate tighter code when values are known at JIT time.
- **Dynamic parameters preserve generality** — a single compiled kernel can handle varying input sizes without recompilation.

4\. CuTe DSL Code-Generation Modes ------------------------------

CuTe’s Python front-end combines the techniques above into **two mutually exclusive modes**, selectable with the `preprocessor` flag of the `@jit` decorator:

1\. Tracing mode `@jit(preprocess=False)` – tracing only. This results in the fastest compilation path and is recommended only for kernels that are guaranteed to be straight-line arithmetic. It suffers from all tracing limitations listed in the previous section.

2\. Preprocessor mode (**default**) `@jit(preprocess=True)` – **AST rewrite + tracing**. The AST pass captures every loop and branch, eliminating the correctness and optimisation problems of pure tracing; tracing then fills in the arithmetic. This hybrid “preprocessor” pipeline is unique to CuTe DSL and was designed specifically to overcome the disadvantages identified above.

<figure class="align-center">
<img src="images/dsl_modes.png" width="400" alt="Left: tracing mode records only the path that executed. Right: preprocessor mode emits structured intermediate representation (IR) for every branch and loop before tracing the arithmetic." />
<figcaption aria-hidden="true"><em>Left</em>: tracing mode records only the path that executed. <em>Right</em>: preprocessor mode emits structured <span>intermediate representation (IR)</span> for every branch and loop before tracing the arithmetic.</figcaption>
</figure>

---

<!-- source: cute_dsl_general/dsl_control_flow.rst -->

## Control Flow

### Overview

CuTe DSL walks Python's AST and converts each control-flow construct it finds into structured intermediate representation (IR). You can therefore write ordinary Python loops and branches while the compiler decides—statement by statement—whether to

- **evaluate at compile time** if it's a native Python control flow, or
- **emit intermediate representation (IR)** when the control flow is marked as dynamic.

Passing intermediate representation (IR) values to a native Python control flow will result in an error.

For a high-level discussion of the overall pipeline, see the code-generation overview.

### For Loops

CuTe DSL recognises three kinds of ranges for `for` loops:

- `range` – the Python built-in, always lowered to intermediate representation (IR)
- `cutlass.range` - Same as Python built-in `range`, but supports advanced unrolling and pipelining control
- `cutlass.range_constexpr` – unrolled at compile time

#### range(...)/cutlass.range(...)

Use when you *always* want a loop in the generated intermediate representation (IR), even if the inputs are Python values.

#### cutlass.range_constexpr(...)

Runs in the Python interpreter and is fully unrolled before code generation. All loop indices must be **Constexpr** (compile-time Python value).

**Example:**

``` python
@cute.jit
def control_flow_examples(bound: cutlass.Int32):
    n = 10

    # ✅ This loop is Python loop, evaluated at compile time.
    for i in cutlass.range_constexpr(n):
        cute.printf("%d\\n", i)

    # ✅ This loop is dynamic, even when bound is Python value.
    for i in range(n):
        cute.printf("%d\\n", i)

    # ❌ This loop bound is a dynamic value, not allowed in Python loop.
    # Should use `range` instead.
    for i in cutlass.range_constexpr(bound):
        cute.printf("%d\\n", i)

    # ✅ This loop is dynamic, emitted IR loop.
    for i in range(bound):
        cute.printf("%d\\n", i)

    # ✅ This loop is dynamic, emitted IR loop with unrolling
    for i in cutlass.range(bound, unroll=2):
        cute.printf("%d\\n", i)
```

#### Software Pipelining

Software pipelining is a technique used to optimize loops. Typically, this involves writing a prefetch loop and a main loop.

``` python
@cute.jit
def example():
    ...
    # build a circular buffer
    buffer = ...

    # prefetch loop
    for i in range(prefetch_stages):
        cute.copy(atom, gmem[i], buffer[i], ...)

    # main loop
    for i in range(bound):
        if i + prefetch_stages < bound:
            cute.copy(atom, gmem[i + prefetch_stages], buffer[(i + prefetch_stages) % total_stages], ...)

        use(buffer[i % total_stages])

    ...
```

This can be tedious to write and tune. CuTe DSL provides a loop attribute to ask the compiler to do this.

``` python
@cute.jit
def example():
    ...
    # build a circular buffer
    buffer = ...

    for i in cutlass.range(bound, prefetch_stages=prefetch_stages):
        # Compiler automatically handles the pipelining:
        # - Generates prefetch loop for initial stages
        # - In main loop, prefetches future data while using current data
        cute.copy(atom, gmem[i], buffer[i % total_stages], ...)
        use(buffer[i % total_stages])  # Uses data from previous iterations

    ...
```

Compiler will automatically generate the prefetch loop with `prefetch_stages` iterations and a corresponding main loop.

This feature is experimental and only supported on sm90 and above.

### If-Else Statements

Standard Python `if`/`elif`/`else` is supported.

- **Predicate without annotation** → lowered to intermediate representation (IR).
- **Predicate annotated with \`cutlass.const_expr\`** → evaluated at compile time.

**Example:**

``` python
@cute.jit
def main(const_var: cutlass.Constexpr, dynamic_var: cutlass.Int32):
    # ✅ This branch is Python branch, evaluated at compile time.
    if cutlass.const_expr(const_var):
        cute.printf("Const branch\\n")
    else:
        cute.printf("Const else\\n")

    # ✅ This branch is dynamic branch, emitted IR branch.
    if dynamic_var == 10:
        cute.printf("Dynamic True\\n")
    else:
        cute.printf("Dynamic False\\n")

    # ❌ Using a dynamic value with `cutlass.const_expr` is not allowed.
    if cutlass.const_expr(dynamic_var == 10):
        cute.printf("Bound is 10\\n")
```

### While Loops

Standard Python `while` is supported.

- **Condition without annotation** → lowered to intermediate representation (IR).
- **Condition annotated with \`cutlass.const_expr\`** → evaluated at compile time.

**Example:**

``` python
@cute.jit
def main(dynamic_var: cutlass.Int32):
    n = 0

    # ✅ This is Python while loop, evaluated at compile time.
    while cutlass.const_expr(n < 10):
        cute.printf("Const branch\\n")
        n += 1

    # ✅ This is dynamic while loop, emitted IR while loop.
    while dynamic_var == 10:
        cute.printf("Dynamic True\\n")
        n += 1

    # ❌ Using a dynamic value with `cutlass.const_expr` is not allowed.
    while cutlass.const_expr(n < dynamic_var):
        n += 1
```

### Summary of Control Flow behavior

| **Control Flow** | **Run time evaluation** | **Compile time evaluation** |
|----|----|----|
| if cutlass.const_expr() | ❌ | ✅ |
| if pred | ✅ | ❌ |
| while cutlass.const_expr() | ❌ | ✅ |
| while pred | ✅ | ❌ |
| for i in cutlass.range_constexpr() | ❌ | ✅ |
| for i in range() | ✅ | ❌ |
| for i in cutlass.range() (support advanced unrolling and pipelining) | ✅ | ❌ |

### Compile-Time Metaprogramming

Mix compile-time constructs with normal CuTe DSL code to generate specialised kernels without runtime overhead. A compile-time flag can, for example, toggle an optional **ReLU** epilogue:

``` python
@cute.kernel
def gemm(..., do_relu: cutlass.Constexpr):
    # main GEMM work
    ...
    if cutlass.const_expr(do_relu):    # compile-time guard
        # ReLU code is emitted only when do_relu is True
        ...
```

``` text
gemm(..., False)   # ReLU is omitted from the generated |IR|
gemm(..., True)    # ReLU is included
```

#### Limitations of Dynamic Control Flow

- Early-exit `break`, `continue`, `pass` or raising exception from control flow body are not yet supported.
- Operations in the control flow body are traced only when tracing is active in that region.
- Values originating in control flow body are not available outside the control flow.
- Changing type of a variable in control flow body is not allowed.

**Example:**

``` python
@cute.jit
def control_flow_negative_examples(predicate: cutlass.Boolean):
    n = 10

    # ❌ This loop is dynamic, early-exit isn't allowed.
    for i in range(n):
        if i == 5:
            break         # Early-exit

    if predicate:
        val = 10
        # ❌ return from control flow body is not allowed.
        return
        # ❌ Raising exception from control flow body is not allowed.
        raise ValueError("This is not allowed")
        # ❌ Using pass in control flow body is not allowed.
        pass

    # ❌ val is not available outside the dynamic if
    cute.printf("%d\\n", val)

    if predicate:
        # ❌ Changing type of a variable in control flow body is not allowed.
        n = 10.0
```

---

<!-- source: cute_dsl_general/dsl_jit_arg_generation.rst -->

## JIT Function Argument Generation

### In a nutshell

When using the `@jit` or `@kernel` decorators to define a JIT-compiled function, the arguments to the function are traced to determine the JIT function's signature. CuTe DSL provides a Pythonic way to write the arguments for JIT function as one normally would in Python, and the CuTe DSL will take care of the rest for you.

Specifically, CuTe DSL honors following when generating the JIT function's arguments:

- JIT function arguments are assumed to be **dynamic arguments** by default.
- If an argument is explicitly type annotated with `cutlass.Constexpr`, it is treated as a **compile-time constant**.
- If type annotation is provided, CuTe DSL validates the argument type at compile time for **type safety**.
- CuTe DSL provides **runtime checkable protocols** (`JitArgument` and `DynamicExpression`) for generating JIT function arguments for customized types.

More details below for each of the above.

### Static argument vs. Dynamic argument

CuTe DSL supports both static and dynamic arguments for JIT functions.

1.  **Static arguments** hold values that are known at compile time. It is not included in the generated JIT function signature.
2.  **Dynamic arguments** hold values that are only known at runtime.

By default, CuTe DSL assumes dynamic arguments and tries to infer the argument types from the call-site argument types. An explicit type annotation `cutlass.Constexpr` can be used to specify a static argument.

``` python
import cutlass
import cutlass.cute as cute

@cute.jit
def foo(x: cutlass.Int32, y: cutlass.Constexpr):
    print("x = ", x)        # Prints x = ?
    print("y = ", y)        # Prints y = 2
    cute.printf("x: {}", x) # Prints x: 2
    cute.printf("y: {}", y) # Prints y: 2

foo(2, 2)
```

In the example above, `x` is a dynamic argument with type cutlass.Int32 and `y` is a static argument.

With the `cutlass.Constexpr` annotation, a more sophisticated uses case of static argument in the JIT functions can be something like:

``` python
import cutlass
import cutlass.cute as cute

@cute.kernel
def kernel(
    self,
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    tma_atom_c: Optional[cute.CopyAtom],
    mC_mnl: cute.Tensor,
    cluster_layout_vmnk: cute.Layout,
    a_smem_layout_staged: cute.ComposedLayout,
    b_smem_layout_staged: cute.ComposedLayout,
    c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
    epi_tile: cute.Tile,
    epilogue_op: cutlass.Constexpr,
):
    ...

    # Perform epilogue op on accumulator and convert to C type
    acc_vec = tTR_rAcc.load()
    acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
    tTR_rC.store(acc_vec)
```

In this example, `epilogue_op` is a static argument in the JIT kernel where the argument is used for the epilogue fusion. Upon calling the kernel, an elementwise lambda function can be passed in as the `epilogue_op` argument. For example, a ReLU can be applied for epilogue fusion by simply setting the `epilogue_op` to `lambda x: cute.where(x > 0, x, cute.full_like(x, 0))`

Refer to the [Blackwell dense GEMM example](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py) for a complete example.

> [!NOTE]
> For the per-thread/partition naming convention used above (`tTR_rAcc`, `tTR_rC`, and related tokens such as `tAgA`, `bSG_sC`, `tQgQ_qdl`, …), see the cute dsl naming conventions.

### Type safety

CuTe DSL makes good use of type annotation in JIT function signature and validates the JIT function argument types at compile time for **type safety**.

``` python
import cutlass
import cutlass.cute as cute
import numpy as np

@cute.jit
def foo(x: cute.Tensor, y: cutlass.Float16):
    ...

a = np.random.randn(10, 10).astype(np.float16)
b = 32

foo(a, b)
foo(b, a)  # This will fail at compile time due to type mismatch
```

The type safety check helps catch the type mismatch issue early at the compile time with clear error message to avoid tricky runtime errors which is usually more expensive to debug. In the example above, the second call to `foo` will fail at compile time due to the type mismatch with a clear error message:

    cutlass.base_dsl.common.DSLRuntimeError: DSLRuntimeError: expects argument #1 (a) to be <class 'cutlass.cute.typing.Tensor'>, but got <class 'int'>

### JIT function arguments with customized types

CuTe DSL supports customized types for JIT function arguments by providing two runtime checkable protocols:

- `JitArgument` which is used for host JIT functions to be called from Python.
  - `__c_pointers__`: Generate a list of ctypes pointers for the current object.
  - `__get_mlir_types__`: Generate a list of MLIR types for the current object.
  - `__new_from_mlir_values__`: Create a new object from MLIR values.

- `DynamicExpression` which is used for device JIT functions to be called from the host JIT functions.
  - `__extract_mlir_values__`: Generate a dynamic expression for the current object.
  - `__new_from_mlir_values__`: Create a new object from MLIR values.

Refer to [typing.py](https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL/base_dsl/typing.py) for more details on these protocol APIs.

Depending on different cases of the customized types, CuTe DSL provides easy ways to adopt customized types for JIT function arguments.

#### 1. Direct protocol implementation in customized types

One way is to implement the protocol methods directly in the customized types to enable the protocol based JIT function argument generation.

``` python
import cutlass
import cutlass.cute as cute

# Customized type that implements the DynamicExpression protocol
class MyDynamicExpression:
    def __init__(self, tensor, offset):
        self._tensor = tensor # Dynamic argument
        self._offset = offset # Dynamic argument

    def __extract_mlir_values__(self):
        return [self._tensor.__extract_mlir_values__(), self._offset.__extract_mlir_values__()]

    def __new_from_mlir_values__(self, values):
        return MyDynamicExpression(values[0], values[1])

@cute.kernel
def my_kernel(x: MyDynamicExpression):
    ...
```

In the example above, the `MyDynamicExpression` implements the `DynamicExpression` protocol and CuTe DSL will generate the JIT function arguments for the JIT kernel `my_kernel` based on the protocol methods.

#### 2. Adaptor based protocol implementation for customized types

For the case where directly changing the customized types to implement the protocol is not feasible, CuTe DSL provides adaptor based approach to adapt the customized types for JIT function argument generation.

The JIT function argument adaptor is a callable object that implements the desired protocol methods for the registered customized types. This way, CuTe DSL automatically queries the JIT argument adaptor registry to generate the JIT function arguments for the given customized types.

``` python
@cutlass.register_jit_arg_adapter(MyFrameworkObject)
class MyFrameworkObjectAdapter:
    """
    Convert a 3rd party framework object to a JIT function argument with JitArgument protocol
    """

    def __init__(self, arg):
        self._arg = arg

    def __c_pointers__(self):
        # Convert the framework object to a C-ABI compatible object
        # thru its C-ABI interface
        return [self._arg.get_cabi_pointer()]

    def __get_mlir_types__(self):
        # Return the list of MLIR types the framework object represents
        return [self._arg.get_data().mlir_type]

    def __new_from_mlir_values__(self, values):
        # Convert the MLIR values back to the framework object
        return MyFrameworkObject(values[0])
```

In this example, the `MyFrameworkObjectAdapter` implements an adaptor class which bridges the CuTe DSL and the 3rd party framework type `MyFrameworkObject`. The registration is done by just decorating the adaptor with `cutlass.register_jit_arg_adapter` for the customized type. With the registered adaptor, CuTe DSL will automatically use the adaptor to generate the JIT function arguments for `MyFrameworkObject` typed arguments.

---

<!-- source: cute_dsl_general/dsl_dynamic_layout.rst -->

## Static vs Dynamic layouts

### Static Layout

When integrating with popular deep learning frameworks, one question is how to deal with the layout of the converted `cute.Tensor`. For example, when converting a `torch.Tensor` to a `cute.Tensor`, the shape of the `torch.Tensor` is honored for the layout of `cute.Tensor`.

``` python
import torch
import cutlass
from cutlass.cute.runtime import from_dlpack

@cute.jit
def foo(tensor):
    print(f"tensor.layout: {tensor.layout}")  # Prints tensor layout at compile time
    cute.printf("tensor: {}", tensor)         # Prints tensor values at runtime
```

In this example, we define a JIT function `foo` that takes a `cute.Tensor` as input and prints its layout. Note that Python print is used to print the layout at compile time. This works fine for static layout whose value is known at compile time.

Now let's try to run the JIT function `foo` with different shapes of the input `torch.Tensor`.

``` python
a = torch.tensor([1, 2, 3], dtype=torch.uint16)
a_pack = from_dlpack(a)
compiled_func = cute.compile(foo, a_pack)
compiled_func(a_pack)
```

Here we first convert a 1D `torch.Tensor` with 3 elements to a `cute.Tensor` using `from_dlpack`. Then we compile the JIT function `foo` with the converted `cute.Tensor` and call the compiled function.

    tensor.layout: (3):(1)
    tensor: raw_ptr(0x00000000079e5100: i16, generic, align<2>) o (3):(1) =
    ( 1, 2, 3 )

It prints `(3):(1)` for the layout because the converted `cute.Tensor` has a static layout with shape `(3)` which is the shape of the `a`.

Now if we call the compiled function with a different shape of the input `torch.Tensor`, it would result in an unexpected result at runtime due to the mismatch of the type since `compiled_func` expects a `cute.Tensor` with layout `(3):(1)` while `b` has shape `(5)`.

``` python
b = torch.tensor([11, 12, 13, 14, 15], dtype=torch.uint16)
b_pack = from_dlpack(b)
compiled_func(b_pack)  # ❌ This results in an unexpected result at runtime due to type mismatch
```

Following is the output which is unexpected due to the type mismatch.

    tensor: raw_ptr(0x00000000344804c0: i16, generic, align<2>) o (3):(1) =
    ( 11, 12, 13 )

To fix that, we would have to trigger another code generation and compilation for the new shape for `b`.

``` python
compiled_func_2 = cute.compile(foo, b_pack)  # This would trigger another compilation
compiled_func_2(b_pack)                      # ✅ Now this works fine
```

As shown in the example above, with the newly compiled `compiled_func_2`, we can pass in `b_pack` to the compiled JIT function `compiled_func_2`.

    tensor.layout: (5):(1)
    tensor: raw_ptr(0x0000000034bb2840:: i16, generic, align<2>) o (5):(1) =
    ( 11, 12, 13, 14, 15 )

Now it recompiles and prints the values of `b` correctly.

It's obvoius that we need distinct codes generated and compiled for different static layout. In this case, one for layout `(3):(1)` and the other for layout `(5):(1)`.

### Dynamic Layout

In order to avoid generating and compiling multiple times for different shapes of the input `torch.Tensor`, CuTe DSL provides a way to generate and compile JIT function with dynamic layout.

To get dyanmic layout of the `cute.Tensor`, a `torch.Tensor` object can be passed into the JIT function directly which instructs CuTe DSL to call `cute.mark_layout_dynamic` automatically on the converted `cute.Tensor` per the leading dimension of the layout.

``` python
import torch
import cutlass
from cutlass.cute.runtime import from_dlpack

@cute.jit
def foo(tensor):
    print(tensor.layout)  # Prints (?,?):(?,1) for dynamic layout

a = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint16)
compiled_func = cute.compile(foo, a)
compiled_func(a)

b = torch.tensor([[11, 12], [13, 14], [15, 16]], dtype=torch.uint16)
compiled_func(b)  # Reuse the same compiled function for different shape
```

In the example above, a single compilation of the JIT function `foo` is reused for different shapes of the input `torch.Tensor`. This is possible because the converted `cute.Tensor` has a dynamic layout `(?,?):(?,1)` which is compatible with the shape of the input `torch.Tensor` of both calls.

Alternatively, for compact layout, `cute.mark_compact_shape_dynamic` can be called for a finer-grained control to specify the mode of the layout for dynamic and the divisibility constraint for the dynamic dimension.

Refer to Integration with Frameworks for more details on `from_dlpack`, `mark_layout_dynamic`, and `mark_compact_shape_dynamic`.

### Static Layout vs. Dynamic Layout

Per the previous sections, we have seen that static layout leads to distinct JIT code generations while dynamic layout leads to a single compilation for different shapes.

That said, creating JIT function with static layout is useful when the use cases targeting input data with fixed shapes. Since more information is available at compile time, the compiler would be able to kick in optimizations that otherwise would not be possible for the code generated for dynamic layout.

On the other hand, dynamic layout would be more flexible for the cases where the input data has varying shapes. This provides more scalability of the generated code to deal with varying input data of different shapes.

### Programming with Static and Dynamic Layout

CuTe DSL provides intuitive way to program with static and dynamic layout in the codes.

``` python
import torch
import cutlass
from cutlass.cute.runtime import from_dlpack

@cute.jit
def foo(tensor, x: cutlass.Constexpr[int]):
    print(cute.size(tensor))  # Prints 3 for the 1st call
                              # Prints ? for the 2nd call
    if cute.size(tensor) > x:
        cute.printf("tensor[2]: {}", tensor[2])
    else:
        cute.printf("tensor size <= {}", x)

a = torch.tensor([1, 2, 3], dtype=torch.uint16)
foo(from_dlpack(a), 3)   # First call with static layout

b = torch.tensor([1, 2, 3, 4, 5], dtype=torch.uint16)
foo(b, 3)                # Second call with dynamic layout
```

In this example, the JIT function `foo` is compiled with a static layout `(3):(1)` for the first call, which means the size of the tensor is known at compile time. CuTe DSL makes good use of this and automatically handles the if condition at the compile time. Hence the generated codes are efficient without the if condition at all.

For the second call, the JIT function `foo` is compiled with a dynamic layout `(?):(1)` hence the tensor size is only evaluated at runtime. CuTe DSL automatically generates the code to handle the dynamic layout and the if condition at runtime.

The same applies to loop as well:

``` python
@cute.jit
def foo(tensor, x: cutlass.Constexpr[int]):
    for i in range(cute.size(tensor)):
        cute.printf("tensor[{}]: {}", i, tensor[i])

a = torch.tensor([1, 2, 3], dtype=torch.uint16)
foo(from_dlpack(a), 3)   # First call with static layout

b = torch.tensor([1, 2, 3, 4, 5], dtype=torch.uint16)
foo(b, 3)                # Second call with dynamic layout
```

With the static layout in the first call, CuTe DSL is able to fully unroll the loop at compile time. While in the second call, the generated codes will have the loop executed at runtime based on the dynamic layout.

With the single JIT function implementation, CuTe DSL is able to handle control-flow constructs and automatically generate the optimized codes for different cases. This is all possible because CuTe DSL is able to walk the Python AST and convert each control-flow construct it finds accordingly.

Please refer to Control Flow for more details.

---

<!-- source: cute_dsl_general/dsl_struct_types.rst -->

## Struct-like JIT Arguments

CuTe DSL supports several struct-like Python types as JIT function arguments. Each provides a different trade-off between mutability, syntax convenience, and low-level control.

### Overview

| Type | Mutable fields? | Notes |
|----|----|----|
| `typing.NamedTuple` | **No** | Tuple subclass — fields fixed at construction. Flattened field-by-field through the pytree system. |
| `@native_struct` | **Yes** | Generates an LLVM struct type. `llvm.insertvalue` replaces field values in-place. |
| `@dataclass(frozen=True)` | **No** | Frozen dataclass — treated as a read-only pytree container, similar to `NamedTuple`. |

### NamedTuple

A `typing.NamedTuple` whose fields are DSL scalar types (`Int32`, `Float32`, etc.) can be passed directly to `@cute.jit` / `cute.compile` without any boilerplate or protocol implementation.

**How it works.** NamedTuples are registered as pytree containers in the DSL tree system. Each field is flattened individually through the existing DSL type paths and reconstructed by calling the NamedTuple constructor on the way into the kernel body. Field attribute access (`tup.a`, `tup.b`, …) works exactly as in native Python.

#### Basic usage

``` python
from typing import NamedTuple
import cutlass
import cutlass.cute as cute

class Vec3(NamedTuple):
    x: cutlass.Int32
    y: cutlass.Int32
    z: cutlass.Int32

@cute.jit
def print_vec(v: Vec3):
    cute.printf("x=%d y=%d z=%d\n", v.x, v.y, v.z)

v = Vec3(x=cutlass.Int32(1), y=cutlass.Int32(2), z=cutlass.Int32(3))
cute.compile(print_vec, v)(v)
```

#### Control flow on fields

Fields are DSL values inside the kernel, so they work in `if`/`else` branches and `for` loops:

``` python
@cute.jit
def clamp_positive(v: Vec3, out: cute.Tensor):
    """Write max(field, 0) for each component."""
    out[0] = cutlass.Int32(0) if v.x < cutlass.Int32(0) else v.x
    out[1] = cutlass.Int32(0) if v.y < cutlass.Int32(0) else v.y
    out[2] = cutlass.Int32(0) if v.z < cutlass.Int32(0) else v.z

@cute.jit
def triangular_sum(v: Vec3, out: cute.Tensor):
    """Sum 0..v.x-1 into out[0], and so on."""
    s = cutlass.Int32(0)
    for i in range(v.x):
        s = s + i
    out[0] = s
```

#### Creating a new NamedTuple value inside the kernel

NamedTuple fields are **immutable** — the same constraint as native Python tuples. Assigning `tup.x = ...` inside a kernel raises `AttributeError`. To "update" a field, construct a replacement NamedTuple:

``` python
@cute.jit
def scale(v: Vec3, factor: cutlass.Int32, out: cute.Tensor):
    # Construct a new Vec3 with all fields scaled
    scaled = Vec3(x=v.x * factor, y=v.y * factor, z=v.z * factor)
    out[0] = scaled.x
    out[1] = scaled.y
    out[2] = scaled.z
```

### `@native_struct`

Use `@native_struct` when kernel logic needs to **accumulate into or update** a struct field. Unlike NamedTuple, fields are mutable: each write generates an `llvm.insertvalue` to replace the field in the underlying LLVM struct.

``` python
import cutlass
import cutlass.cute as cute

@cute.native_struct
class Accumulator:
    total: cutlass.Int32
    count: cutlass.Int32

@cute.jit
def accumulate(acc: Accumulator, values: cute.Tensor, n: cutlass.Int32):
    for i in range(n):
        acc.total = acc.total + values[i]
        acc.count = acc.count + cutlass.Int32(1)
```

`@native_struct` also supports:

- `zero_init=False` — initialize with `llvm.mlir.undef` instead of zero.
- `packed=True` — create a packed LLVM struct (no padding between fields).
- `Constexpr` fields — excluded from the native struct and passed as ordinary Python values.

### Choosing the right type

| Use case | Recommended type |
|----|----|
| Read-only config / parameters passed into a kernel | `NamedTuple` or `@dataclass(frozen=True)` |
| Accumulator or running state updated inside a kernel | `@native_struct` |
| Want Python-native immutable semantics (hashable, unpackable) | `NamedTuple` |
| Need fine-grained LLVM struct control (packing, zero-init) | `@native_struct` |

### See also

- JIT Argument Generation — overview of JIT function argument protocols
- JIT Argument: Layouts — passing `Layout` objects as JIT arguments

---

<!-- source: cute_dsl_general/dsl_jit_caching.rst -->

## JIT Caching

### Zero Compile and JIT Executor

Zero Compile is a feature that enables explicit kernel compilation on demand through `cute.compile`. When `cute.compile` is called, it compiles the kernel and returns a JIT Executor instance. This JIT Executor instance can be cached and reused directly for subsequent executions without compiling the kernel again.

The JIT Executor is a component that independently executes compiled code. It can be created either through `cute.compile` or implicit compilation. The JIT Executor instance behaves like a callable object to execute the compiled code. Each JIT Executor instance maintains a single compiled host function.

It encompasses all necessary execution components:

- Host function pointer and its MLIR execution engine
- CUDA modules (optional)
- Argument specifications defining how Python arguments are converted to C ABI-compatible types. Note that arguments with the `cutlass.Constexpr` hint are excluded from argument specifications since they are evaluated at compile time rather than runtime.

For example, in the following code, `print_result` is a `cutlass.Constexpr` value that is **NOT** evaluated at runtime:

``` python
import cutlass.cute as cute

@cute.jit
def add(a, b, print_result: cutlass.Constexpr):
   if print_result:
      cute.printf("Result: %d\n", a + b)
   return a + b

jit_executor = cute.compile(add, 1, 2, True)

jit_executor(1, 2) # output: ``Result: 3``
```

The JIT Executor ensures all components are properly initialized and loaded after compilation.

For example, all CUDA modules are loaded (via `cuModuleLoad`) and kernel function pointers are extracted (via `cuModuleGetFunction`).

When calling a JIT Executor instance, it:

- Parses Python runtime arguments and converts them to C ABI-compatible types according to argument specifications
- Invokes the host function with the converted arguments

#### Custom Caching with `cute.compile`

`cute.compile` bypasses caching in CuTe DSL and always performs compilation, returning a fixed JIT Executor instance. This allows implementing custom caching strategies as shown below:

``` python
@cute.jit
def add(b):
   return a + b

# Define a custom cache
custom_cache = {}

a = 1
compiled_add_1 = cute.compile(add, 2)
custom_cache[1] = compiled_add_1
compiled_add_1(2) # result = 3

a = 2
compiled_add_2 = cute.compile(add, 2)
custom_cache[2] = compiled_add_2
compiled_add_2(2) # result = 4

# Use the custom cache
custom_cache[1](2) # result = 3
custom_cache[2](2) # result = 4
```

### Cache in CuTe DSL

By default, cache in CuTe DSL is implicitly enabled to avoid recompilation when kernels are called repeatedly without changes.

The cache is implemented as a map storing compiled JIT Executor instances within CuTe DSL.

The cache key combines hashes of:

- MLIR bytecode of the MLIR program generated by CuTe DSL
- All CuTe DSL Python source files
- All CuTe DSL shared libraries
- All CuTe DSL environment variables

The cache value is a compiled JIT Executor instance.

On a cache hit, compilation is skipped and the cached JIT Executor instance is reused.

On a cache miss, the kernel is compiled and the new JIT Executor instance is stored in the cache.

Here is an example demonstrating automatic caching of the `add` kernel:

``` python
# Global variable
a = 1

@cute.jit
def add(b):
   return a + b

# Cache is empty at beginning

# First call: cache miss triggers compilation
result = add(2) # result = 3
# Cache now has one instance

# Second call: cache hit reuses cached JIT Executor
result = add(2) # result = 3

a = 2
# Third call: cache miss due to changed IR code triggers recompilation
result = add(2) # result = 4
# Cache now has two instances
```

The cache can be serialized to files for subsequent runs. After serialization, compiled MLIR bytecode is stored in file. The cache directory is `/tmp/{current_user}/cutlass_python_cache`. During compilation, the cache loads the corresponding kernel from file (if it exists) into memory as needed, and after compilation, it saves any newly compiled executables back to file.

Note that for efficiency, the default cache directory is located in a temporary folder. However, this location is not persistent, it may be cleared by the system (for example, during a reboot or disk space cleanup). If you wish to preserve the cache across sessions, set the `CUTE_DSL_CACHE_DIR` environment variable to point to a persistent directory.

The following environment variables control file caching:

``` bash
# Disable file caching while keeping in-memory cache available, defaults to False.
export CUTE_DSL_DISABLE_FILE_CACHING=True

# Cache directory, defaults to /tmp/{current_user}/cutlass_python_cache.
export CUTE_DSL_CACHE_DIR=/home/user/local_cutlass_python_cache/dense_gemm_cache/
```

#### Limitations

The intention of caching is to reduce the host launch overhead before each execution. As above example shows, the consistency between the original Python code and the MLIR program is hard to maintain because of the impact of dynamic factors such as global variables. Therefore, the MLIR program **MUST** always be generated to verify that the kernel content matches what was previously built.

For optimal host launch latency, we recommend using above custom caching method with `cute.compile`.

---

<!-- source: cute_dsl_general/dsl_jit_compilation_options.rst -->

## JIT Compilation Options

### JIT Compilation Options Overview

When compiling a JIT function using CuTe DSL, you may want to control various aspects of the compilation process, such as optimization level, or debugging flags. CuTe DSL provides a flexible interface for specifying these compilation options when invoking `cute.compile`.

Compilation options allow you to customize how your JIT-compiled functions are built and executed. This can be useful for:

- Enabling or disabling specific compiler optimizations
- Generating debug information for troubleshooting

These options can be passed as keyword arguments to `cute.compile` or set globally for all JIT compilations. The available options and their effects are described in the following sections, along with usage examples to help you get started.

The CuTe DSL provides multiple ways to specify compilation options - either by specifying additional arguments to `cute.compile` or by using a more Pythonic approach with separate Python types for `cute.compile`.

### `cute.compile` Compilation Options as strings

You can provide additional compilation options as a string when calling `cute.compile`. The CuTe DSL uses `argparse` to parse these options and will raise an error if any invalid options are specified.

| **Option** | **Description** | **Default** | **Type** |
|----|----|----|----|
| `opt-level` | Optimization level of compilation. The higher the level, the more optimizations are applied. The valid value range is \[0, 3\]. | 3 (highest level of optimization) | int |
| `enable-assertions` | Enable host and device code assertions. | False | bool |
| `keep-cubin` | Keep the generated CUBIN file. | False | bool |
| `keep-ptx` | Keep the generated PTX file. | False | bool |
| `ptxas-options` | The options to pass to the PTX Compiler library. | "" | str |
| `generate-line-info` | Generate line information for debugging. | False | bool |
| `gpu-arch` | The GPU architecture to compile for. | "" | str |
| `enable-tvm-ffi` | Enable Apache TVM FFI. | False | bool |

You can use the following code to specify compilation options:

``` python
jit_executor_with_opt_level_2 = cute.compile(add, 1, 2, options="--opt-level 2")
jit_executor_with_opt_level_1 = cute.compile(add, 1, 2, options="--opt-level 1")
jit_executor_with_enable_assertions = cute.compile(add, 1, 2, options="--enable-assertions")
jit_executor_with_keep_cubin = cute.compile(add, 1, 2, options="--keep-cubin")
jit_executor_with_keep_ptx = cute.compile(add, 1, 2, options="--keep-ptx")
jit_executor_with_ptxas_options = cute.compile(add, 1, 2, options="--ptxas-options '--opt-level=2'")
```

### `cute.compile` Compilation Options as separate Python types

Alternatively, you can also use a more Pythonic way to specify compilation options with separate Python types. Compilation options can be programmatically composed using tuple and passed to `cute.compile` separately.

``` python
from cutlass.cute import OptLevel, EnableAssertions, GenerateLineInfo, KeepCUBIN, KeepPTX

my_debugging_options = (OptLevel(1), EnableAssertions, GenerateLineInfo, KeepCUBIN, KeepPTX)
compiled_kernel_1 = cute.compile[my_debugging_options](my_kernel_1, ...)
compiled_kernel_2 = cute.compile[my_debugging_options](my_kernel_2, ...)
```

This approach causes invalid options to raise errors immediately, making it much easier to detect typos when specifying multiple options. Notebly, boolean options are automatically converted to True instances of the option type for convenience.

``` python
jit_executor_with_opt_level_2 = cute.compile[OptLevel(2)](add, 1, 2)
jit_executor_with_opt_level_1 = cute.compile[OptLevel(1)](add, 1, 2)
jit_executor_with_enable_assertions = cute.compile[EnableAssertions](add, 1, 2)
jit_executor_with_keep_cubin = cute.compile[KeepCUBIN](add, 1, 2)
jit_executor_with_keep_ptx = cute.compile[KeepPTX](add, 1, 2)
jit_executor_with_ptxas_options = cute.compile[PtxasOptions("--opt-level=2")](add, 1, 2)
```

---

<!-- source: cute_dsl_general/types.rst -->

## Types

### Overview

CuTe DSL provides a set of core types that form the foundation of tensor layout algebra and GPU programming. These types enable precise control over memory layout, data representation, and tensor operations. This document covers the key types available in `cutlass.cute.core`.

### Core Numeric Types

#### IntValue

`IntValue` is an internal representation of constrained integer types with divisibility information. It serves as a proxy for constrained integer types in the CuTe IR, automatically tracking divisibility constraints that are crucial for layout operations.

**Key Features:**

- Inherits from `ArithValue` with extensions for divisibility tracking
- Automatically emits `cute.get_scalars` operations in the IR
- Supports arithmetic operations that propagate divisibility information
- Used internally for type-safe integer operations in layout algebra

**API Methods:**

- `get_typed_value()` - Returns the value as an IntTupleType
- `get_divisibility()` - Returns the divisibility constraint of the value
- `divisibility` - Property that returns the divisibility constraint

**Supported Operations:**

The `IntValue` type supports standard arithmetic operations with divisibility tracking:

``` python
# Addition, subtraction, multiplication, division, and modulo
result = int_val1 + int_val2
result = int_val1 - int_val2
result = int_val1 * int_val2
result = int_val1 // int_val2
result = int_val1 % int_val2
```

**String Representation:**

``` python
# IntValue with divisibility 1
str(int_val)  # Returns "?"

# IntValue with divisibility 4
str(int_val)  # Returns "?{div=4}"
```

#### Ratio

`Ratio` represents a rational number as a ratio of two integers. It is used in CuTe to represent exact fractional values that arise in tensor layout operations, particularly in composition operations where divisibility conditions may not be satisfied.

**Constructor:**

``` python
ratio = cute.Ratio(numerator, denominator)
```

param numerator
The numerator of the ratio

type numerator
int

param denominator
The denominator of the ratio

type denominator
int

raises TypeError
If numerator or denominator are not integers

**Methods:**

- `is_integral()` - Returns `True` if the ratio represents an integer value (numerator divisible by denominator)
- `reduced()` - Returns a new Ratio with numerator and denominator reduced to lowest terms
- `to(dtype)` - Converts the ratio to another type (Ratio, float, or int)

**Arithmetic Operations:**

``` python
# Multiplication with another ratio
ratio1 = cute.Ratio(1, 2)
ratio2 = cute.Ratio(3, 4)
result = ratio1 * ratio2  # Returns Ratio(3, 8)

# Multiplication with integer
ratio = cute.Ratio(2, 3)
result = ratio * 5  # Returns Ratio(10, 3)
result = 5 * ratio  # Returns Ratio(10, 3)
```

**Type Conversion:**

``` python
ratio = cute.Ratio(3, 2)

# Convert to float
float_val = ratio.to(float)  # Returns 1.5

# Convert to int (floor division)
int_val = ratio.to(int)  # Returns 1
```

### Layout Algebra Types

#### ScaledBasis

`ScaledBasis` represents a scaled basis element in CuTe's layout algebra. It consists of a scale value and a mode that identifies which basis element in the layout algebra is being referenced. ScaledBasis elements are fundamental to CuTe's coordinate system representation.

**Constructor:**

``` python
sb = cute.ScaledBasis(value, mode)
```

param value
The scale value

type value
Union\[int, Integer, Ratio, ir.Value\]

param mode
The mode identifying the basis element

type mode
Union\[int, List\[int\]\]

raises TypeError
If mode is not an integer or list of integers

**Examples:**

``` python
# Create a scaled basis with integer scale and mode
sb1 = cute.ScaledBasis(2, 0)  # 2 * E(0)

# Create a scaled basis with a Ratio scale
sb2 = cute.ScaledBasis(cute.Ratio(1, 2), 1)  # (1/2) * E(1)

# Create a scaled basis with a list of modes
sb3 = cute.ScaledBasis(4, [0, 1])  # 4 * E([0, 1])

# Scaled basis elements are commonly used in layout strides
layout = cute.make_layout((4, 8), stride=(cute.ScaledBasis(2, 0), cute.ScaledBasis(1, 1)))

# This creates a layout with strides (2@0, 1@1) representing
# a coordinate system where each dimension has its own basis

# Example: Mapping coordinates to indices using the layout
coord = (2, 3)
idx = cute.crd2idx(coord, layout)  # Maps (2, 3) to (4, 3)
```

**Properties:**

- `value` - Get the scale value
- `mode` - Get the mode as a list of integers
- `is_static()` - Returns `True` if the value is statically known

**Methods:**

- `to(dtype)` - Convert to another type (ScaledBasis or internal ScaledBasis)

**Operations:**

``` python
# Right multiplication by a scale factor
sb = cute.ScaledBasis(2, 0)
result = 3 * sb  # Creates ScaledBasis(6, 0)
```

**Utility Function:**

``` python
# Create a basis element with unit scale
basis = cute.E(mode)  # Equivalent to ScaledBasis(1, mode)
```

#### Swizzle

`Swizzle` is a transformation that permutes the elements of a layout. Swizzles are used to rearrange data elements to improve memory access patterns and computational efficiency, particularly for avoiding bank conflicts in shared memory.

**Swizzle Parameters:**

A swizzle is defined by three parameters:

- **MBase**: The number of least-significant bits to keep constant
- **BBits**: The number of bits in the mask
- **SShift**: The distance to shift the mask

**Bit Pattern:**

``` text
0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
                              ^--^ MBase (least-sig bits kept constant)
                 ^-^       ^-^     BBits (number of bits in mask)
                   ^---------^     SShift (distance to shift YYY)
                                      (positive: right, negative: left)

Given:    0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx
Result:   0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx
          where AA = ZZ xor YY
```

**Usage:**

Swizzles are typically created using CuTe's swizzle factory functions and composed with layouts to create optimized memory access patterns.

#### Layout

`Layout` is CuTe's core abstraction for representing tensor layouts. A Layout maps from a logical coordinate space to an index space, defined by a pair of (Shape, Stride). Layouts present a common interface to multidimensional array access that abstracts away the details of how array elements are organized in memory.

**Key Concepts:**

- **Shape**: Defines the abstract dimensions of the Layout
- **Stride**: Defines how coordinates within the Shape map to linear indices
- **Hierarchical Structure**: CuTe layouts are inherently hierarchical, constructed from smaller nested layouts

**Properties:**

- `shape` - An IntTuple representing the dimensions of the layout
- `stride` - An IntTuple representing the strides of the layout
- `max_alignment` - The maximum alignment of the layout in bytes

**Examples:**

``` python
# Creating a layout with shape (4,8) and default stride (column major)
layout = cute.make_layout((4, 8))

# Creating a layout with explicit shape and stride (row major)
layout = cute.make_layout((4, 8), stride=(8, 1))

# Accessing layout properties
shape = layout.shape      # Returns (4, 8)
stride = layout.stride    # Returns (8, 1)

# Mapping a coordinate to an index: (2, 3) -> 2 * 8 + 3 * 1 = 19
idx = cute.crd2idx((2, 3), layout)
```

**Layout Operations:**

Layouts support a rich algebra of operations:

- **Concatenation**: Combining layouts along dimensions
- **Coalescence**: Merging adjacent modes
- **Composition**: Composing layouts with functions or other layouts
- **Complement**: Computing the complement space
- **Inversion**: Inverting the layout mapping

**String Representation:**

``` python
layout = cute.make_layout((4, 8), stride=(1, 4))
print(layout)  # Prints "shape:stride" format, e.g., "(4,8):(1,4)"
```

#### ComposedLayout

`ComposedLayout` represents a composition of layouts and transformations. It is a generalization of normal layouts that can support arbitrary function mappings from coordinate to coordinate as an inner layout.

**Structure:**

A ComposedLayout consists of three components:

- **inner**: The inner transformation (Swizzle or Layout)
- **offset**: An offset applied to coordinates
- **outer**: The outer layout

**Properties:**

- `inner` - Returns the inner transformation (Union\[Swizzle, Layout\])
- `offset` - Returns the offset as an IntTuple
- `outer` - Returns the outer layout
- `shape` - Returns the shape of the composed layout
- `max_alignment` - Returns the maximum alignment
- `is_normal` - Returns `True` if this is a normal layout (not a general composition)

**Examples:**

``` python
# ComposedLayouts are typically created through composition operations
# For example, composing a layout with a swizzle
layout = cute.make_layout((8, 8))
swizzle = cute.make_swizzle(...)
composed = cute.composition(swizzle, layout)

# Accessing components
inner = composed.inner      # Returns the swizzle
outer = composed.outer      # Returns the layout
offset = composed.offset    # Returns the offset
```

**String Representation:**

``` python
print(composed)  # Prints "inner o offset o outer" format
```

### Memory and Pointer Types

#### Pointer

`Pointer` represents a memory address with specific properties. Pointers are a fundamental type of iterator/engine that support random-access operations. They can be offset by elements of a layout's codomain and dereferenced to produce values.

**Properties:**

- `dtype` - The type of value this pointer points to
- `type` - The MLIR type of the pointer
- `memspace` - The memory space where the pointer data resides (e.g., `gmem`, `smem`, `rmem`)
- `alignment` - The alignment of the pointer in bytes
- `max_alignment` - The maximum alignment of the pointer in bytes

**Operations:**

``` python
# Pointer arithmetic
ptr2 = ptr + offset     # Offset pointer forward
ptr3 = offset + ptr     # Offset pointer forward (commutative)
ptr4 = ptr - offset     # Offset pointer backward

# Convert pointer to integer
int_addr = ptr.toint()

# Align pointer to specified byte boundary
aligned_ptr = ptr.align(16)  # Align to 16-byte boundary
```

**Tensor Composition:**

When composed with a layout, a pointer forms a tensor: `T = E ∘ L`, where `E` is the pointer (engine) and `L` is the layout. The tensor evaluates the layout by mapping a coordinate `c` to the codomain, offsets the pointer accordingly, and dereferences the result:

``` text
T(c) = (E ∘ L)(c) = *(E + L(c))
```

**Methods:**

- `llvm_ptr` - Get the LLVM pointer representation (low-level use only)
- `align(min_align)` - Align pointer to specified byte alignment (must be power of 2)
- `toint()` - Convert pointer to integer address (Int64 for gmem/generic, Int32 otherwise)

**Examples:**

``` python
# Create a pointer from a tensor's data
ptr = tensor.data()

# Offset the pointer
offset_ptr = ptr + 16

# Check pointer properties
print(f"Memory space: {ptr.memspace}")
print(f"Alignment: {ptr.alignment}")
print(f"Data type: {ptr.dtype}")
```

### Structured Data Types

#### struct

The `struct` decorator abstracts C structures in Python DSL. It allows you to define structured data types with precise control over layout, alignment, and nesting.

**Supported Elements:**

- Base DSL scalar int/float elements
- Arrays (MemRange)
- Nested structures
- Aligned elements

**Basic Usage:**

``` python
# Define a simple struct
@cute.struct
class complex:
    real : cutlass.Float32
    imag : cutlass.Float32

# Define a struct with arrays and nested structures
@cute.struct
class StorageA:
    mbarA : cute.struct.MemRange[cutlass.Int64, stage]
    compA : complex
    intA : cutlass.Int16
```

**Alignment Control:**

``` python
# Define a struct with explicit alignment
@cute.struct
class StorageB:
    a: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, size_a], 1024
    ]
    b: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, size_b], 1024
    ]
    x: cute.struct.Align[cutlass.Int32, 16]
    compA: cute.struct.Align[complex, 16]
```

**Static Queries:**

``` python
# Get size and alignment at compile time
size = StorageB.__sizeof__()
align = StorageB.__alignof__()
```

**Allocation and Access:**

``` python
# Allocate and reference elements
storage = allocator.allocate(StorageB)

# Access struct members
storage.a[0] = ...
storage.x = ...
... = storage.compA.real.ptr
... = storage.x.ptr.load()
```

**Methods:**

- `__sizeof__()` - Returns the size of the struct in bytes
- `__alignof__()` - Returns the alignment of the struct in bytes
- `size_in_bytes()` - Returns the size of the struct in bytes

##### struct.MemRange

`MemRange` defines a contiguous range of memory with a specific element type and size.

**Syntax:**

``` python
cute.struct.MemRange[dtype, size]
```

param dtype
The data type (must be a DSL scalar type)

type dtype
Type\[Numeric\]

param size
The number of elements in the range

type size
int

**Properties:**

- `size` - Number of elements in the range
- `elem_width` - Width of each element in bits
- `size_in_bytes` - Total size in bytes

**Methods:**

- `data_ptr()` - Returns a pointer to the start of the memory range
- `get_tensor(layout, swizzle=None, dtype=None)` - Creates a tensor from the memory range
- `__getitem__(index)` - Returns the element at the specified index

**Examples:**

``` python
@cute.struct
class Buffer:
    data : cute.struct.MemRange[cutlass.Float32, 128]

# Allocate buffer
buf = allocator.allocate(Buffer)

# Get pointer to data
ptr = buf.data.data_ptr()

# Access individual elements
element = buf.data[5]

# Create tensor from memory range
layout = cute.make_layout((8, 16))
tensor = buf.data.get_tensor(layout)
```

##### struct.Align

`Align` specifies explicit alignment requirements for struct members.

**Syntax:**

``` python
cute.struct.Align[dtype, alignment]
```

param dtype
The type to align (scalar, MemRange, or struct)

type dtype
Type

param alignment
The alignment in bytes (must be \> 0)

type alignment
int

**Properties:**

- `dtype` - The data type being aligned
- `align` - The alignment value

**Examples:**

``` python
@cute.struct
class AlignedStorage:
    # Align scalar to 16 bytes
    counter: cute.struct.Align[cutlass.Int32, 16]

    # Align array to 1024 bytes
    buffer: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 256], 1024
    ]
```

#### union

The `union` decorator abstracts C unions in Python DSL. Similar to `struct`, but all members start at offset 0, and the size is the maximum size of all members.

**Layout Characteristics:**

- All objects start at offset 0
- Alignment is the maximum alignment of all objects
- Size is the maximum size of all objects

**Usage:**

``` python
# Define a union with scalar elements
@cute.union
class value_union:
    as_int : cutlass.Int32
    as_float : cutlass.Float32

# Allocate union
val = allocator.allocate(value_union)

# Access different interpretations of same memory
val.as_int = 42
float_val = val.as_float.ptr.load()  # Interpret same bits as float
```

**Methods:**

Same as `struct`:

- `__sizeof__()` - Returns the size of the union in bytes
- `__alignof__()` - Returns the alignment of the union in bytes

### Type Hierarchies and Relationships

**Type Protocol Support:**

Many CuTe types implement standard Python protocols for integration:

- `__str__()` - String representation for debugging
- `__eq__()` / `__ne__()` - Equality comparison
- `__getitem__()` - Indexing operations
- `__add__()` / `__sub__()` / `__mul__()` / `__floordiv__()` / `__mod__()` - Arithmetic

**MLIR Integration:**

Internal types like `IntValue`, `Layout`, `Pointer`, and `ComposedLayout` are registered as MLIR value casters, enabling seamless integration with the underlying compiler infrastructure.

### Best Practices

**Choosing Between Static and Dynamic:**

- Use static values (Python `int`) when dimensions are known at compile time for maximum optimization
- Use dynamic values (`IntValue`) when dimensions must be determined at runtime
- Refer to JIT Argument: Layouts for detailed guidance on static vs dynamic layouts

**Memory Alignment:**

- Always specify alignment requirements for shared memory structures to avoid bank conflicts
- Use `struct.Align` to enforce alignment constraints
- Check `max_alignment` properties to verify pointer and layout alignment

**Layout Operations:**

- Prefer built-in layout operations (`make_layout`, `composition`, etc.) over manual construction
- Use `ScaledBasis` for explicit control over stride modes in multi-modal layouts
- Leverage `ComposedLayout` for complex transformations like swizzling

**Type Safety:**

- Use type annotations in `@jit` and `@kernel` functions
- Let the DSL infer types when possible for cleaner code
- Check `dtype` and `memspace` properties when working with pointers

### See Also

- Introduction - Introduction to CuTe DSL decorators and calling conventions
- Control Flow - Control flow with static and dynamic values
- JIT Argument: Layouts - Working with static and dynamic layouts
- Integration with Frameworks - Integration with deep learning frameworks
- Debugging with the DSL - Debugging techniques for CuTe DSL programs

---

<!-- source: cute_dsl_general/framework_integration.rst -->

## Integration with Frameworks

In order to facilitate the integration of CUTLASS Python with popular frameworks, we leverage the [DLPack protocol](https://github.com/dmlc/dlpack) and transform tensors originating from these frameworks to CuTe tensors. The present page documents the conventions, the API available to the user, and provide example code snippets for common usage patterns. We also provide a section on how to bypass the DLPack protocol and directly call the JIT function.

### Implicit Conversion

Tensors originating from frameworks supporting the DLPack protocol can be directly provided to a JIT function as a regular parameter. CuTe DSL's runtime implicitly converts the original tensor to a CuTe tensor with a fully dynamic layout except for the stride element corresponding to the leading dimension. The example below demonstrates this use case.

``` python
import torch
import cutlass.cute as cute

@cute.jit
def foo(src):
    """
    The following lines print

    ptr<f32, generic> o (?,?,?):(?,?,1)
    <class 'cutlass.cute.core._Tensor'>
    """
    print(src)
    print(type(src))

a = torch.randn(30, 20, 32, device="cpu")
foo(a)
```

### Explicit conversion using `from_dlpack`

CuTe DSL's runtime provides an interface for converting DLPack-compatible tensors to CuTe tensors,

``` python
b = cute.runtime.from_dlpack(a)
```

where `a` is a tensor supporting the DLPack protocol with the `__dlpack__` and `__dlpack_device__` methods. The resulting CuTe tensor `b` has a fully static layout. This conversion is performed without copying any tensor data, enabling seamless integration with major frameworks. Users can create tensors using NumPy, PyTorch, etc. and directly feed them into JIT functions writtnen using CuTe DSL.

The resulting CuTe tensor shares the same underlying memory buffer as the original tensor. This zero-copy approach maximizes performance by eliminating unnecessary data duplication. However, it is important to note that the CuTe tensor's validity is tied to the lifetime of the original tensor. If the source tensor is destroyed or goes out of scope, the corresponding CuTe tensor becomes invalid since it references the original memory location.

The full signature of from_dlpack is as follows:

``` python
def from_dlpack(tensor, assumed_align=None, use_32bit_stride=False):
```

The `assumed_align` integer parameter specifies the alignment of the tensor in unit of bytes. The tensor's base address must be divisible by `assumed_align`. When not provided explicitly, the alignment is set to the natural alignment of the tensor's element type. Note that the alignment information is part of the pointer type in the generated IR. Therefore, programs with different alignments have a different IR and identical IRs are required for hitting the kernel caching mechanism of CuTe DSL.

The `use_32bit_stride` parameter determines whether to use 32-bit stride for the tensor's dynamic stride values. By default, it is set to False (64bit) to ensure that address calculations do not risk overflow. For smaller problem sizes (where `cosize(layout_of_tensor) <= Int32_MAX`), users may set it to True (32bit) to improve performance by reducing register usage and the number of address calculation instructions. When `use_32bit_stride` is set to True, a runtime check is performed to ensure that the layout does not overflow. Please note that this parameter only has an effect when the tensor's layout is marked as dynamic.

For packed subbyte torch dtypes such as `torch.float4_e2m1fn_x2`, `from_dlpack` exposes the logical element layout expected by CuTe instead of the packed storage layout. For example, a torch tensor with shape `(128, 128)` and dtype `torch.float4_e2m1fn_x2` is exposed as a logical FP4 tensor with shape `(128, 256)`. The same logical reinterpretation also applies when the leading dimension is not the last mode.

#### Code Example

The following code demonstrates how to convert a PyTorch tensor to a CuTe tensor using the `from_dlpack` function with default parameters.

``` python
import torch
import cutlass
from cutlass.cute.runtime import from_dlpack

x = torch.randn(30, 20, device="cpu")
y = from_dlpack(x)
```

Once converted, we can access the tensor's information through various attributes. The following list shows the attributes of the converted tensor:

- `tensor.shape`: the tensor's shape
- `tensor.stride`: the tensor's stride
- `tensor.memspace`: the tensor's memory space
- `tensor.element_type`: the tensor's element data type

``` python
import torch
import cutlass
from cutlass.cute.runtime import from_dlpack

x = torch.randn(30, 20, device="cpu")
y = from_dlpack(x)

print(y.shape)        # (30, 20)
print(y.stride)       # (20, 1)
print(y.memspace)     # generic (if torch tensor in on device memory, memspace will be gmem)
print(y.element_type) # Float32
print(y)              # Tensor<0x000000000875f580@generic o (30, 20):(20, 1)>
```

The string format of the resulting CuTe tensor is

```
Tensor<0x{tensor.data_ptr:016x}@{tensor.memspace} o {tensor.shape}:{tensor.stride}>
```

As can be seen in the example above, `from_dlpack` first results in a tensor with a static layout. To obtain dynamic or mixed static/dynamic layouts after calling `from_dlpack`, the `mark_layout_dynamic` and `mark_compact_shape_dynamic` functions are used and described in the following sections.

#### When to Use Explicit Conversion?

The DLPack protocol is a widely used protocol for interoperability between different frameworks. However, there is some associated overhead. Based on our benchmark, it usually takes between 2 to 3 us per call to `from_dlpack`.

Explicit conversion allows for caching the converted CuTe tensors in order to avoid the overhead of repeated calls to `from_dlpack`.

``` python
x = torch.randn(30, 20, device="cpu")
if key not in cached_tensors:
    # Do the conversion only for cache misses
    cached_tensors[key] = cute.runtime.from_dlpack(x)
foo(cached_tensors[key])
```

Another use case for explicit conversion is to gain fine-grain control over which modes of a tensor are considered dynamic from the perspective of the generated program.

### Mark the Tensor's Layout as Dynamic with `mark_layout_dynamic`

After calling this function, all shape modes become dynamic. The stride modes also become dynamic with the following two exceptions:

1.  the leading dimension's stride remains fixed at 1;
2.  stride elements equal to 0 (which indicates broadcasting) are retained.

The full signature of `mark_layout_dynamic` is as follows:

``` python
def mark_layout_dynamic(self, leading_dim: int|None = None):
```

The `leading_dim` parameter specifies the leading dimension of the tensor. The leading dimension's stride is set to 1 unless inconsistent with the layout of the DLPack tensor. For example,

- For a tensor with layout `(2,2,3,4):(2,1,4,12)`, if `leading_dim` is specified to be 1, the layout will be marked as `(?,?,?,?):(?,1,?,?)`.
- If `leading_dim` is specified to be 0, a deduction failure error is raised because the stride of dimension 0 is 2 (not 1).

The default value for `leading_dim` is `None`. In such case, the system automatically deduces it from the tensor's layout using the following logic:

1.  If exactly one dimension has stride 1, that dimension is the leading dimension.
2.  If multiple dimensions have stride 1, deduction succeeds only when exactly one of them has size \> 1 (that dimension is used). If none or more than one has size \> 1, an error is raised. Note that after converting a **PyTorch** tensor to the DLPack format, the stride for dimensions with size 1 are canonicalized to 1, which can produce multiple stride-1 dimensions.
3.  If no dimension has stride 1, all strides remain dynamic.

For example:

- For a tensor with layout `(2,2,3,4):(2,1,4,12)`, the leading dimension is 1. The layout will be marked as `(?,?,?,?):(?,1,?,?)`.
- For a tensor with layout `(1,5,1):(1,1,1)`, multiple dimensions have stride 1 but exactly one has size \> 1 (dim 1). The leading dimension is deduced to be 1: `(?,?,?):(?,1,?)`.
- For a tensor with layout `(2,2):(8,2)`, no dimension has stride 1, so all strides remain dynamic: `(?,?):(?,?)`.

The leading dimension accepts negative index which means the dimension is counted from the last dimension. For example,

- For a tensor with layout `(2,2,3,4):(2,1,4,12)`, if `leading_dim` is specified to be -1, the layout will be marked as `(?,?,?,?):(?,?,?,1)`.

#### Code Example

The following example demonstrates how to use `mark_layout_dynamic` to specify dynamic tensor layouts.

- `t0` shows the usage of `mark_layout_dynamic` with unspecified `leading_dim` and the automatic deduction of leading dimension.
- `t1` & `t2` shows the usage of `mark_layout_dynamic` with specified `leading_dim`.
- `t3` shows the usage of `mark_layout_dynamic` with no leading dimension.
- `t4` shows the usage of `mark_layout_dynamic` with broadcasted dimensions.
- `t5` shows automatic deduction for tensor `b` (multiple stride-1, exactly one has size \> 1 → dim 1).
- `t5_fail` demonstrates the deduction failure when multiple dimensions have stride 1 but none has size \> 1.
- `t6` & `t7` demonstrate incorrect settings for `leading_dim` and expected errors.

``` python
import torch
from cutlass.cute.runtime import from_dlpack

# (8,4,16,2):(2,16,64,1)
a = torch.empty(16, 4, 8, 2).permute(2, 1, 0, 3)
# (1,4,1,32,1):(4,1,4,4,4) => torch tensor when dimension has shape 1, its stride is degenerated to 1,
# resulting in (1,4,1,32,1):(1,1,1,4,1)
b = torch.empty(32, 1, 1, 1, 4).permute(3, 4, 1, 0, 2)
# (2,2):(8,2)
c = torch.empty(3, 4)[::2, ::2]
# (3,1,1,5):(5,0,0,1)
d = torch.empty(3, 1, 1, 5).expand(3, 4, 2, 5)

# auto deduce the leading dimension to be 3
t0 = from_dlpack(a).mark_layout_dynamic()
print(t0)
# (?,?,?,?):(?,?,?,1)

t1 = from_dlpack(b).mark_layout_dynamic(leading_dim=0)
print(t2)
# (?,?,?,?,?):(1,?,?,?,?)

t2 = from_dlpack(b).mark_layout_dynamic(leading_dim=2)
print(t3)
# (?,?,?,?,?):(?,?,1,?,?)

t3 = from_dlpack(c).mark_layout_dynamic()
print(t3)
# (?,?):(?,?)

t4 = from_dlpack(d).mark_layout_dynamic()
print(t4)
# (?,?,?,?):(?,0,0,1)

# b has layout (1,4,1,32,1):(1,1,1,4,1); dim 1 has size > 1, so deduction succeeds to dim 1.
t5 = from_dlpack(b).mark_layout_dynamic()
print(t5)
# (?,?,?,?,?):(?{i64},1,?{i64},?{i64},?{i64})

# Rejected: multiple stride-1, none with size > 1 (e.g. torch.ones(1,1,1)).
t5_fail = from_dlpack(torch.ones(1, 1, 1)).mark_layout_dynamic()
# Can't deduce the leading dimension from layout (multiple dimensions have stride 1 but none has size > 1)...

t6 = from_dlpack(a).mark_layout_dynamic(leading_dim=1)
# Expected strides[leading_dim] == 1, but got 16

t7 = from_dlpack(b).mark_layout_dynamic(leading_dim=3)
# Expected strides[leading_dim] == 1, but got 4

c = torch.empty(1000000000, 1000000000)
t8 = from_dlpack(c, use_32bit_stride=True).mark_layout_dynamic()
# Layout in DLTensorWrapper has int32 overflow risk. Please set use_32bit_stride to False.
```

### Mark the Tensor's Layout as Dynamic with `mark_compact_shape_dynamic`

The `mark_compact_shape_dynamic` function provides fine-grain control over dynamic shapes for compact layouts. The full signature of `mark_compact_shape_dynamic` is as follows:

``` python
def mark_compact_shape_dynamic(self, mode: int, stride_order: tuple[int, ...]|None = None, divisibility: int = 1):
```

The `mode` parameter determines which shape dimension becomes dynamic. After calling this function, the specific shape dimension given by `mode` is marked as dynamic immediately. The stride will be updated accordingly. For modes that have a shape of size 1, their stride are canonicalized to 0.

The `stride_order` parameter specifies the ordering of strides in the tensor. It is consistent with `torch.Tensor.dim_order()` and defaults to `None`. The parameter indicates the order of modes (dimensions) if the current layout were to be converted to row-major order. It starts from the outermost to the innermost dimension when reading it from left to right. This parameter must be explicitly set when the stride order cannot be automatically deduced from the tensor's layout, such as when multiple dimensions have a stride of 1.

For example:

- Layout `(4,2):(1,4)` has a `stride_order` of `(1,0)` indicates the innermost dimension is 0 (`4:1`), the outermost dimension is 1 (`2:4`).
- Layout `(5,3,2,4):(3,1,15,30)` has a `stride_order` of `(3,2,0,1)` indicates the innermost dimension is 1 (`3:1`), the outermost dimension is 3 (`4:30`).

If `stride_order` is not specified, the system automatically deduces it from the tensor's layout using the following logic:

1.  Sort the strides in descending order.
2.  If multiple dimensions have a stride of 1, a deduction failure error is raised.

For example:

- For a tensor with layout `(2,2,3,4):(2,1,4,12)`, the deduced `stride_order` is `[3,2,0,1]`.
- For a tensor with layout `(1,5,1):(1,1,1)`, `stride_order`'s deduction fails because all dimensions have an identical stride of 1, making it impossible to determine the correct ordering.

If `stride_order` is specified, the system validates that the order is consistent with the tensor's layout.

The `divisibility` parameter specifies the divisibility of the dynamic shape. It could be used to represent the assumption alignment of the input. Defaults to 1.

Note that this API is only available for compact tensors. For non-compact tensors, we can use `cute.assume` to attach divisibility information to a specific shape mode in a host JIT function, as demonstrated in the following example:

``` python
@cute.jit
def foo(a: cute.Tensor):
    new_shape = a.shape
    # use cute.assume to set shape of mode=0 with divisibility=16
    new_shape[0] = cute.assume(new_shape[0], 16)
    new_layout = cute.make_layout(new_shape, stride=a.stride)
    new_a = cute.make_tensor(a.iterator, new_layout)
```

#### Code Example

The following example demonstrates how to use `mark_compact_shape_dynamic` to specify dynamic tensor layouts.

- `t0` & `t1` show the usage of `mark_compact_shape_dynamic` with unspecified `stride_order` and different `mode` and `divisibility`.
- `t2` shows the usage of consecutive `mark_compact_shape_dynamic` with unspecified `stride_order` and different `mode` and `divisibility`.
- `t3` & `t4` show the usage of `mark_compact_shape_dynamic` with different specified `stride_order`.
- `t5`, `t6`, `t7`, `t8`, `t9`, `t10`, `t11`, and `t12` demonstrate incorrect settings for parameters and expected errors.

``` python
import torch
from cutlass.cute.runtime import from_dlpack

# (8,4,16,2):(2,16,64,1)
a = torch.empty(16, 4, 8, 2).permute(2, 1, 0, 3)
# (1,4,1,32,1):(4,1,4,4,4) => torch tensor when dimension has shape 1, its stride is degenerated to 1,
# resulting in (1,4,1,32,1):(1,1,1,4,1)
# b.dim_order() is (3,2,4,0,1)
b = torch.empty(32, 1, 1, 1, 4).permute(3, 4, 1, 0, 2)

# auto deduce the stride order to be [2,1,0,3]
t0 = from_dlpack(a).mark_compact_shape_dynamic(
    mode=0, divisibility=2
)
# (?{div=2},4,16,2):(2,?{div=4},?{div=16},1)
print(t0)

t1 = from_dlpack(a).mark_compact_shape_dynamic(
    mode=1, divisibility=2
)
# (8,?{div=2},16,2):(2,16,?{div=32},1)
print(t1)

t2 = from_dlpack(a).mark_compact_shape_dynamic(
    mode=1, divisibility=2
).mark_compact_shape_dynamic(
    mode=3, divisibility=2
)
# (8,?{div=2},16,?{div=2}):(?{div=2},?{div=16},?{div=32},1)
print(t2)

t3 = from_dlpack(b).mark_compact_shape_dynamic(
    mode=2, divisibility=1, stride_order=(3, 0, 2, 4, 1)
)
# (1,4,?,32,1):(0,1,4,?{div=4},0)
print(t3)

t4 = from_dlpack(b).mark_compact_shape_dynamic(
    mode=2, divisibility=1, stride_order=(2, 3, 4, 0, 1)
)
# (1,4,?,32,1):(0,1,128,4,0)
print(t4)

t5 = t2.mark_compact_shape_dynamic(
    mode=3, divisibility=5, stride_order=(0, 1, 2, 3)
)
# The stride_order is not consistent with the last stride_order

t6 = from_dlpack(a).mark_compact_shape_dynamic(
    mode=3, divisibility=5, stride_order=(0, 1, 2, 3)
)
# The stride_order is not consistent with the deduced stride_order

t7 = from_dlpack(b).mark_compact_shape_dynamic(
    mode=0, divisibility=4
)
# The layout could not be deduced, please specify the stride_order explicitly

t8 = from_dlpack(b).mark_compact_shape_dynamic(
    mode=30, divisibility=5, stride_order=(3, 0, 2, 4, 1)
)
# Expected mode value to be in range [0, 5), but got 30

t9 = from_dlpack(b).mark_compact_shape_dynamic(
    mode=3, divisibility=5, stride_order=(2, 1, 2, 3, 4)
)
# Expected stride_order to contain all the dimensions of the tensor, but it doesn't contain 0.

t10 = from_dlpack(b).mark_compact_shape_dynamic(
    mode=3, divisibility=5, stride_order=(0, 1, 2, 3, 4, 5)
)
# Expected stride_order to have 5 elements, but got 6.

t11 = from_dlpack(b).mark_compact_shape_dynamic(
    mode=0, divisibility=4, stride_order=b.dim_order()
)
# The shape(1) of mode(0) is not divisible by the divisibility(4)

t12 = from_dlpack(b).mark_compact_shape_dynamic(
    mode=0, divisibility=1, stride_order=(2, 1, 3, 0, 4)
)
# The stride_order is not consistent with the layout

c = torch.empty(1000000000, 1000000000)
t13 = from_dlpack(c, use_32bit_stride=True).mark_compact_shape_dynamic(
    mode=0, divisibility=1
)
# Layout in DLTensorWrapper has int32 overflow risk. Please set use_32bit_stride to False.
```

### Leveraging TVM FFI for Faster PyTorch Interop

The latest version of CuTe DSL supports TVM FFI to improve interoperability with PyTorch and other machine learning frameworks. Using TVM FFI provides the following features:

- Faster JIT function invocation.
- Direct acceptance of `torch.Tensor` objects as function arguments.
- Enhanced error handling and kernel validation.
- Seamless integration with multiple programming languages.

For more details, see Compile with TVM FFI.

### Bypass the DLPack Protocol

In certain scenarios, users may wish to bypass the DLPack protocol and invoke the JIT function directly. This can be accomplished by creating a lightweight JIT wrapper around the existing JIT function, utilizing `cute.ptr` and `cute.make_tensor` to pass pointers and construct tensors directly.

Typical use cases for bypassing DLPack include: 1. Users want to call the JIT function directly to avoid the overhead introduced by the DLPack protocol. 2. DLPack canonicalizes the stride of shape-1 dimensions to 1, which may result in incorrect alignment propagation and affect memory access or performance. 3. DLPack may lack support for some narrow data types.

The following example illustrates how to bypass the DLPack protocol when invoking a JIT function. Assume we have a pre-defined `TensorOpGemm` kernel whose JIT interface expects three arguments of type `cute.Tensor`. To enable direct invocation without DLPack, we first define a JIT wrapper function that accepts `cute.Pointer` types as parameters. Within this wrapper, we use `cute.make_tensor` to construct tensors from the provided pointers, and then call the `TensorOpGemm` kernel as usual.

``` python
@cute.jit
def tensor_op_gemm_wrapper(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    m: cutlass.Int32,
    n: cutlass.Int32,
    k: cutlass.Int32,
    l: cutlass.Int32,
):

    # Assume alignment of shape to call tensorop_gemm example
    m = cute.assume(m, divby=8)
    n = cute.assume(n, divby=8)

    # Torch is row major
    a_layout = cute.make_ordered_layout((m, k, l), order=(0, 1, 2))
    b_layout = cute.make_ordered_layout((n, k, l), order=(0, 1, 2))
    c_layout = cute.make_ordered_layout((m, n, l), order=(1, 0, 2))
    mA = cute.make_tensor(a_ptr, layout=a_layout)
    mB = cute.make_tensor(b_ptr, layout=b_layout)
    mC = cute.make_tensor(c_ptr, layout=c_layout)

    # TensorOpGemm is a pre-defined kernel from our example
    tensor_op_gemm = TensorOpGemm(
        a_ptr.value_type, c_ptr.value_type, cutlass.Float32, (2, 2, 1)
    )

    tensor_op_gemm(mA, mB, mC)
```

To pass a PyTorch tensor to this new JIT wrapper, we retrieve the raw pointer from the PyTorch tensor and create a `cute.Pointer` instance using `cute.make_ptr`. This approach allows us to bypass the DLPack protocol entirely, avoiding its overhead and potential issues with shape-1 dimension handling.

``` python
a = torch.randn(
    m, k, l, dtype=torch.float16, device="cuda"
).permute(2, 1, 0)
b = torch.randn(
    n, k, l, dtype=torch.float16, device="cuda"
).permute(2, 1, 0)
c = torch.randn(
    n, m, l, dtype=torch.float16, device="cuda"
).permute(1, 2, 0)

# from cutlass.cute.runtime import make_ptr
a_ptr = make_ptr(
    cutlass.Float16, a.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=32
)
b_ptr = make_ptr(
    cutlass.Float16, b.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=32
)
c_ptr = make_ptr(
    cutlass.Float16, c.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=32
)
tensor_op_gemm_wrapper(a_ptr, b_ptr, c_ptr, m, n, k, l)
```

---

<!-- source: cute_dsl_general/debugging.rst -->

## Debugging

This page provides an overview of debugging techniques and tools for CuTe DSL programs.

### Getting Familiar with the Limitations

Before diving into comprehensive debugging capabilities, it's important to understand the limitations of CuTe DSL. Understanding these limitations will help you avoid potential pitfalls from the start.

Please refer to Limitations for more details.

### Source Code Correlation

CuTe DSL provides Python code to PTX/SASS correlation to enable the profiling/debugging of generated kernels with debug symbols by generating line info when compiling the kernel.

You can enable that globally via the environment variable CUTE_DSL_LINEINFO=1. Alternative, you can use compilation options to enable that per kernel. Please refer to JIT Compilation Options for more details.

### Debug Mode

To turn on a broad set of debugging aids at once, set the `CUTE_DSL_DEBUG` environment variable. It is a convenience switch for diagnosing problems and for reporting issues to the CUTLASS team:

``` bash
# Enable debug mode (default: False)
export CUTE_DSL_DEBUG=1
```

When debug mode is enabled, CuTe DSL raises the defaults of several individual debugging settings so you get more diagnostics from a single switch:

- Line info is generated for Python-to-PTX/SASS correlation (same effect as `CUTE_DSL_LINEINFO=1`).
- Full, unfiltered Python stack traces are shown on failure (internal DSL frames are no longer hidden).
- Optimization warnings that are normally suppressed are surfaced.
- Trace-time operation verification runs as operations are built, so malformed operations are reported earlier instead of late in compilation.
- Full per-launch argument validation is performed, so a mismatched or unsupported argument is reported with a clear error instead of failing later inside the compiled kernel.

Each of these behaviors is also controlled by its own environment variable, so debug mode only changes their *defaults*, and setting a variable explicitly takes precedence -- except trace-time operation verification, which stays on while debug mode is enabled. For example, to enable debug mode but keep line info off:

``` bash
export CUTE_DSL_DEBUG=1
export CUTE_DSL_LINEINFO=0
```

> [!NOTE]
> Debug mode adds extra checks and diagnostics that increase compile time and may affect the generated code (for example, by embedding line info). Enable it while debugging, not for production runs.

> [!NOTE]
> The settings debug mode raises -- line info in particular -- change the emitted IR/PTX, and every one of these settings is folded into the JIT kernel cache key. A kernel compiled with debug mode on is therefore cached separately from the same kernel compiled with it off: toggling `CUTE_DSL_DEBUG` forces a recompile instead of reusing a cached kernel, and the kernel you inspect or profile under debug mode is not identical to the one produced for a normal (debug-off) run. Validate performance and generated-code conclusions with debug mode disabled. Because these settings are part of the cache key, a debug-built kernel is never silently reused for a production run. See JIT caching for how the cache key is formed.

### DSL Debugging

CuTe DSL provides built-in logging mechanisms to help you understand the code execution flow and some of the internal state.

#### Enabling Logging

CuTe DSL provides environment variables to control logging level:

``` bash
# Enable console logging (default: False)
export CUTE_DSL_LOG_TO_CONSOLE=1

# Log to file instead of console (default: False).
# Set to 1/True to enable; the log file path is chosen automatically by the DSL.
export CUTE_DSL_LOG_TO_FILE=1

# Control log verbosity (0=disabled, 1=all messages (debug and above), 10=debug, 20=info, 30=warning, 40=error, 50=critical; default: 1)
export CUTE_DSL_LOG_LEVEL=20
```

#### Log Categories and Levels

Similar to standard Python logging, different log levels provide varying degrees of detail:

| Level | Description |
|-------|-------------|
| 0     | Disabled    |
| 10    | Debug       |
| 20    | Info        |
| 30    | Warning     |
| 40    | Error       |
| 50    | Critical    |

#### Save generated artifacts to files

CuTe DSL can save generated artifacts (IR, PTX, CUBIN, …) to files for offline inspection. Use `CUTE_DSL_KEEP` with a comma-separated list of artifact tokens:

``` bash
# Save clean IR (after canonicalize+cse, human-readable) to a .mlir file
export CUTE_DSL_KEEP=ir

# Save raw IR (before any passes) to a .mlir file
export CUTE_DSL_KEEP=ir-debug

# Save PTX assembly to a .ptx file
export CUTE_DSL_KEEP=ptx

# Save CUBIN binary to a .cubin file
export CUTE_DSL_KEEP=cubin

# Save SASS disassembly to a file (requires nvdisasm in PATH)
export CUTE_DSL_KEEP=sass

# Save LLVM IR to a file
export CUTE_DSL_KEEP=llvm

# Save multiple artifacts at once
export CUTE_DSL_KEEP=ir,ptx,cubin

# Save all supported artifacts
export CUTE_DSL_KEEP=all
```

Files are written to the current working directory by default. Use `CUTE_DSL_DUMP_DIR` to redirect them (see [Change the dump directory](#change-the-dump-directory) below).

> [!NOTE]
> The `sass` token requires `nvdisasm` (or `nvdisasm_internal`) to be available in your `PATH`. It is usually installed with the CUDA toolkit.

#### Print the generated IR to the console

To print the IR directly to the console (without writing a file):

``` bash
# Print generated IR to stdout (default: False)
export CUTE_DSL_PRINT_IR=1
```

#### Access the dumped contents programmatically

For compiled kernels, the generated PTX/CUBIN/IR can be accessed programmatically as well through following attributes:

- `__ptx__`: The generated PTX code of the compiled kernel.
- `__cubin__`: The generated CUBIN data of the compiled kernel.
- `__mlir__`: The generated IR code of the compiled kernel.

``` python
compiled_foo = cute.compile(foo, ...)
print(f"PTX: {compiled_foo.__ptx__}")
with open("foo.cubin", "wb") as f:
    f.write(compiled_foo.__cubin__)
```

#### Change the dump directory

By default, all dumped files are saved in the current working directory. To specify a different directory for the dumped files, please set the environment variable CUTE_DSL_DUMP_DIR accordingly.

### Kernel Functional Debugging

#### Using Python's `print` and CuTe's `cute.printf`

CuTe DSL programs can use both Python's native `print()` as well as our own `cute.printf()` to print debug information during kernel generation and execution. They differ in a few key ways:

- Python's `print()` executes during compile-time only (no effect on the generated kernel) and is typically used for printing static values (e.g. a fully static layouts).
- `cute.printf()` executes at runtime on the GPU itself and changes the PTX being generated. This can be used for printing values of tensors at runtime for diagnostics, but comes at a performance overhead similar to that of `printf()` in CUDA C.

For detailed examples of using these functions for debugging, please refer to the associated notebook referenced in Educational Notebooks.

#### Handling Unresponsive/Hung Kernels

When a kernel becomes unresponsive and `SIGINT` (`CTRL+C`) fails to terminate it, you can follow these steps to forcefully terminate the process:

1.  Use `CTRL+Z` to suspend the unresponsive kernel
2.  Execute the following command to terminate the suspended process:

``` bash
# Terminate the most recently suspended process
kill -9 $(jobs -p | tail -1)
```

CuTe DSL can also be debugged using standard NVIDIA CUDA tools.

#### Using Compute-Sanitizer

For detecting memory errors and race conditions:

``` bash
compute-sanitizer --some_options python your_dsl_code.py
```

Please refer to the [compute-sanitizer documentation](https://developer.nvidia.com/compute-sanitizer) for more details.

#### Set function name prefix

By default, the function name (host function or kernel function) is automatically generated based on the function name and its parameters. Sometimes you may want to attach some runtime information to the function name to make performance profiling and debugging easier, e.g., the kernel configs or the rank ids. You can assign a name prefix to the name by calling the `set_name_prefix` method on the host function or kernel function.

``` python
@cute.kernel
def kernel(arg1, arg2, ...):
    ...
@cute.jit
def launch_kernel():
    kernel.set_name_prefix("your_custom_name_prefix")
    kernel(arg1, arg2, ...).launch(grid=[1, 1, 1], block=[1, 1, 1], ...)
```

For above example, the generated kernel name will be "your_custom_name_prefix_xxx".

### Conclusion

This page covered several key methods for debugging CuTe DSL programs. Effective debugging typically requires a combination of these approaches. If you encounter issues with DSL, you can enable logging and share the logs with the CUTLASS team as a GitHub issue to report a bug.

---

<!-- source: cute_dsl_general/iket_profiling.rst -->

## IKET Profiling

> [!WARNING]
> IKET is an experimental profiling feature for CuTe DSL kernels. The API, output format, profiler workflow, and overhead characteristics may change in future releases. Users should understand the intended use cases and limitations before interpreting profiling results. IKET dialect tool support may also move to official NVIDIA Nsight tools in the future.

IKET, short for In-Kernel Event Tracing, lets CuTe DSL kernels emit named markers and ranges from inside the kernel. The `run-iket` profiler collects those events and generates timeline output that can be inspected in the Perfetto UI at <https://ui.perfetto.dev/>, along with machine-readable JSON output. `run-iket` is a purpose-built standalone profiler for collecting IKET traces in this experimental workflow. Conceptually, IKET is similar to CPU-side NVTX ranges and markers, but the events are emitted from device code inside the kernel. IKET supports Hopper and newer GPU architectures, including SM90, SM100, SM103, SM110, and SM120. The `run-iket` profiler is released with the `nvidia-cutlass-dsl` package.

IKET records device-side events at instrumentation points. The figure below is a conceptual illustration, not a `run-iket` output format. It shows the kind of producer/consumer activity that user-defined IKET ranges and markers can make visible inside a kernel. Actual timeline viewing uses the Perfetto trace described later in this guide.

<figure>
<img src="images/iket_concept_timeline.svg" style="width:90.0%" alt="Conceptual IKET timeline showing producer and consumer warp ranges" />
<figcaption>Conceptual view of IKET ranges and markers inside a kernel. <code>run-iket</code> emits Perfetto and JSON traces; it does not emit this simplified diagram.</figcaption>
</figure>

### Requirements

Use a `nvidia-cutlass-dsl` installation that includes `run-iket`. A quick first check is:

``` bash
run-iket --help
```

The profiled workload must run on a supported GPU architecture and must JIT-compile the instrumented CuTe DSL kernel during the `run-iket` profiling process. Kernels that are already compiled and reused without recompilation do not gain IKET instrumentation during that run.

### End-to-End Quick Start

This section shows the minimal end-to-end flow: add IKET calls in kernel code, run the workload under `run-iket`, and open the generated trace.

**Step 1: Add IKET instrumentation.** Place IKET calls inside a `@cute.kernel` function. IKET calls in host-side Python wrappers do not emit in-kernel events.

The following example shows a small kernel fragment with a marker, a token-based range, and a stack-based range.

``` python
import cutlass
import cutlass.cute as cute


@cute.kernel
def kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    bidx, _, _ = cute.arch.block_idx()

    cute.experimental.iket.mark("kernel_start", bidx)

    load_token = cute.experimental.iket.range_start("load")
    # Load data from gA and gB.
    cute.experimental.iket.range_end(load_token)

    cute.experimental.iket.range_push("compute")
    # Compute and store results.
    cute.experimental.iket.range_pop()
```

A complete CuTe DSL GEMM example with IKET instrumentation is available at `examples/python/CuTeDSL/dsl_tutorials/fp16_gemm_4_iket.py`.

**Step 2: Run the application under** `run-iket`. The profiler automatically requests IKET lowering for kernels that are JIT-compiled during the profiled run.

``` bash
run-iket profile --postprocess perfetto -- \
  python fp16_gemm_4_iket.py \
  --mnk 512,1024,64
```

**Step 3: Open the trace.** Open the generated `*.pftrace` file in the Perfetto UI at <https://ui.perfetto.dev/> and inspect the in-kernel markers and ranges.

<figure>
<img src="images/fp16_gemm_4_iket_quickstart.png" style="width:95.0%" alt="Cropped IKET timeline in Perfetto UI" />
<figcaption>Cropped Perfetto view from the GEMM tutorial, showing nested IKET ranges across several warp roles.</figcaption>
</figure>

The rest of this guide explains the API, instrumentation patterns, trace output, limitations, and overhead guidance in more detail.

IKET API calls are stripped by default. If neither `run-iket` is used to profile the target kernel nor an explicit compile option enables IKET lowering, the `iket.*` operations do not add instrumentation code to the final kernel.

To build IKET instrumentation outside a `run-iket` profiling run, enable IKET for all JIT compilations in the process:

``` bash
export CUTE_DSL_COMPILER_OPT=iket
python my_kernel.py
```

Alternatively, enable IKET for one explicit compilation:

``` python
compiled = cute.compile(host_function, *args, options="iket")
```

### API Reference

IKET APIs are available under `cutlass.cute.experimental.iket`. The table below uses `iket` as shorthand for that module. IKET API calls should be placed inside `@cute.kernel` code. Instrumentation in host-side Python code does not create in-kernel events.

IKET has three basic concepts:

- An event is one warp-level runtime record emitted by a kernel. Each event records a timestamp and the metadata needed to identify it in the trace.
- A marker is a point annotation and emits one event.
- A range represents a duration and is usually built from two events: one at the start and one at the end. Each event can optionally carry one payload value to record a runtime variable.

| API | Purpose | Notes |
|----|----|----|
| `iket.mark(name)` | Emit a single timestamped marker. | Use for point events. |
| `iket.mark(name, payload)` | Emit a marker with a numeric payload. | The payload is stored with the event. |
| `iket.range_push(name)` | Start a stack-based range. | Closed by the next matching `iket.range_pop()` in LIFO order. |
| `iket.range_push(name, payload)` | Start a stack-based range with a numeric payload. | The payload is attached to the push event. |
| `iket.range_pop()` | End the most recent stack-based range. | Does not take a range name. |
| `iket.range_start(name)` | Start a token-based range and return a token. | Closed by `iket.range_end(token)`. |
| `iket.range_start(name, payload)` | Start a token-based range with a numeric payload. | Closed by `iket.range_end(token, payload)` with matching payload type. |
| `iket.range_end(token)` | End a token-based range. | The token must come from `iket.range_start` or `iket.sentinel_token`. |
| `iket.range_end(token, payload)` | End a token-based range with a numeric payload. | The corresponding `iket.range_start` call must also have a payload, and the payload types must match. |
| `iket.sentinel_token(name)` | Create a token without a real runtime event for cross-iteration ranges. | Use it when `range_end` appears before the later `range_start` in source order. |

#### Choosing a Range API

CuTe DSL IKET provides two valid range-pairing models. Choose the one that makes the pairing easiest to see in the kernel source.

Use `range_push` / `range_pop` when the range is naturally nested and the push and pop calls can stay in the same structured scope. This is often the clearest shape for phase-style instrumentation such as setup, mainloop, wait, issue, and epilogue ranges.

Use `range_start` / `range_end` when an explicit handle makes the pairing clearer. This can be useful when a range ends at a later synchronization point, crosses an iteration boundary, or has multiple mutually exclusive close sites.

#### Payloads

Payloads attach a runtime value to an event. They are useful for recording values such as loop indices, block coordinates, or small computed metrics. Supported payloads include Python boolean, integer, and floating-point literals, plus CuTe DSL numeric and index scalar values. Do not use tensors, tuples, or other aggregate values as payloads. Prefer warp-uniform payload values, such as loop indices or block coordinates, when they describe the event clearly. For example:

``` python
for k_tile in cutlass.range(k_tile_count):
    cute.experimental.iket.range_push("k_tile", k_tile)
    # Work for this K tile.
    cute.experimental.iket.range_pop()
```

IKET events are warp-level events. If active threads in the participating warp evaluate a payload expression to different values, the dumped payload value is from the first active thread. To record the payload value from a specific thread, guard the IKET call with a predicate such as `if tidx == 0:`. For range endpoints, guard paired endpoints consistently.

Plain Python integer literals are emitted as 32-bit integer payloads, and plain Python floating-point literals are emitted as 32-bit floating-point payloads. Use explicit CuTe DSL scalar types for 64-bit literal payloads:

``` python
cute.experimental.iket.mark("large_count", cutlass.Int64(0x100000000))
cute.experimental.iket.mark("scale", cutlass.Float64(3.141592653589793))
```

For token-based ranges, the start and end payload forms must match. It is not allowed to start a range with a payload and end it without one, or to use different payload types between `range_start` and `range_end`.

### Example Instrumentation Patterns

The examples below use `cute.experimental.iket` inside `@cute.kernel` code. IKET calls in host-side Python wrappers do not emit in-kernel events.

#### Before Adding Events

Start by identifying the kernel body and the work you want to measure.

1.  Find the `@cute.kernel` function. Host-side `@cute.jit` functions and launch wrappers are useful context, but IKET instrumentation should be placed in device kernel code.
2.  Split the kernel into natural phases. For a GEMM-shaped kernel this may be setup, TMA or copy issue, mainloop, MMA, waits, and epilogue. Other kernels should use names that match their own algorithmic phases.
3.  Note warp-specialized regions such as `if warp_idx == 0:` or `if is_leader_cta:`. Put both ends of a range inside the same role or guard when the work is role-specific.
4.  Identify asynchronous work. For example, TMA copies, `cp.async`-style copies, WGMMA or MMA issue, and pipeline or mbarrier operations may have separate issue and completion points.

#### Coarse Phase Timing

Begin with a small number of coarse ranges. This provides orientation in the trace and keeps overhead manageable while you decide where more detail is needed.

``` python
@cute.kernel
def kernel(...):
    user_warp_lifetime = cute.experimental.iket.range_start(
        "user_warp_lifetime"
    )

    cute.experimental.iket.range_push("setup")
    # Allocate/register fragments, partition tensors, initialize pipelines.
    cute.experimental.iket.range_pop()  # setup

    cute.experimental.iket.range_push("mainloop")
    for k_tile in cutlass.range(k_tile_count):
        # Main loop body.
    cute.experimental.iket.range_pop()  # mainloop

    cute.experimental.iket.range_push("epilogue")
    # Convert accumulators and store results.
    cute.experimental.iket.range_pop()  # epilogue

    cute.experimental.iket.range_end(user_warp_lifetime)
```

For warp-specialized code, place the range inside the guard for the warp that does the work:

``` python
if warp_idx == tma_warp_id:
    cute.experimental.iket.range_push("tma_main")
    # TMA producer work.
    cute.experimental.iket.range_pop()  # tma_main

if warp_idx == mma_warp_id:
    cute.experimental.iket.range_push("mma_main")
    # MMA consumer work.
    cute.experimental.iket.range_pop()  # mma_main
```

#### Timing Waits and Async Work

For asynchronous operations, decide whether the range measures issue time or completion/wait time. An event immediately after an async issue point measures issue-side timing. Completion is usually observed at a pipeline or mbarrier wait.

To measure issue time:

``` python
issue_token = cute.experimental.iket.range_start("tma_issue", k_tile)
cute.copy(tma_atom, src_tensor, dst_tensor, tma_bar_ptr=barrier)
cute.experimental.iket.range_end(issue_token, k_tile)
```

To measure wait time:

``` python
cute.experimental.iket.range_push("ab_wait")
ab_full = ab_consumer.wait_and_advance()
cute.experimental.iket.range_pop()
```

The same wait pattern can be used around pipeline acquire calls, mbarrier waits, allocator waits, or other synchronization points whose return marks completion of the waited-for work.

#### Cross-Iteration Wait-Boundary Timing

Some pipelined loops start work for iteration `N` and observe the next useful boundary for that work in iteration `N + 1`. For example, the next iteration may reach a pipeline wait, mbarrier wait, or other `wait_and_advance`-style call before the previous tile's range should close. Use `sentinel_token` to initialize the token before the loop. Creating the sentinel token emits no runtime event. Calling `range_end` on the initial sentinel token is valid and emits no runtime event; after the variable is replaced by a token from `range_start`, `range_end` emits the runtime end event for that real range.

``` python
iter_token = cute.experimental.iket.sentinel_token("mma_k_tile")

for k_tile in cutlass.range(k_tile_count):
    ...  # some setup codes for k_tile
    ab_full = ab_consumer.wait_and_advance()

    # Close the previous tile only after this wait boundary is reached.
    if k_tile > 0:
        cute.experimental.iket.range_end(iter_token)

    iter_token = cute.experimental.iket.range_start("mma_k_tile")
    # Work for this tile.
    cute.gemm(tiled_mma, tCtAcc, tCrA, tCrB, tCtAcc)
    ab_full.release()

if k_tile_count > 0:
    ...  # final drain or synchronization boundary for the last tile
    cute.experimental.iket.range_end(iter_token)
```

Use this pattern only when the cross-iteration boundary is meaningful. For a simple per-iteration range whose start and end are both inside the same loop iteration, a push/pop pair inside the loop is simpler:

``` python
for k_tile in cutlass.range(k_tile_count):
    cute.experimental.iket.range_push("k_tile", k_tile)
    # Work for this tile.
    cute.experimental.iket.range_pop()
```

#### Warp-Specialized Mainloop Example

The following skeleton shows how to layer ranges by role and by loop level in a warp-specialized kernel. Adapt the role names and phase names to the actual kernel.

``` python
@cute.kernel
def kernel(...):
    user_warp_lifetime = cute.experimental.iket.range_start(
        "user_warp_lifetime"
    )

    # Work shared by all participating warps.
    cute.experimental.iket.range_push("prologue")
    # Tensor partitioning, pipeline setup, scheduler setup.
    cute.experimental.iket.range_pop()  # prologue

    if warp_idx == tma_warp_id:
        cute.experimental.iket.range_push("tma_main")
        while work_tile.is_valid_tile:
            cute.experimental.iket.range_push("tma_tile")

            for k_tile in cutlass.range(k_tile_count):
                cute.experimental.iket.range_push("tma_k_tile", k_tile)
                ...  # some setup codes for k_tile

                cute.experimental.iket.range_push("tma_acquire")
                ab_empty = ab_producer.acquire_and_advance()
                cute.experimental.iket.range_pop()  # tma_acquire

                issue_token = cute.experimental.iket.range_start(
                    "tma_issue", k_tile
                )
                cute.copy(tma_a_atom, tAgA, tAsA, tma_bar_ptr=ab_empty.barrier)
                cute.copy(tma_b_atom, tBgB, tBsB, tma_bar_ptr=ab_empty.barrier)
                cute.experimental.iket.range_end(issue_token, k_tile)

                cute.experimental.iket.range_pop()  # tma_k_tile

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()
            cute.experimental.iket.range_pop()  # tma_tile

        ab_producer.tail()
        cute.experimental.iket.range_pop()  # tma_main

    if warp_idx == mma_warp_id:
        cute.experimental.iket.range_push("mma_main")
        while work_tile.is_valid_tile:
            cute.experimental.iket.range_push("mma_tile")

            for k_tile in cutlass.range(k_tile_count):
                cute.experimental.iket.range_push("mma_k_tile", k_tile)

                cute.experimental.iket.range_push("ab_wait")
                ab_full = ab_consumer.wait_and_advance()
                cute.experimental.iket.range_pop()  # ab_wait

                cute.experimental.iket.range_push("mma_issue")
                cute.gemm(tiled_mma, tCtAcc, tCrA, tCrB, tCtAcc)
                cute.experimental.iket.range_pop()  # mma_issue

                ab_full.release()
                cute.experimental.iket.range_pop()  # mma_k_tile

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()
            cute.experimental.iket.range_pop()  # mma_tile

        cute.experimental.iket.range_pop()  # mma_main

    cute.experimental.iket.range_end(user_warp_lifetime)
```

This style makes each role visible in the trace. Prefixing names with the role such as `tma_` and `mma_` also makes JSON output easier to filter.

#### Instrumentation Guidelines

Consider the following when placing IKET calls. These choices affect what the profiler can reconstruct from the runtime events, so unclear or mismatched instrumentation may produce results that do not match the intended measurement or even cause the profiler to fail when postprocessing the data.

- Every dynamic `range_push` has exactly one matching `range_pop` on each participating warp execution path, in LIFO order.
- Every dynamic non-sentinel `range_start` is closed by `range_end` on each participating warp execution path. It is valid to close the same token in multiple mutually exclusive branches, as long as each executed path closes it once.
- Start and end points are in the same warp role and compatible control-flow path. Do not start a range in one thread-divergent branch and close it in another. Violating this may cause undefined profiling results.
- Payload-bearing token ranges use matching payload types at start and end.
- Event names are at most 32 characters.
- Reuse the same descriptive name for the same recurring phase inside a loop. This creates many runtime events but only one unique marker or range name.
- When more than 30 unique marker or range names are used, instrumentation overhead increases.
- Avoid high-frequency events in innermost unrolled loops unless that detail is necessary. IKET events can affect compiler scheduling and can create large traces.
- Do not put IKET range operations inside `cutlass.range(..., prefetch_stages=...)`. That loop form is not currently supported for IKET range instrumentation.

### Profiling with the `run-iket` Tool

The `run-iket` tool is installed with the `nvidia-cutlass-dsl` package. During profiling, `run-iket` automatically enables IKET lowering for JIT-compiled kernels so that the final kernel contains instrumentation code. The application command must appear after `--` so that workload arguments are not parsed as profiler arguments.

``` bash
run-iket \
    --output-dir ./iket_output \
    --clobber \
    profile \
    --postprocess all \
    -- \
    python my_kernel.py
```

Important options:

| Option | Description |
|----|----|
| `--output-dir <dir>` | Directory for profiler output and intermediate files. |
| `--clobber` | Remove an existing output directory and create a new one without prompting. |
| `profile` | Start a profiling run. |
| `--postprocess perfetto` | Generate a Perfetto timeline trace. |
| `--postprocess json` | Generate JSON output for script-based analysis. |
| `--postprocess all` | Generate both Perfetto and JSON output. |

#### Output Files

With `--postprocess perfetto`, `run-iket` writes one or more `*.pftrace` files under the output directory. With `--postprocess json`, it also writes JSON traces containing the collected ranges, markers, timestamps, and payloads.

The exact filenames may include the profiled process ID. If the workload uses multiple processes or GPUs, `run-iket` may produce separate traces that are **NOT** aligned to a single global timeline.

For a single-process run with `--postprocess all`, the output directory contains files shaped like this:

``` text
iket_output/
  *.pftrace   # Open this in Perfetto UI.
  *.json      # Use this for script-based analysis.
```

There may also be profiler intermediate files in the same directory. Start with the `*.pftrace` file for visual inspection, then use the JSON file when you need scripted filtering or aggregation.

#### JSON Trace Shape

The JSON output is intended for script-based analysis. Its schema is also experimental and may change in future releases. A trace is organized around profiled kernel launches, with ranges, markers, warp locations, payload fields, and warp lifetimes. A simplified trace looks like this:

``` json
{
  "launches": [
    {
      "gridId": 0,
      "kernelName": "my_kernel",
      "ranges": [
        {
          "rangeName": "mainloop",
          "rangeScope": 0,
          "startTs": 1000,
          "endTs": 2500,
          "warpLocs": [
            {
              "smId": 0,
              "tpcId": 0,
              "gpcId": 0,
              "ctaId": [0, 0, 0],
              "warpId": 0
            }
          ],
          "internalEvents": []
        }
      ],
      "markers": [
        {
          "markerName": "checkpoint",
          "timestamp": 1500,
          "location": {
            "smId": 0,
            "tpcId": 0,
            "gpcId": 0,
            "ctaId": [0, 0, 0],
            "warpId": 0
          },
          "payloadType": 0,
          "payloadVal": 0
        }
      ],
      "warpLifetimes": [
        {
          "startTs": 900,
          "endTs": 3800,
          "warpLocation": {
            "smId": 0,
            "tpcId": 0,
            "gpcId": 0,
            "ctaId": [0, 0, 0],
            "warpId": 0
          }
        }
      ]
    }
  ]
}
```

Important fields include:

- `launches[]`: profiled kernel launches.
- `gridId` and `kernelName`: launch identity and kernel name.
- `ranges[]`: duration ranges with `rangeName`, `startTs`, `endTs`, `rangeScope`, and one or more `warpLocs`. Use `endTs - startTs` as the range duration.
- `markers[]`: point events with `markerName`, `timestamp`, `location`, and optional payload fields such as `payloadType` and `payloadVal`.
- `warpLifetimes[]`: active spans for warps observed during the profiled launch.

Timestamp values are in a trace-local timebase. Compare timestamp differences within one trace, but do not compare absolute timestamp values across separate traces. `rangeScope` and `payloadType` are profiler metadata fields whose exact numeric values are experimental. A `warpLocs` entry identifies the GPU location for a warp that emitted or participated in the range.

### Viewing a Trace in Perfetto

Open the generated `*.pftrace` file in the Perfetto UI at <https://ui.perfetto.dev/>.

The basic workflow is:

1.  Open the trace file.
2.  Search for the profiled kernel or zoom into the kernel region. Use the `W` / `A` / `S` / `D` keys on your keyboard to pan and zoom.
3.  Expand the relevant tracks.
4.  Click a marker or range to inspect its name, timing, and payload values.

The trace is organized around profiled kernel launches and the warp-level IKET records collected inside those launches. After expanding a kernel region, look for tracks grouped by the recorded GPU location, such as CTA and warp identity. The visible marker and range names come from the strings passed to the IKET API in the kernel.

The image below shows a more complete expanded trace from the GEMM tutorial. Use it as a guide to the track structure:

- The left track hierarchy is expanded by GPU location, then by CTA and warp.
- `WarpLifeTime` tracks are generated automatically for kernels that contain IKET instrumentation.
- Long ranges under each warp show user-instrumented phases such as `kernel_e2e`, `epi_main`, `mma_tile`, and `tma_tile`. Token-based `range_start` / `range_end` ranges use separate tracks for each range name.
- Stack-based `range_push` / `range_pop` ranges are shown on `StackedRanges` tracks, and markers are shown on `Marker` tracks.
- Very short colored blocks and marker glyphs represent fine-grained events. Payload values may appear in event labels and are also available by clicking the event and inspecting the details panel.

<figure>
<img src="images/fp16_gemm_4_iket_instrument.png" style="width:100.0%" alt="Example IKET timeline in Perfetto UI" />
<figcaption>Example IKET timeline in Perfetto UI. The exact tracks and event names depend on the kernel instrumentation and workload. This trace was generated from <code>examples/python/CuTeDSL/dsl_tutorials/fp16_gemm_4_iket.py</code>.</figcaption>
</figure>

Trace viewing is powered by the Perfetto UI (<https://ui.perfetto.dev/>), part of the Perfetto project (<https://perfetto.dev/>) licensed under the Apache License 2.0. Perfetto UI is provided by a third-party site. This product does not modify or redistribute Perfetto UI code. Perfetto is only a viewer for the generated trace. The trace content comes from the IKET instrumentation emitted by the CuTe DSL kernel and collected by `run-iket`.

### Assumptions, Limitations, and Impact

The `run-iket` profiler assumes well-formed instrumentation and sufficiently convergent execution within participating warps. IKET events are warp-level records, so placement is easiest to interpret when all participating threads in a warp follow the same instrumentation path.

#### Range Pairing and Warp Divergence

For token-based ranges, every dynamic non-sentinel `range_start` must be closed by `range_end` on each participating warp execution path. A token may be closed in multiple mutually exclusive branches, but each executed path should close that dynamic range once. For stack-based ranges, every `range_push` must be balanced by `range_pop` using LIFO stack semantics.

Avoid these patterns:

- Starting a range in one thread-divergent branch and ending it in another branch.
- Returning or otherwise exiting early between paired range endpoints.
- Emitting different push/pop nesting on different warp execution paths.

If a range pair is inside a branch where different threads in a warp diverge, the trace may be incomplete or may reflect serialized divergent execution. Incorrectly paired `range_start` / `range_end` or `range_push` / `range_pop` calls may cause profiling to fail, may produce incorrect timeline visualization, or may cause undefined profiling results. Warp-uniform placement gives cleaner and easier-to-interpret results.

#### Event Name Count and Overhead

Unique user names include marker names and range names from `mark`, `range_start`, and `range_push`. Repeatedly emitting the same name is supported and is the expected way to record a loop phase or recurring marker.

Kernels with more than 30 unique marker or range names may use a wider event encoding during IKET lowering, which can increase instrumentation overhead. Keep the number of unique names as small and stable as practical, especially in performance-sensitive kernels.

Event names may use arbitrary characters and must be at most 32 characters. Longer names are not supported; use a shorter stable name instead.

#### Timestamp Semantics

An IKET timestamp records the instrumentation point itself. The timer granularity is 32 ns. If a range starts and ends very close together, its start and end timestamps may be identical. In the final Perfetto visualization, such a very short range may look similar to a marker. For asynchronous operations such as TMA copies, placing an end event immediately after the issue point usually measures issue-side timing, not completion. To measure completion or wait time, place the corresponding range endpoint around the synchronization or wait point where completion is observed.

#### Workload Size

`run-iket` profiles the whole workload it launches. It does not currently support selecting a smaller profiling or capture window. Prefer small workloads with a limited number of instrumented kernel launches while collecting IKET traces. A kernel with many IKET events can generate a large amount of data because records are collected per warp, and often across many CTAs. Large workloads, such as workloads with many instrumented kernel launches, may run much more slowly under the profiler or may run out of memory.

#### Kernel Launch and Overlap Timing

`run-iket` can collect traces from workloads that launch multiple instrumented kernels. However, IKET is intended for in-kernel timing. Do not use the IKET trace to measure host-side launch latency, inter-kernel launch gaps, kernel overlap, or CPU/GPU scheduling behavior. These workload-level analyses are not IKET's target use case, and the corresponding timing views include additional `run-iket` overhead and are not currently optimized for launch or overlap analysis. When a trace contains multiple kernels, interpret the IKET markers and ranges within each individual kernel launch.

Use NVIDIA Nsight Systems in a separate run when you need accurate kernel launch latency, kernel overlap, CPU/GPU scheduling, or whole-application timeline analysis.

#### Profiler Compatibility

`run-iket` cannot run at the same time as NVIDIA Nsight Compute, NVIDIA Nsight Systems, or other CUPTI-based profiling and tracing tools due to conflicts over driver profiling resources. Do not run them together on the same workload. Use separate runs when collecting IKET traces and other profiler outputs.

#### Buffer Sizing

`run-iket` uses multiple profiling passes to allocate device-side trace buffers. An initial pass estimates how much buffer space the workload needs, and a later pass allocates memory with some margin and collects the timestamp and payload records.

This assumes the number of emitted IKET records per warp is reasonably stable between those passes. If a kernel emits a very different number of records per warp between the sizing pass and the collection pass, the trace may contain incorrect data, and the workload may fail with an illegal memory access. Prefer deterministic profiled workloads and avoid data-dependent instrumentation rates that vary substantially between runs.

#### Unsupported Prefetch-Stage Loops

IKET range operations are not currently supported inside a `cutlass.range` loop that uses `prefetch_stages`, such as `cutlass.range(..., prefetch_stages=...)`. This is a known limitation of the prefetch-stage loop form. Place IKET range instrumentation outside that loop, or use a loop form without prefetch stages when profiling that region.

#### Compiler and Runtime Impact

IKET events do slightly change the generated kernel code. Avoid placing too many events in innermost hot loops, especially loops the compiler may unroll. IKET markers and ranges can act partly like code-motion barriers and may reduce loop interleaving optimizations after unrolling.

High-frequency instrumentation, many unique range or marker names (especially more than 30), payloads, many kernel launches, or large workloads can increase overhead, overflow buffers, alter compiler behavior, or create large traces. Payloads also increase the amount of stored trace data.

IKET profiling may also add fixed per-kernel entry and exit overhead. This overhead can affect the trace-reported kernel duration, but it is not shown as a separate IKET marker or range on the Perfetto timeline.

IKET profiling adds CPU-side overhead in addition to in-kernel overhead. Host wall-clock measurements outside CUDA Driver/API timing are therefore not a clean measure of kernel event overhead.

### Performance Overhead Guidance

The primary way to compare profiled overhead is to use kernel durations from the IKET Perfetto trace. Do not use application wall-clock timing or timing taken outside CUDA Driver/API boundaries as the measure of event overhead.

For a reasonable comparison:

1.  Instrument the target kernel.
2.  Run `run-iket` with a minimal instrumentation variant that emits one instrumented event site, so the profiler recognizes and profiles the kernel.
3.  Run `run-iket` again with many instrumented event sites or with the intended full event set in the instrumented kernel.
4.  Compare the trace-reported kernel execution times in Perfetto.

The difference between those trace-reported kernel times estimates incremental event overhead. To report amortized overhead, divide the runtime delta by the number of executed instrumentation points, preferably on a per-warp basis.

As a separate compiler and binary sanity check, you can compare an uninstrumented kernel against the same kernel compiled with IKET instrumentation enabled but not profiled. To build that IKET-enabled binary outside a `run-iket` profiling run, use `CUTE_DSL_COMPILER_OPT=iket` or `cute.compile(..., options="iket")`. This is not a replacement for the Perfetto-based profiled-overhead comparison above.

Payloads can significantly increase overhead and trace volume. 64-bit payloads are more expensive than no-payload or 32-bit payload events.

### How It Works

At compile time, CuTe DSL emits `iket.*` IR operations and event metadata for the IKET API calls in the kernel. By default, the compiler strips the `iket.*` operations before lowering, so IKET calls do not contribute instrumentation code to the final kernel.

When IKET lowering is enabled, either explicitly with `CUTE_DSL_COMPILER_OPT=iket` or `cute.compile(..., options="iket")`, or automatically by `run-iket` during a profiling run, lowering emits placeholder instrumentation and metadata into the compiled kernel.

During profiling, `run-iket` prepares the kernel for collection, patches the placeholder instrumentation at runtime, collects timestamp and payload records, and postprocesses the records into Perfetto and JSON output.

### Troubleshooting

Empty or missing trace
Confirm that the workload is launched under `run-iket` and that the instrumented kernels are JIT-compiled in that profiled process. If running without `run-iket`, confirm that `CUTE_DSL_COMPILER_OPT=iket` is set or that the kernel is compiled with `options="iket"`.

Trace is very large or profiling fails
Reduce the number of instrumented kernel launches, reduce event frequency, or reduce payload use. If the workload can legitimately emit more records per warp than the profiler detects automatically, increase `--max-ts-cnt-per-warp <N>` to reserve space for up to `N` events per warp. Choose `N` above the largest expected number of marker, range-start, range-end, range-push, and range-pop events emitted by one warp in one kernel launch.

Expected events do not appear
Confirm the IKET calls are inside `@cute.kernel` code and are reachable on the execution path being profiled. Also check that paired range endpoints are not split across divergent branches. If the workload caches a compiled kernel or executor, make sure that compilation happens inside the `run-iket` profiling process or clear the application-level compiled kernel cache before profiling.

Unexpected timing around asynchronous work
Move range endpoints to the wait or synchronization point that observes completion.

Inspect generated IR
Use existing CuTe DSL debugging options such as `CUTE_DSL_KEEP=ir` or `CUTE_DSL_PRINT_IR=1` to inspect generated IR when diagnosing whether IKET operations were emitted.

---

<!-- source: cute_dsl_general/autotuning_gemm.rst -->

## Guidance for Auto-Tuning

Numerous GEMM kernel code examples are offered within our codebase. When integrating these kernels into frameworks, auto-tuning becomes essential for achieving optimal performance. This involves selecting the appropriate kernel parameters based on the inputs of real applications. Next, we'll briefly introduce some tips on how to perform auto-tuning.

The auto-tuning process typically involves the following steps:

1.  Define search space
2.  Benchmark each configuration and select the kernel with the best performance
3.  Enable caching to reduce the tuning cost

The search space defines the valid combinations of kernel parameters that can be used to run the kernels. Different inputs (shapes, data types, etc.) typically require different kernel parameters to achieve optimal performance. The search space is related to the kernel. We take the Blackwell GEMM persistent kernel as an example. The search space is as follows:

- `mma_tiler_mn`: Defines the dimensions of the matrix tile that each Matrix Multiply-Accumulate (MMA) instruction processes in a single operation.
- `cluster_shape_mn`: Specifies the number of CTAs along each dimension within a cluster. Refer [Parallel Thread Execution ISA documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions) for the possible mma tiler size and cluster shape for different tensor data types.
- `use_2cta_instrs`: Whether to utilize Blackwell's 2 CTA instructions for MMA/Copy.
- `use_tma_store`: Whether to use Tensor Memory Access (TMA) instructions to store the result back to global memory.

After defining the search space, we could traverse all parameter combinations to find the optimal kernel. The `autotune_gemm` function below demonstrates a simple exhaustive search approach - it iterates through configurations, compiles and benchmarks each kernel, and returns the best performing one. Since kernel compilation incurs overhead, it's important to cache and reuse compiled kernels to minimize host launch latency. CuTe DSL facilitates this through its separate compilation and execution workflow. More details can be found in JIT Caching. As demonstrated in the `autotune_gemm` function (between the `begin of cache the compiled GEMM kernel` and `end of cache the compiled GEMM kernel` comments), we can use `cute.compile()` to compile a kernel once, cache the compiled result, and reuse the cached JIT executor for multiple kernel executions. We could maintain a global configuration-to-kernel dictionary (`config_kernel_dict`) to cache the compiled GEMM kernels, where each key (`kernel_cache_key`) uniquely identifies a kernel based on its characteristics. Usually we could use the {dtype + kernel configs} as the cached key for GEMM compilation. For example,

``` python
kernel_cache_key = f"{ab_dtype}x{c_dtype}x{acc_dtype}x{use_2cta_instrs}x{mma_tiler}x{cluster_shape_mn}x{use_tma_store}"
```

If the input tensor's layout is static, we should add the shape in the cached key too. Users can customize the `benchmark` function to measure kernel execution time. For stable and reliable performance measurements:

1.  Run a few warmup iterations (e.g., 5-10) to stabilize GPU temperature
2.  Execute multiple timed iterations (e.g., 100-1000) for statistical significance
3.  Use CUDA events and synchronization for precise timing
4.  Lock GPU frequencies (SM and memory frequencies) with nvidia-smi
5.  Process results by removing outliers and using min/avg statistics as measurements.

This ensures reliable kernel selection through proper benchmarking.

``` python
# get the best GEMM kernel for given input tensors
def autotune_gemm(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    stream: cuda.CUstream,
    use_2cta_instrs_list: List[bool] = [True],
    use_tma_store_list: List[bool] = [True],
    mma_tiler_m_list: List[int] = [256],
    mma_tiler_n_list: List[int] = [256],
    cluster_shape_m_list: List[int] = [2],
    cluster_shape_n_list: List[int] = [1],
):
    best_kernel = None
    min_time = float("inf")
    # traverse the search space
    for use_2cta_instrs in use_2cta_instrs_list:
        for use_tma_store in use_tma_store_list:
            for mma_tiler_mn in product(mma_tiler_m_list, mma_tiler_n_list):
                for cluster_shape_mn in product(cluster_shape_m_list, cluster_shape_n_list):
                    acc_dtype = cutlass.Float32
                    hardware_info = cutlass.utils.HardwareInfo()
                    max_active_clusters = hardware_info.get_max_active_clusters(
                        cluster_shape_mn[0] * cluster_shape_mn[1]
                    )
                    # instance a GEMM kernel
                    gemm = PersistentDenseGemmKernel(
                        acc_dtype,
                        use_2cta_instrs,
                        mma_tiler_mn,
                        cluster_shape_mn,
                        use_tma_store,
                    )
                    # begin of cache the compiled GEMM kernel
                    if kernel_cache_key not in config_kernel_dict:
                        # compile gemm kernel
                        compiled_gemm = cute.compile(
                            gemm,
                            a,
                            b,
                            c,
                            max_active_clusters,
                            stream,
                        )
                        config_kernel_dict[kernel_cache_key] = compiled_gemm
                    else:
                        compiled_gemm = config_kernel_dict[kernel_cache_key]
                    # end of cache the compiled GEMM kernel
                    try:
                        # define a benchmark function to measure the execution time of the compiled GEMM kernel
                        cur_time = benchmark(
                            partial(compiled_gemm, a, b, c, stream),
                        )
                    except Exception as e:
                        print(f"Execution error: {e}")
                        cur_time = float("inf")
                    if cur_time < min_time:
                        min_time = cur_time
                        best_kernel = compiled_gemm
    if best_kernel is None:
        raise ValueError("No best kernel found")
    return best_kernel
```

This brute-force approach ensures we could find the optimal parameters, though at the cost of trying every possibilities. For more advanced use cases, users can explore sophisticated optimization techniques like search space pruning and genetic algorithms to reduce tuning overhead and discover better configurations more efficiently.

To further optimize tuning performance, we can utilize caching mechanisms to avoid redundant computations. We could cache the tuning results in a input-to-kernel dictionary (e.g., `input_kernel_dict`). When processing inputs with matching `config_key` values, the cached kernel can be reused directly without re-tuning. The `config_key` is related with the input tensor's characteristics, such as the shape, data type, etc. The setup of `config_key` is very flexible, users can customize it based on their own application. For instance, if the data type is fixed in users' application, we could use the input tensor's shape as the key, i.e., `(m, n, k)`. To further reduce tuning overhead, we could consider using a simplified key like `config_key = (power_of_2(m), power_of_2(n), power_of_2(k))`, where `m`, `n`, and `k` are rounded up to the nearest power of 2. This simplification can significantly reduce the number of unique keys while still maintaining good performance in most cases. However, it's important to validate that this approximation doesn't negatively impact performance for your specific use case.

``` python
config_key = (m, n, k)
if config_key in input_kernel_dict:
    compiled_gemm = input_kernel_dict[config_key]
else:
    compiled_gemm = autotune_gemm(...)
    input_kernel_dict[config_key] = compiled_gemm
# launch gemm kernel
compiled_gemm(a_tensor, b_tensor, c_tensor, stream)
```

By following the methods above, you can customize your own auto-tuner to find the optimal GEMM kernel configuration for specific matrix dimensions and data types, significantly improving computational performance for models.

---

<!-- source: cute_dsl_general/notebooks.rst -->

## Educational Notebooks

A number of notebooks for educational purposes are provided in the [CUTLASS GitHub repository](https://github.com/NVIDIA/cutlass). A list with handful links is given below:

- ["Hello world"](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/hello_world.ipynb)
- [Printing](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/print.ipynb)
- [Data Types Basics](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/data_types.ipynb)
- [Tensors](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/tensor.ipynb)
- [The TensorSSA Abstraction](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/tensorssa.ipynb)
- [Layout Algebra](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/cute_layout_algebra.ipynb)
- [Element-wise Add Tutorial](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/elementwise_add.ipynb)
- [Using CUDA Graphs](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/cuda_graphs.ipynb)

---

<!-- source: deprecation.rst -->

## Deprecation Policy

### Purpose

The goal of this policy is to evolve the DSL and its APIs while keeping user programs stable. Features or APIs are deprecated only when they are redundant, unsafe, or block better designs.

### Deprecation Process

**Step 1 — Soft Deprecation**

When a feature is considered for removal, it is first annotated with the `@deprecated` decorator or `DeprecationWarning` and documented with a suggested alternative. At this stage, the feature continues to work normally.

Users are encouraged to provide feedback and describe their use cases. If there is strong justification, we may keep or redesign the feature.

**Step 2 — Removal (the subsequent release)**

If no valid use cases remain, the deprecated feature will be removed in the following **minor** release.

> [!NOTE]
> The release version follows the format `<major>.<minor>.<patch>`.

### Communication

All deprecations are announced through:

- This page
- In-code warning messages

### Soft Deprecations

**Version 4.2.1**

- `cute.arch.warpgroup_reg_alloc` and `cute.arch.warpgroup_reg_dealloc` → Scheduled for deprecation. Use `cute.arch.setmaxregister_increase` and `cute.arch.setmaxregister_decrease` instead.
- `alignment` argument in `CooperativeGroup` constructor → Scheduled for deprecation. It was unused; no replacement is suggested.
- `cute.AddressSpace` → Supported today as a quiet compatibility alias. It is scheduled for deprecation in a later release, where accesses will start emitting a deprecation warning. Use `cutlass.AddressSpace` for new code. The lowercase address-space members such as `gmem`, `smem`, `rmem`, `tmem`, and `dsmem` keep the same spelling.

### Deprecated Features

*(None currently.)*

---

<!-- source: cute_dsl_general/compile_with_tvm_ffi.rst -->

## Compile with TVM FFI

Apache TVM FFI is an open ABI and FFI for machine learning systems. More information can be found in the [official documentation](https://tvm.apache.org/ffi/).

To install TVM FFI, you can run the following command:

``` bash
pip install apache-tvm-ffi
# optional package for improved torch tensor calling performance
pip install torch-c-dlpack-ext
```

In CuTe DSL, TVM FFI can be enabled as an option for JIT-compiled functions. Using TVM FFI can lead to faster JIT function invocation and provides better interoperability with machine learning frameworks (e.g., directly take `torch.Tensor` as arguments).

Enable Apache TVM FFI in CuTe DSL ------------------------------

First, install the `tvm-ffi` package by following its [installation guide](https://tvm.apache.org/ffi/#installation).

There are two ways to enable TVM FFI in CuTe DSL:

1.  Use the `options` argument in `cute.compile` to specify the TVM FFI option. For example:

``` python
# Assuming you have defined a function `add` decorated with @cute.jit
def example_compile():
   a_torch = torch.randn(10, 20, 30).to(torch.float16)
   b_torch = torch.randn(10, 20, 30).to(torch.float16)
   a_cute = cute.runtime.from_dlpack(a_torch, enable_tvm_ffi=True).mark_layout_dynamic()
   b_cute = cute.runtime.from_dlpack(b_torch, enable_tvm_ffi=True).mark_layout_dynamic()

   compiled_add = cute.compile(add, a_torch, b_torch, options="--enable-tvm-ffi")
```

Note that the object returned by `cute.compile` is a Python function specific to TVM FFI.

2.  Alternatively, you can enable TVM FFI globally by setting the environment variable `CUTE_DSL_ENABLE_TVM_FFI=1`. Please note that this setting will apply to all JIT compilations within the environment.

### Minimizing Host Overhead

Eager kernel invocation overhead on the CPU host can sometimes become a bottleneck for latency-sensitive applications. TVM FFI can help greatly reduce this overhead. To maximize performance benefits, we recommend setting up your workflow as follows (detailed instructions are provided in subsequent sections):

- **Compile the kernel with TVM FFI enabled.**
- **Declare shape constraints using fake tensors** and reuse the compiled function throughout your execution.
- **Pass PyTorch tensors directly** to the compiled function to avoid explicit DLPack conversion.
- **Use the environment stream flag** to implicitly pass the current PyTorch stream.
- **Rely on compiled argument validation** instead of Python-side attribute validation, as TVM FFI functions perform fast compiled checks.

Following these steps can significantly reduce the host-side overhead of eager kernel execution. The sections below provide detailed examples and explanations for each step. You may find it helpful to refer back to this summary after you review the implementation details.

### Fake tensor for compilation

The TVM FFI function accepts DLPack-compatible tensors as arguments, such as those from torch or jax. However, during compilation, it is necessary to specify the tensors' dynamic properties in CuTe DSL. To clearly distinguish between the compilation phase and runtime, CuTe DSL provides a "fake tensor" that can be used for compilation. For example:

``` python
import cutlass.cute as cute
import torch

@cute.kernel
def device_add_one(a: cute.Tensor, b: cute.Tensor):
   threads_per_block = 128
   cta_x_, _, _ = cute.arch.block_idx()
   tid_x, _, _ = cute.arch.thread_idx()
   tid = cta_x_ * threads_per_block + tid_x
   if tid < a.shape[0]:
      b[tid] = a[tid] + 1.0

@cute.jit
def add_one(a: cute.Tensor, b: cute.Tensor):
   n = a.shape[0]
   threads_per_block = 128
   blocks = (n + threads_per_block - 1) // threads_per_block
   device_add_one(a, b).launch(
      grid=(blocks, 1, 1),
      block=(threads_per_block, 1, 1),
   )

def example_add_one():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   # compile the kernel with "--enable-tvm-ffi" option and example input tensors
   compiled_add_one = cute.compile(add_one, a_cute, b_cute, options="--enable-tvm-ffi")
   # now compiled_add_one is a TVM-FFI function that can be called with torch.Tensor as input
   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")
   compiled_add_one(a_torch, b_torch)
   print("result of b_torch after compiled_add_one(a_torch, b_torch)")
   print(b_torch)
```

The fake tensor is a placeholder that mimics the interface of a real tensor but does not hold real data or allow indexing. It is used in compilation or testing scenarios where only shape/type/layout information is needed. All attempts to access or mutate data will raise errors.

#### Note on Stride Order

Note that CuTe's convention is to write the stride order for dimensions from left to right, where a lower order number means higher priority. In the context of the `make_fake_compact_tensor` API, for shape `(2, 3, 4)` and stride order `(0, 1, 2)`, the stride is `(1, 2, 6)`. This is commonly known as column-major order. If you want to create a fake tensor with compact row-major order, you should explicitly pass in `stride_order=tuple(reversed(range(len(shape))))` to `make_fake_compact_tensor`. Alternatively, you can always precisely control the stride via the `stride` argument in the `make_fake_tensor` API.

### `cute.Tensor` adapter for TVM FFI

To adapt the `cute.Tensor` to the TVM FFI function, you can use the `cute.runtime.from_dlpack` function with the `enable_tvm_ffi=True` option or the environment variable `CUTE_DSL_ENABLE_TVM_FFI=1`. For example:

``` python
def example_from_dlpack():
   a_cute = cute.runtime.from_dlpack(a_torch, enable_tvm_ffi=True).mark_layout_dynamic()
   b_cute = cute.runtime.from_dlpack(b_torch, enable_tvm_ffi=True).mark_layout_dynamic()

   compiled_add_one(a_cute, b_cute)
```

Note that because the `cute.runtime.from_dlpack` function performs an explicit DLPack conversion, it is less efficient than passing the `torch.Tensor` directly. You can also use `cute.Tensor` as an argument hint for `cute.compile`.

``` python
compiled_add_one = cute.compile(add_one, a_cute, b_cute, options="--enable-tvm-ffi")
```

### Working with torch Tensors

As you may have noticed in the examples above, TVM FFI-compiled functions can directly accept `torch.Tensor` objects (and other DLPack-compatible tensors) as inputs. The resulting functions add minimal overhead, enabling faster eager invocations thanks to the optimized calling path.

### Working with Streams

In many cases, a CuTe kernel needs to run on a specific CUDA stream. CuTe DSL provides two ways to work with streams through TVM FFI. The first is to pass the stream explicitly as an argument. The following example demonstrates this approach; the function accepts `torch.cuda.Stream`, `CUstream` or any stream class that implements the CUDA stream protocol.

``` python
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream

@cute.kernel
def device_add_one(a: cute.Tensor, b: cute.Tensor):
   threads_per_block = 128
   cta_x_, _, _ = cute.arch.block_idx()
   tid_x, _, _ = cute.arch.thread_idx()
   tid = cta_x_ * threads_per_block + tid_x
   if tid < a.shape[0]:
      b[tid] = a[tid] + 1.0

@cute.jit
def add_one_with_stream(a: cute.Tensor, b: cute.Tensor, stream: CUstream):
   n = a.shape[0]
   threads_per_block = 128
   blocks = (n + threads_per_block - 1) // threads_per_block
   device_add_one(a, b).launch(
      grid=(blocks, 1, 1),
      block=(threads_per_block, 1, 1),
      stream=stream,
   )

def example_add_one_with_stream():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   # Fake stream is a placeholder for stream argument
   stream = cute.runtime.make_fake_stream()
   compiled_add_one = cute.compile(
      add_one_with_stream, a_cute, b_cute, stream, options="--enable-tvm-ffi"
   )
   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")
   torch_stream = torch.cuda.current_stream()
   compiled_add_one(a_torch, b_torch, torch_stream)
   torch_stream.synchronize()
   print("result of b_torch after compiled_add_one(a_torch, b_torch, torch_stream)")
   print(b_torch)
```

#### Using Environment Stream

The second option is to rely on the environment stream flag. Pass `use_tvm_ffi_env_stream=True` to `make_fake_stream` to mark the stream argument as an environment stream, which means it no longer needs to be provided explicitly. TVM FFI will automatically use its environment stream (i.e., the current PyTorch stream) as the stream argument. The example below demonstrates this flow:

``` python
def example_add_one_with_env_stream():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   # Fake stream is a placeholder for stream argument
   # we will use TVM FFI environment stream
   stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
   compiled_add_one = cute.compile(
      add_one_with_stream, a_cute, b_cute, stream, options="--enable-tvm-ffi"
   )
   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")
   torch_stream = torch.cuda.current_stream()
   with torch.cuda.stream(torch_stream):
      # no need to pass in the stream explicitly, env stream will be synced
      # to torch.cuda.current_stream() before the function call.
      compiled_add_one(a_torch, b_torch)
   torch_stream.synchronize()
   print("result of b_torch after compiled_add_one(a_torch, b_torch)")
   print(b_torch)
```

Using the environment stream flag both speeds up calls and simplifies integration with frameworks such as PyTorch, since no explicit stream parameter is required. We recommend using the environment stream flag to both simplify framework integration and minimize host-side calling overhead.

### Working with Tuples

TVM FFI functions can also accept tuples as arguments. Tuples can be recursively composed of the types that are supported by TVM FFI. The example below shows how to use tuples as arguments:

``` python
import torch
from cutlass import cute

@cute.kernel
def device_add_one(a: cute.Tensor, b: cute.Tensor, c: cute.Float32):
   threads_per_block = 128
   cta_x_, _, _ = cute.arch.block_idx()
   tid_x, _, _ = cute.arch.thread_idx()
   tid = cta_x_ * threads_per_block + tid_x
   if tid < a.shape[0]:
      b[tid] = a[tid] + c

@cute.jit
def add_one_with_tuple(a: Tuple[cute.Tensor, cute.Tensor, cute.Float32]):
   n = a[0].shape[0]
   threads_per_block = 128
   blocks = (n + threads_per_block - 1) // threads_per_block
   device_add_one(a[0], a[1], a[2]).launch(grid=(blocks, 1, 1), block=(threads_per_block, 1, 1))

def example_add_one_with_tuple():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   compiled_add_one = cute.compile(
      add_one_with_tuple, (a_cute, b_cute, cute.Float32(4)),
      options="--enable-tvm-ffi"
   )
   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")
   compiled_add_one((a_torch, b_torch, 5))
   print("result of b_torch after compiled_add_one((a_torch, b_torch, 5))")
   print(b_torch)

example_add_one_with_tuple()
```

### Working with Variadic Tuples

Sometimes it is helpful to annotate a tuple with no explicit element types. This can be useful to build up a generic template for a function that accepts a variable number of elements. The compiled function's signature will be determined by the tuple argument passed to the `cute.compile` function. The following example shows how to use a variadic tuple to build such a generic template.

``` python
import cutlass
import torch
from cutlass import cute

@cute.kernel
def device_add_one(a: cute.Tensor, b: cute.Tensor, extra_value: tuple):
   threads_per_block = 128
   cta_x_, _, _ = cute.arch.block_idx()
   tid_x, _, _ = cute.arch.thread_idx()
   tid = cta_x_ * threads_per_block + tid_x
   if tid < a.shape[0]:
      if cutlass.const_expr(len(extra_value) != 0):
            b[tid] = a[tid] + 1 + extra_value[0]
      else:
            b[tid] = a[tid] + 1

@cute.jit
def add_one_with_extra_value(a: cute.Tensor, b: cute.Tensor, extra_value: tuple):
   n = a.shape[0]
   threads_per_block = 128
   blocks = (n + threads_per_block - 1) // threads_per_block
   device_add_one(a, b, extra_value).launch(grid=(blocks, 1, 1), block=(threads_per_block, 1, 1))

def example_add_one_with_variadic_tuple():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   compiled_add_one_no_extra = cute.compile(
      add_one_with_extra_value, a_cute, b_cute, (),
      options="--enable-tvm-ffi"
   )
   compiled_add_one_with_extra = cute.compile(
      add_one_with_extra_value, a_cute, b_cute, (cute.Float32(4),),
      options="--enable-tvm-ffi"
   )
   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")
   compiled_add_one_no_extra(a_torch, b_torch, ())
   print("result of b_torch after compiled_add_one_no_extra(a_torch, b_torch, ())")
   print(b_torch)
   compiled_add_one_with_extra(a_torch, b_torch, (4,))
   print("result of b_torch after compiled_add_one_with_extra(a_torch, b_torch, (4,))")
   print(b_torch)

example_add_one_with_variadic_tuple()
```

#### Working with Named Tuples

Named tuples are also supported and help logically group related arguments together. The example below shows how to use named tuples as arguments. Under the hood, named tuples are passed as unnamed tuples at the ABI level. When errors occur, the function signature in error messages will display unnamed tuple arguments. Ensure that the compile-time CuTe named tuple type definition has the same fields as the runtime PyTorch named tuple. Currently, users need to explicitly unpack the named tuple outside of conditionals and then use the unpacked variables inside the conditionals.

``` python
from typing import NamedTuple
from cutlass import cute
import torch

class CuteNamedTuple(NamedTuple):
   a: cute.Tensor
   b: cute.Tensor
   c: cute.Float32 = cute.Float32(1)

   def __new_from_mlir_values__(self, values):
      return CuteNamedTuple(*values)

class TorchNamedTuple(NamedTuple):
   a: torch.Tensor
   b: torch.Tensor
   c: float = 1

@cute.kernel
def device_add_one_named_tuple(value: CuteNamedTuple):
   tid = cute.arch.block_idx()[0] * 128 + cute.arch.thread_idx()[0]
   # need to unpack namedtuple outside conditionals
   a = value.a
   b = value.b
   c = value.c
   if tid < a.shape[0]:
      b[tid] = a[tid] + c

@cute.jit
def add_one_with_named_tuple(value: CuteNamedTuple):
   n = value.a.shape[0]
   threads_per_block = 128
   blocks = (n + threads_per_block - 1) // threads_per_block
   device_add_one_named_tuple(value).launch(grid=(blocks, 1, 1), block=(threads_per_block, 1, 1))

def example_add_one_with_named_tuple():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))

   compiled_add_one = cute.compile(
      add_one_with_named_tuple, CuteNamedTuple(a=a_cute, b=b_cute),
      options="--enable-tvm-ffi"
   )
   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")
   compiled_add_one(TorchNamedTuple(a=a_torch, b=b_torch))
   print("result of b_torch")
   print(b_torch)

example_add_one_with_named_tuple()
```

### Supported types

The TVM FFI function supports the following CuTe DSL-specific types as arguments:

- `cute.Tensor`
- `cutlass.Boolean`, `cutlass.Int8`, `cutlass.Int16`, `cutlass.Int32`, `cutlass.Int64`, `cutlass.Uint8`, `cutlass.Uint16`, `cutlass.Uint32`, `cutlass.Uint64`, `cutlass.Float32`, `cutlass.Float64`
- `cute.Shape`, `cute.Stride`, `cute.Coord`, `cute.Tile`, `cute.IntTuple`

| Compile-time type | Call-time type |
|----|----|
| `cute.Pointer` | `ctypes.c_void_p` or a class that implements `__tvm_ffi_opaque_ptr__` protocol. |
| `cute.runtime.FakeTensor` | `torch.Tensor` and other DLPack-compatible tensors. |
| Scalar types (e.g. `cutlass.Boolean`, `cutlass.Int32`) | Python scalars (e.g. True, 123). |
| CuTe algebra types (e.g. `cute.Shape`, `cute.Stride`) | `tvm_ffi.Shape` or python tuple of ints. |
| CUDA stream `cuda.CUstream` | A stream class that implements the CUDA stream protocol (e.g. `torch.cuda.Stream`, `cuda.CUstream`). |
| Tuple of types (e.g. `Tuple[cute.Tensor, cute.Tensor, cutlass.Int32]`) | Python tuple of corresponding call-time types. |

### Error handling

TVM FFI functions will enable validation of arguments to make sure they match the expected type and value constraints declared by the user. These checks are compiled into the function, run very fast, and have no observable overhead during function invocation. Each of those errors will translate into a proper Python exception that can be caught and handled. The example below shows some example error cases that can be checked:

``` python
def example_constraint_checks():
   n = cute.sym_int(divisibility=16)
   # assume align to 16 bytes (4 int32), both should share same shape variable n
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,), assumed_align=16)
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,), assumed_align=16)
   compiled_add_one = cute.compile(add_one, a_cute, b_cute, options="--enable-tvm-ffi")
   a = torch.zeros(128, dtype=torch.float32, device="cuda")
   b = torch.zeros(128, dtype=torch.float32, device="cuda")

   try:
      # raises type mismatch error because we expect a and b to be float32
      compiled_add_one(a, 1)
   except TypeError as e:
      # Mismatched type on argument #1 when calling:
      # `add_one(a: Tensor([n0], float32), b: Tensor([n0], float32))`,
      # expected Tensor
      print(f"TypeError: {e}")

   try:
      # raises shape mismatch error because we expect both a and b have shap [n]
      compiled_add_one(a, b[:126])
   except ValueError as e:
      # Mismatched b.shape[0] on argument #1 when calling:
      # `add_one(a: Tensor([n0], float32), b: Tensor([n0], float32))`,
      # expected to match a.shape[0]
      print(f"ValueError: {e}")

   try:
      # triggers divisibility mismatch error because 126 is not divisible by 16
      compiled_add_one(a[:126], b[:126])
   except ValueError as e:
      # Invalid a.shape[0] on argument #0 when calling:
      # `add_one(a: Tensor([n0], float32), b: Tensor([n0], float32)`,
      # expected to be divisible by 16
      print(f"ValueError: {e}")

   try:
      a = torch.zeros(129, dtype=torch.float32, device="cuda")
      b = torch.zeros(129, dtype=torch.float32, device="cuda")
      # triggers data alignment mismatch error because x and y are not aligned to 16 bytes
      compiled_add_one(a[1:], b[1:])
   except ValueError as e:
      # raises: Misaligned Tensor data on argument #0 when calling:
      # `add_one(a: Tensor([n0], float32), b: Tensor([n0], float32)`,
      # expected data alignment=16 bytes
      print(f"ValueError: {e}")
```

Any CUDA errors encountered will also be automatically converted into Python exceptions by the TVM FFI function.

``` python
@cute.jit
def add_one_invalid_launch(a: cute.Tensor, b: cute.Tensor):
   # Intentionally exceed the maximum block dimension (1024 threads) so the
   # CUDA runtime reports an invalid configuration error.
   device_add_one(a, b).launch(grid=(1, 1, 1), block=(4096, 1, 1))

def example_error_cuda_error():
   a_torch = torch.zeros((10,), dtype=torch.float32, device="cuda")
   b_torch = torch.zeros((10,), dtype=torch.float32, device="cuda")

   a_cute = cute.runtime.from_dlpack(a_torch, enable_tvm_ffi=True)
   b_cute = cute.runtime.from_dlpack(b_torch, enable_tvm_ffi=True)
   compiled_add_one_invalid_launch = cute.compile(
      add_one_invalid_launch, a_cute, b_cute, options="--enable-tvm-ffi"
   )

   try:
      compiled_add_one_invalid_launch(a_torch, b_torch)
   except RuntimeError as e:
      # raises RuntimeError: CUDA Error: cudaErrorInvalidValue
      print(f"RuntimeError: {e}")
```

### Working with Devices

TVM FFI-compiled functions naturally work across GPU devices. The device index of the first input GPU tensor determines the kernel's device context. The TVM FFI function calls `cudaSetDevice` to set the correct device before launching the kernel based on that tensor's device index. For advanced scenarios that pass raw pointers instead of tensors, you should call `cudaSetDevice` explicitly through the CUDA Python API.

### Exporting Compiled Module

The TVM FFI function supports exporting the compiled module to an object file for further use. For example:

``` python
import subprocess
import cutlass.cute as cute

def example_add_one_export():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   # compile the kernel with "--enable-tvm-ffi" option and example input tensors
   compiled_add_one = cute.compile(add_one, a_cute, b_cute, options="--enable-tvm-ffi")
   # export the compiled module to object file
   compiled_add_one.export_to_c("./add_one.o", function_name="add_one")
   # obtain necessary runtime libs for loading the shared library
   runtime_libs = cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)
   # compile the object file to a shared library
   cmd = ["gcc", "-shared", "-o", "./add_one.so", "./add_one.o", *runtime_libs]
   print(cmd)
   subprocess.run(cmd, check=True)
   print(f"Successfully created shared library: ./add_one.so")
```

Then you can load back the exported module and use it in different ways:

``` python
import torch
from cutlass import cute

def example_load_module_add_one():
   mod = cute.runtime.load_module("./add_one.so", enable_tvm_ffi=True)
   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")
   mod.add_one(a_torch, b_torch)
   print("result of b_torch after mod.add_one(a_torch, b_torch)")
   print(b_torch)
```

The exported object file exposes the function symbol `__tvm_ffi_add_one` that is compatible with TVM FFI and can be used in various frameworks and programming languages. You can either build a shared library and load it back, or link the object file directly into your application and invoke the function via the `InvokeExternC` mechanism in TVM FFI. For more information, see the [quick start guide](https://tvm.apache.org/ffi/get_started/quickstart) in the official documentation.

When you build your own libraries, make sure you link against the necessary runtime libraries. You can use `cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)` to get the path to these libraries. `cute.runtime.load_module(path, enable_tvm_ffi=True)` will load these libraries automatically before loading an exported module. You can also manually load these libraries in advanced use cases.

For low-level cute ABI AOT compilation support without TVM FFI, you can refer to Ahead-of-Time (AOT) Compilation.

#### Keyword Arguments and Defaults

The function returned by `cute.compile` supports keyword arguments and defaults. The example below shows how to use keyword arguments and defaults:

``` python
import torch
from cutlass import cute

@cute.kernel
def device_add_scalar(a: cute.Tensor, b: cute.Tensor, offset: cutlass.Float32):
   threads_per_block = 128
   cta_x_, _, _ = cute.arch.block_idx()
   tid_x, _, _ = cute.arch.thread_idx()
   tid = cta_x_ * threads_per_block + tid_x
   if tid < a.shape[0]:
      b[tid] = a[tid] + offset

@cute.jit
def add_constant(a: cute.Tensor, b: cute.Tensor, offset: cutlass.Float32=cutlass.Float32(1)):
   n = a.shape[0]
   threads_per_block = 128
   blocks = (n + threads_per_block - 1) // threads_per_block
   device_add_scalar(a, b, offset).launch(grid=(blocks, 1, 1), block=(threads_per_block, 1, 1))

def example_kwargs_and_defaults():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   compiled_add_constant = cute.compile(add_constant, a_cute, b_cute, options="--enable-tvm-ffi")
   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")
   compiled_add_constant(a_torch, b_torch)
   print("result of b_torch after compiled_add_constant(a_torch, b_torch)")
   print(b_torch)
   compiled_add_constant(a_torch, b_torch, offset=4)
   print("result of b_torch after compiled_add_constant(a_torch, b_torch, offset=4)")
   print(b_torch)
```

For efficiency and portability reasons, TVM FFI ABI supports functions with positional-only arguments. If you export the compiled module to an object file and then load it back, the function will only accept positional arguments in the order of the arguments in the function signature. You can rewrap the function or use the TVM FFI wrapper generator to generate a kwargs wrapper. The code block below shows how to do this:

``` python
def example_kwargs_and_defaults():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   compiled_add_constant = cute.compile(add_constant, a_cute, b_cute, options="--enable-tvm-ffi")
   # export the compiled module to object file
   compiled_add_constant.export_to_c("./add_constant.o", function_name="add_constant")
   # obtain necessary runtime libs for loading the shared library
   runtime_libs = cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)
   # compile the object file to a shared library
   cmd = ["gcc", "-shared", "-o", "./add_constant.so", "./add_constant.o", *runtime_libs]
   subprocess.run(cmd, check=True)

   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")

   mod = cute.runtime.load_module("./add_constant.so")
   try:
      mod.add_constant(a_torch, b_torch)
   except Exception as e:
      # Raises a missing arguments error because kwargs and default information are lost
      print(e)
   # We rewrap the function to regain argument and kwargs support.
   # Alternatively, use the TVM FFI wrapper generator to generate a kwargs wrapper function.
   from tvm_ffi.utils import kwargs_wrapper
   # arg_defaults are aligned to the end of the argument list
   wrapped_func = kwargs_wrapper.make_kwargs_wrapper(
      mod.add_constant, arg_names=["a", "b", "offset"], arg_defaults=(1,)
   )
   wrapped_func(a_torch, b_torch)
   print("result of b_torch after wrapped_func(a_torch, b_torch)")
   print(b_torch)
   # You can also use the signature of the original function
   # to generate a kwargs wrapper function. Make sure to exclude
   # arguments that are not included in the runtime,
   # such as 'self', constexpr, and env stream arguments.
   wrapped_func = kwargs_wrapper.make_kwargs_wrapper_from_signature(
      mod.add_constant, signature=inspect.signature(add_constant),
      exclude_arg_names=["self"]
   )
   wrapped_func(a_torch, b_torch, offset=4)
   print("result of b_torch after wrapped_func(a_torch, b_torch, offset=4)")
   print(b_torch)
```

### Limitations

The Fake Tensor flow is ONLY compatible with TVM FFI because TVM FFI supports more flexible constraints on Tensor arguments. For instance, fake tensor can specify per-mode static shape or constraints on shape and strides which are not supported by existing `from_dlpack` flow. It's expected that JIT function compiled with fake tensor will have different ABI compared to tensor converted by `from_dlpack`.

``` python
import cutlass.cute as cute
import torch

n = cute.sym_int()
# Dynamic Shape
fake_a = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))

# Compile without tvm-ffi
compiled_fn = cute.compile(foo, fake_a)

# Wrong, in compatible ABI
compiled_fn(from_dlpack(a))
```

In order to avoid such issue, it's recommended to use fake tensor only with TVM FFI backend. Practically speaking, as we only want to call `from_dlpack` once and reuse for both compilation and runtime, the benefit of using fake tensor is limited in this case.

---

<!-- source: cute_dsl_general/dsl_ahead_of_time_compilation.rst -->

## Ahead-of-Time (AOT) Compilation

This guide demonstrates how to use CuTe DSL's Ahead-of-Time (AOT) compilation features to export compiled kernels for use in production environments.

### Overview

CuTe DSL Ahead-of-Time (hereinafter referred to as AOT) compilation allows you to:

- **Compile once, enable cross-compilation**: Write kernels in Python and cross-compile them for multiple GPU architectures.
- **Remove JIT overhead**: Eliminate compilation delays in production by pre-compiling kernels.
- **Flexible integration**: Easily integrate compiled kernels into both Python and C/C++ codebases using flexible deployment options.

We provide 2 levels of AOT ABI:

1.  **Low-Level CuTe ABI**: This ABI is expressed using CuTe DSL types and tensors, mirroring the original Python function.
2.  **High-Level Apache TVM FFI ABI**: For interop with various frameworks (e.g., PyTorch, JAX), and offer high-level stable ABI access.

This guide will focus on the CuTe ABI AOT. For the Apache TVM FFI AOT, please refer to the section "Exporting Compiled Module" in Compile with TVM FFI.

### CuTe ABI AOT Workflow

#### Export Interface

The `export_to_c` interface is provided by the `JitCompiledFunction` class. It accepts the following parameters:

- `file_path`: The path to the directory where the header and object files will be saved.
- `file_name`: The base name for the header and object files. The same file name will always overwrite existing files.
- `function_prefix`: The prefix of the function symbol in the generated object file. This should be a unique identifier to avoid symbol conflicts. Users should ensure the function prefix is unique for each exported function. Defaults to the `file_name`.

It generates the following files:

- `{file_path}/{file_name}.h`: A C header file containing API function declarations. This header specifies the runtime function signatures in C, mirroring the original Python function interfaces.
- `{file_path}/{file_name}.o`: A standard object file containing the compiled kernel code. You can link this object file into either a static or shared library. It includes the host entry function, fatbin data, and helper functions such as `cuda_init` and `cuda_load_to_device`. Additionally, it embeds metadata for runtime loading and version verification.

Example:

``` python
import cutlass.cute as cute
import cutlass.cute.cuda as cuda

@cute.kernel
def print_tensor_kernel(a: cute.Tensor):
    cute.printf("a: {}", a)

@cute.jit
def print_tensor(a: cute.Tensor, stream: cuda.CUstream):
    print_tensor_kernel(a).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)

compiled_func = cute.compile(print_tensor)
# Export compiled functions to object files and headers
compiled_func.export_to_c(file_path="./artifacts", file_name="print_tensor_example", function_prefix="print_tensor")
```

#### Loading in Python

Load pre-compiled object files or shared libraries into Python for execution.

``` python
import cutlass.cute as cute
import torch
from cutlass.cute import from_dlpack
import cutlass.cute.cuda as cuda

# Load module from object file
module = cute.runtime.load_module("./artifacts/print_tensor_example.o")
# or
module = cute.runtime.load_module("./artifacts/libprint_tensor_example.so")

# Prepare data
a = torch.arange(160, dtype=torch.float32, device="cuda").reshape(16, 10)
a_cute = from_dlpack(a).mark_layout_dynamic()
stream = cuda.CUstream(0)

# Call the function (no JIT compilation needed!)
module.print_tensor(a_cute, stream=stream)

# This will fail because 'non_existing_api' was not exported:
# module.non_existing_api()
```

#### C++ Integration with Static Linking

Integrate compiled kernels directly into your C++ executable during the build process. The generated header file supplies the necessary API for loading the module and invoking the function.

Example:

``` cpp
#include "print_tensor_example.h"
#include <cuda_runtime.h>

void run_print_tensor() {
    // Prepare tensor, the tensor declaration is in the header file
    print_tensor_Tensor_a_t tensor_a;
    tensor_a.data = nullptr; // GPU memory is set to nullptr.
    // Set dynamic shapes and strides
    tensor_a.dynamic_shapes[0] = 32;
    tensor_a.dynamic_shapes[1] = 16;
    tensor_a.dynamic_strides[0] = 16;

    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Load module before calling the kernel
    print_tensor_Kernel_Module_t module;
    print_tensor_Kernel_Module_Load(&module);

    // Call the kernel; the kernel wrapper function is defined in the header file
    cute_dsl_print_tensor_wrapper(&module, &tensor_a, stream);

    // Cleanup
    print_tensor_Kernel_Module_Unload(&module);
    cudaStreamDestroy(stream);
}
```

The `print_tensor_example.h` header file is generated by the `export_to_c` interface. It includes:

- The `print_tensor_Kernel_Module_t` type: Represents the kernel module.
- The `print_tensor_Tensor_a_t` type: A tensor-specific type that defines the ABI for a particular CuTe tensor.
- The `cute_dsl_print_tensor_wrapper` function: The user-facing entry point to invoke the kernel.

The compilation of the C++ executable requires the `libcuda_dialect_runtime.so` or `libcuda_dialect_runtime_static.a` library which is involved in `<wheel_install_path>/lib`, along with the CUDA driver and runtime libraries, to function properly.

#### C++ Integration with Dynamic Loading

Dynamically load pre-compiled object files or shared libraries at runtime. By including the `CuteDSLRuntime.h` header, you can load the module, look up exported functions, and invoke them.

``` cpp
#include "CuteDSLRuntime.h"
#include <cuda_runtime.h>

void run_print_tensor() {
    // Load module from shared library
    CuteDSLRT_Module_t *module = nullptr;
    CuteDSLRT_Error_t err = CuteDSLRT_Module_Load(
        &module,
        "./artifacts/libprint_tensor_example.so"
    );
    // or
    CuteDSLRT_Error_t err = CuteDSLRT_Module_Load(
        &module,
        "./artifacts/print_tensor_example.o"
    );
    check_error(err);

    // Lookup function
    CuteDSLRT_Function_t *func = nullptr;
    err = CuteDSLRT_Module_Get_Function(&func, module, "print_tensor");
    check_error(err);

    // Prepare arguments, matching the argument type defined in the header file
    typedef struct {
        void *data;
        int32_t dynamic_shapes[2];
        int64_t dynamic_strides[1];
    } print_tensor_Tensor_a_t;

    print_tensor_Tensor_a_t tensor_a;
    tensor_a.data = nullptr;
    tensor_a.dynamic_shapes[0] = 32;
    tensor_a.dynamic_shapes[1] = 16;
    tensor_a.dynamic_strides[0] = 16;

    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Call the function; the runtime function accepts packed arguments, refer to the wrapper in the header file
    int ret;
    void* args[] = {&tensor_a, &stream, &ret};
    err = CuteDSLRT_Function_Run(func, args, 3);
    check_error(err);
    cudaStreamSynchronize(stream);

    // Cleanup
    CuteDSLRT_Module_Destroy(module);
    cudaStreamDestroy(stream);
}
```

The `CuteDSLRuntime.h` header file can be found in `<wheel_install_path>/include`. It includes:

- The `CuteDSLRT_Error_t` type: Indicates error status.
- The `CuteDSLRT_Module_Load` function: Loads the module.
- The `CuteDSLRT_Module_Get_Function` function: Gets a function from the loaded module. The runtime API will load the CUDA module for kernel execution.
- The `CuteDSLRT_Function_Run` function: Runs the function.
- The `CuteDSLRT_Module_Destroy` function: Destroys the module.

The compilation of the C++ executable requires the `libcute_dsl_runtime.so` library which is involved in `<wheel_install_path>/lib`, along with the CUDA driver and runtime libraries, to function properly.

### Supported Argument Types

CuTe DSL supports the following argument types:

- `cute.Tensor`
- `cute.Shape` / `cute.Coord` / `cute.Tile` / `cute.IntTuple` / `cute.Stride`
- `cuda.CUstream`
- `cutlass.Int8` / `cutlass.Int16` / `cutlass.Int32` / `cutlass.Int64` / `cutlass.Boolean`
- `cutlass.Uint8` / `cutlass.Uint16` / `cutlass.Uint32` / `cutlass.Uint64`
- `cutlass.Float32` / `cutlass.TFloat32` / `cutlass.Float64` / `cutlass.Float16`

Note that:

1.  `cute.Tensor` is a dynamic tensor type that only contains dynamic shapes and strides in its ABI representation. As a result, different compilations may produce different tensor ABIs. This is why declarations for each tensor type are included in the generated header file.
2.  `strides` in `cute.Tensor` are determined by the `use_32bit_strides` compile argument. When `use_32bit_strides` is set to `True`, the strides are 32-bit; when set to `False`, they are 64-bit.
3.  Currently, custom types are not supported for AOT compilation.

### Object File Compatibility Issues

The object file generated by CuTe DSL depends on the CUDA runtime library. Therefore, ensure that the version of the CUDA runtime/toolkit library matches the version used by CuTe DSL. Otherwise, ABI compatibility with the CUDA runtime cannot be guaranteed.

When using C++ static linking integration, compatibility is assured because the header and object files are generated together and guaranteed to match.

For C++ dynamic loading integration and Python loading, the binary file is loaded at runtime. To ensure compatibility, version information is embedded in the metadata of the generated binary file. At runtime, this version information is checked, and if it does not match the expected version, the binary file will be rejected.

### Relation to Apache TVM FFI AOT

Apache TVM FFI AOT offers a comparable capability, enabling TVM functions to be compiled into binary files that can be loaded and executed at runtime. For more information, see the section "Exporting Compiled Module" in Compile with TVM FFI.

The primary distinction is that, when TVM FFI is enabled, CuTe DSL generates a dedicated wrapper function on top of the underlying CuTe ABI. This wrapper adheres to the calling conventions defined by TVM FFI. In contrast, the CuTe ABI entry function is specified directly in the generated header file, which affects how arguments must be provided.

For instance, with the TVM FFI wrapper function, users are able to pass in arguments such as `torch.Tensor` directly. However, when calling the CuTe ABI entry function, arguments should be provided as `cute.Tensor` types.

---

<!-- source: cute_dsl_general/resources.rst -->

## Talks and Presentations

This page collects talks, presentations, and other resources related to CuTe DSL and CUTLASS Python infrastructure.

### Conference Talks

**CuTeDSL: CUTLASS Python DSL Infrastructure** — *LLVM 2025*

An introduction to the CuTe DSL architecture, covering the hybrid AST-rewrite and tracing approach, MLIR code generation, and integration with CUTLASS.

- [Video](https://www.youtube.com/watch?v=5NXd6MbKYNQ)
- [Slides (PDF)](https://llvm.org/devmtg/2025-10/slides/technical_talks/ozen.pdf)

------------------------------------------------------------------------

**Enable Tensor Core Programming in Python with CUTLASS 4.0** — *GTC 2025*

Learn how to leverage Tensor Cores directly from Python using CUTLASS 4.0's new DSL front-end, enabling rapid kernel development without writing CUDA C++.

- [Video](https://www.nvidia.com/en-us/on-demand/session/gtc25-s74639/)

---

<!-- source: cute_dsl_general/naming_conventions.rst -->

## CuTe DSL Naming Conventions

This page summarizes the Hungarian-style naming conventions used for identifiers across the DSL examples and epilogue helpers: tensor partitions, per-thread copy-partitioners, copy atoms, and the axis-order suffixes that encode tensor layouts. It is meant as a lookup reference while reading example code — not as a style rule enforced on new code.

### Memory/space scopes

- `g`: Global memory view (GMEM), e.g., `gB_nkl`, `tTR_gC`
- `s`: Shared memory view (SMEM), e.g., `sA`, `tRS_sC`, `bSG_sC`
- `r`: Register view (RMEM), e.g., `tTR_rAcc`, `tRS_rC`
- `t`: Tensor-memory view (TMEM), used for any TMEM-resident fragment or layout regardless of role. The classical case is the accumulator (`tCtAcc`, `tTR_tAcc`). The same scope letter also appears for non-accumulator TMEM tensors such as `tCtE`, `tCtState`, `tCtQState`, `tCtShared`. Read the operand suffix to distinguish the role from the memory scope.

### Per-thread/partitioned views and families

- `tA…` / `tB…`: TMA load path for A/B

  - `tAgA` / `tAsA`: per-thread partitioned global/shared A for TMA load
  - `tBgB` / `tBsB`: per-thread partitioned global/shared B for TMA load
  - NVFP4/FP8 scale factors mirror this: `tAgSFA` / `tAsSFA`, `tBgSFB` / `tBsSFB`

- `tC…`: Compute/epilogue path for C/Acc

  - `tCgA` / `tCgB` / `tCgC`: per-thread partitions used by MMA/epilogue (derived from global tensors)
  - `tCrA` / `tCrB`: per-thread fragments used by MMA (derived from SMEM A/B)
  - `tCtAcc`: per-thread accumulator fragment/layout in TMEM
  - Additional `tC*` tensors follow the same schema for kernels that carry more than the classical A/B/C/Acc operands (see Operands and roles below): e.g. `tCtState` / `tCtQState` / `tCtShared` (gated-delta-net recurrent state in TMEM), `tCrValpha` / `tCrVbeta` / `tCrVbias` (EVT/EFC broadcast vectors in registers), `tCtAccInter` / `tCtAccIntra` (hierarchical accumulators)

  <!-- -->

  - Sparse GEMM additionally defines `tCtE` for the sparsity metadata tensor in TMEM (sm_140 / Feynman sparse GEMM, not yet released)

- `tTM…`: Per-thread TMEM tiled-copy partitions used by FMHA/attention kernels (e.g. `tTMrO` as the register-side view of a TMEM load partitioned through `thr_tmem_load`)

- Attention/MLA path families (`tQ…`, `tK…`, `tV…`, `tP…`, `tO…`): same schema as `tA…` / `tB…` / `tC…` but specialised to the Q/K/V/P/O operands of attention kernels, e.g.:

  - `tQsQ` / `tQgQ_qdl`: per-thread SMEM / GMEM partitions of Q for TMA load
  - `tKrK` / `tVrV`: per-thread register fragments for K / V
  - `tOtO` / `tOrO`: per-thread TMEM / register views of the attention output accumulator O
  - `tPrP`: per-thread register fragment for the softmax probability matrix P

### Data-movement copy paths

- `tTR_*`: TMEM → Register (T2R)

  - `tTR_tAcc`: TMEM accumulator source for T2R
  - `tTR_rAcc`: Register destination for T2R
  - `tTR_gC`: When not using TMA store, Register → Global C destination partition

- `tRS_*`: Register → Shared (R2S)

  - `tRS_rC`: Register source (C dtype)
  - `tRS_sC`: Shared destination

- `bSG_*`: Thread(b)lock partition for Shared → Global via TMA store

  - `bSG_sC`: Shared source for TMA store
  - `bSG_gC`: Global destination for TMA store
  - Also used for accumulator in some flows: `bSG_sAcc`, `bSG_gAcc`
  - The same schema extends to additional store operands: `bSG_sD` / `bSG_gD`, `bSG_sP` / `bSG_gP`, `bSG_sY` / `bSG_gY`

- `bGS_*`: Thread(b)lock partition for Global → Shared via TMA **load** (the load-path mirror of `bSG_*`)

  - `bGS_gC` / `bGS_sC`: Global source / Shared destination for TMA load of C-like operands (seen in EFC row/column broadcast prologues)

- `simt_atom`: SIMT copy path used when TMA store is disabled (Register → Global)

- Generic SIMT / tiled copy atoms `<src>2<dst>_atom[_suffix]` name the copy direction between two memory scopes:

  - `s2r_atom_*`: Shared → Register atom used in specialised epilogues and attention loads (e.g. `s2r_atom_delta`, `s2r_atom_cumsum`, `s2r_atom_d` in Mamba2 SSD)
  - `r2s_atom`: Register → Shared atom
  - `t2r_atom` / `r2t_atom`: Tensor memory ↔ Register atoms (paired with `thr_tmem_load` / `thr_tmem_store`)
  - `s2s_atom`: Shared → Shared atom (reshape/remap without register spill)
  - `s2t`: Shared → Tensor memory atom

  <!-- -->

  - `sp2t_copy_op_*`: Sparse source → Tensor memory copy op (sm_140 / Feynman sparse GEMM, not yet released: e.g. `Sp2TAsACopyOp`, `Sp2TAsECopyOp`)

  <!-- -->

  - Custom `autovec_copy` paths appear where the DSL auto-vectorises a bespoke layout

### Operands and roles

- `A`, `B`, `C`: GEMM operands
- `Acc`: Accumulator (TMEM/Register paths). Hierarchical MMA kernels split this into `AccInter` / `AccIntra` for the inter-/intra-CTA accumulator halves
- Classical extra outputs / intermediates: `D` (additional output), `Y` (fused output), `SFA` / `SFB` (per-operand scale-factor arrays for NVFP4/FP8), `SF` (generic scale factor)
- Attention / MLA operand letters (Q/K/V/P/O schema):
  - `Q` (query), `K` (key), `V` (value), `P` (softmax probability / score matrix), `O` (attention output)
  - Variants: `Kt` / `Vt` for the transposed view of K/V, `Qi` / `Ki` / `Vi` for per-iteration slices, `QK` / `PV` / `QKV` where a single fragment spans multiple operands of the two back-to-back matmuls
- Mamba / recurrent-state letters: `Delta` / `DeltaA` (time-step and A-decay), `State` / `QState` / `Shared` (gated-delta-net recurrent state tensors), `Cumsumlog` / `Cumprod` (running reductions), `Gate`, `DecayV`

<!-- -->

- Sparse-GEMM letters (sm_140 / Feynman, not yet released): `E` (sparsity metadata tensor in TMEM; paired with `sp2t_*` copy ops)

<!-- -->

- EVT / EFC broadcast vectors: `Valpha` / `Vbeta` (alpha/beta scalars broadcast as vectors), `Vbias` (bias vector), `Ainv` (inverse of A for fused solvers)

<!-- -->

- LUT-based block-scaled GEMM letter (Rubin, not yet released): `LutB` (look-up-table operand)

<!-- -->

- Communication operands (multi-CTA / multicast flows): `CommInMC` / `CommOutMC` (multicast in/out), `CommOutUC` (unicast out)
- Head-dimension variants: `Dv` (value head dimension when distinct from Q/K dim), `Nv` (number of value heads)

### Axis-order suffixes

- Suffix encodes axis order of the view (lowercase letters each stand for one tensor mode):
  - GEMM layouts use `m`/`n`/`k`/`l`:
    - `_mnl`, `_nkl`, `_mkl`, … map to (M, N, K, L) ordering
    - Example: `gB_nkl` is B with axes (N, K, L); `gC_mnl` is C with (M, N, L)
  - Attention / FMHA layouts use `q`/`k`/`d`/`l` (sequence-Q, sequence-K, head-dim, batch):
    - `mQ_qdl`: Q tensor with axes (SeqQ, HeadDim, Batch)
    - `mK_kdl`: K tensor with axes (SeqK, HeadDim, Batch)
    - `mV_dkl`: V tensor with axes (HeadDim, SeqK, Batch) — the `d`-first order reflects the V-transpose that makes the second matmul (P·V) a standard row-major `MxK·KxN`
  - Lower-rank 2D slices drop the batch letter: `_mn`, `_mk`, `_nk`
- Internally, CuTe layouts also expose grouped modes like `MMA_M/N/K`, `EPI_M/N`, `RestM/N/K/L`, `STAGE`, etc. (these are typically implementation details not directly used in example code).

### Reading compound tokens

- From left to right: `[t|b][A|B|C|Q|K|V|P|O|TR|RS|SG|GS|TM]_[g|s|r|t][Operand/Role][AxisSuffix?]`

  - `t` = per-thread/partitioned view; `b` = block/threadblock partition context
  - family/path letters:
    - Operand-based: `A` / `B` / `C` (GEMM), `Q` / `K` / `V` / `P` / `O` (attention)
    - Direction-based: `TR` (TMEM → Register), `RS` (Register → Shared), `SG` (Shared → Global, store), `GS` (Global → Shared, load), `TM` (TMEM tiled-copy partition), `R2G` / `S2R` / `T2R` / `R2T` convenience aliases
  - memory = `g`/`s`/`r`/`t`
  - operand/role = `A`/`B`/`C`/`Acc`/`SFA`/`SFB`/`Q`/`K`/`V`/`P`/`O`/`E`/`State`/…
  - axis suffix = `_mnl`, `_nkl`, `_qdl`, `_kdl`, `_dkl`, `_mn`, … when applicable

- Per-thread-partitioner objects follow a parallel `thr_*` vocabulary, grouped by role:

  - MMA partitioner: `thr_mma`
  - Tiled-copy direction variants `thr_copy_<src>2<dst>`: `thr_copy_g2s`, `thr_copy_s2r`, `thr_copy_t2r`, `thr_copy_r2s`, `thr_copy_r2t`, `thr_copy_s2t`
  - Role-qualified copy variants: `thr_copy_sfa`, `thr_copy_sfb`, `thr_copy_load`, `thr_copy_beta_g2s`
  - MMA variants for multi-matmul kernels: `thr_mma_qk`, `thr_mma_pv`, `thr_mma_kv`, `thr_mma_qkv`, `thr_mma_intra1` / `thr_mma_intra2`, `thr_mma_leader_cta`, `thr_mma_sfb`
  - TMEM access partitioners: `thr_tmem_load`, `thr_tmem_store` (with `_stats` / `_vec` suffix variants)

  The tensor produced by `thr_foo.partition_S(X)` or `.partition_D(X)` is then named by the `[t|b]FamilyPrefix_*` convention above.

### Concrete references

Open these files in the repository to see each pattern in context:

- TMA load partitions for A/B:
  - `tAgA`, `tAsA`, `tBgB`, `tBsB`
  - `CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py` (around TMA partition of A/B)
- Accumulator fragment in TMEM:
  - `tCtAcc`
  - `CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py` (accumulator creation and use)
- TMEM → Register (T2R):
  - `tTR_tAcc`, `tTR_rAcc`, `tTR_gC`
  - `CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py` (`epilog_tmem_copy_and_partition`)
- Register → Shared (R2S):
  - `tRS_rC`, `tRS_sC`
  - `CuTeDSL/cute/blackwell/kernel/mixed_input_gemm/mixed_input_gemm.py` (`epilog_smem_copy_and_partition`)
- Shared → Global via TMA store:
  - `bSG_sC`, `bSG_gC`
  - `CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent.py` (`epilog_gmem_copy_and_partition`)
- NVFP4/FP8 scale factors:
  - `tAgSFA`/`tAsSFA`, `tBgSFB`/`tBsSFB`
  - `CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/nvfp4_gemm_0.py` (scale factor partition and usage)
- Additional examples across `examples/`:
  - Register → Global helper naming in MLA: `tR2G_rO_src`, `tR2G_rO_dst`
  - `CuTeDSL/cute/blackwell/kernel/attention/mla/mla_decode_fp16.py` (output store section)
  - Shared → Register SIMT atoms in Mamba2 SSD: `s2r_atom_delta`, `s2r_atom_cumsum`, `s2r_atom_d`
  - `CuTeDSL/cute/blackwell/kernel/attention/mamba2_ssd/mamba2_ssd.py` (SMEM load paths for delta and D)
  - `thr_*` slices for partitioning per-thread work: `thr_mma`, `thr_copy_t2r`, `thr_copy_r2s`, etc.
  - `CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py` (`thr_mma`, `thr_copy_t2r`, `thr_copy_r2s`)
- Axis-order suffix examples:
  - `gB_nkl`, `gC_mnl`
  - `CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py` (global tensor tiling and partitioning)
- Global → Shared (TMA load) block partition `bGS_*`:
  - `bGS_gC`, `bGS_sC`
  - `CuTeDSL/cute/blackwell/efc/common_efc.py` (row/column broadcast prologue building the C-like input for EVT)
- Attention Q/K/V/P/O families and `_qdl` / `_kdl` / `_dkl` axis suffixes:
  - `tQsQ`, `tQgQ_qdl`, `mK_kdl`, `mV_dkl`
  - `CuTeDSL/cute/hopper/kernel/attention/fmha.py` (Q/K/V TMA partitions)
  - `tOtO`, `tOrO`, `tPrP`
  - `CuTeDSL/cute/blackwell/tutorial/tutorial_fmha/fmha_0.py` (output and softmax fragments)
  - `tKrK`, `tVrV`
  - `CuTeDSL/cute/blackwell/kernel/attention/mixed_input_fmha/mixed_input_fmha_decode.py` (mixed-input K/V register fragments)
- TMEM tiled-copy `tTM*` family and the generalised `<src>2<dst>_atom` naming:
  - `tTMrO` driven by `thr_tmem_load`
  - `CuTeDSL/cute/blackwell/tutorial/tutorial_fmha/fmha_0.py`
- Recurrent-state operands (`State` / `QState` / `Shared`) in TMEM:
  - `tCtState`, `tCtQState`, `tCtShared`
  - `CuTeDSL/cute/blackwell/kernel/attention/gated_delta_net/gated_delta_net_chunked.py`

<!-- -->

- Sparse-metadata operand `E` and `sp2t_*` copy ops (sm_140 / Feynman, not yet released):
  - `tCtE`, `sp2t_copy_op_A`, `sp2t_copy_op_E`
  - `CuTeDSL/internal/feynman/sm140_sparse_gemm.py` and `sm140_sparse_gemm_temporal_split_k.py`
- LUT-based block-scaled GEMM operand `LutB` (Rubin, not yet released):
  - `CuTeDSL/cute/rubin/kernel/blockscaled_gemm/dense_blockscaled_gemm_lut.py`
  - `CuTeDSL/cute_ext/rubin/dense_gemm_lutb.py`

<!-- -->

- Richer `thr_*` and `thr_copy_*` / `thr_mma_*` / `thr_tmem_*` partitioner taxonomy:
  - `thr_copy_g2s`, `thr_copy_s2r`, `thr_copy_s2t`, `thr_copy_r2t`, `thr_mma_qk`, `thr_mma_pv`, `thr_tmem_load`, `thr_tmem_store`
  - The attention and Mamba2 examples above are the densest references; any `fmha_*.py` or `mamba2_ssd.py` file will show the full vocabulary in use

---

<!-- source: mma_docs/wmma_programming.rst -->

## Warp-Level MMA Instructions Programming Guide

Ampere (SM80) introduced the modern **warp-level MMA** PTX instruction family `mma.sync.aligned`. A warp (32 threads) cooperates on one synchronous `D = A * B + C` matrix multiply-accumulate; later architectures extended the family with new data types and shapes — FP8 on Ada (SM89) and block-scaled MX FP4 on Blackwell (SM120a) — while keeping the same warp-synchronous issue model.

Key architectural characteristics:

- **Warp scope:** One MMA is issued collectively by a 32-thread warp rather than by a warpgroup or a single thread.
- **Synchronous issue model:** `mma.sync.aligned` completes in program order within the warp; no fences or commit/wait groups are required.
- **Register-resident operands and accumulator:** A, B, and C/D all live in the register file (RMEM). Each thread holds a small fragment of every operand in its own registers.
- **SMEM → RMEM loading:** Operands A and B are staged in shared memory and loaded into register fragments via `ldmatrix` — a warp-collective SMEM→RMEM load that distributes tiles in the exact layout the MMA expects — or via regular shared-memory loads.
- **Fixed operand layout:** A is row-major (K-major) and B is col-major (K-major); transpose is not supported at the instruction level.

The dense DSL op classes currently exposed are `MmaF16BF16Op` (F16/BF16, SM80+), `MmaFP8Op` (FP8 E4M3/E5M2, SM89+), and `MmaMXF4Op` / `MmaMXF4NVF4Op` (block-scaled MX FP4, SM120a+); see [Setting up the TiledMMA, MMA Ops](#setting-up-the-tiledmma-mma-ops) for their full constructor parameters, instruction shapes, and architecture requirements.

Internal builds additionally expose `MmaF16BF16SparseOp` (2:4 structured sparsity, SM80+).

This guide outlines the CuTe Python DSL programming model for warp-level MMA kernels: stage operands in SMEM, load register fragments with `ldmatrix` or regular shared-memory loads, launch warp-synchronous MMAs, and stage the RMEM accumulator back to GMEM in the epilogue.

### Global Memory (GMEM) to MMA data flow overview

Warp MMA (`mma.sync.aligned`) instructions require all operands --A, B, and the accumulator C/D-- to live in registers (RMEM) of the 32 threads of the warp. Operand data must therefore be explicitly loaded into registers before each MMA instruction. The most common way to implement these GEMMs is to stage A and B from GMEM into SMEM with `cp.async`, then use `ldmatrix` (an SMEM→RMEM warp-collective load) to fill the A/B register fragments just before `cute.gemm()`.

The diagram below traces the full data flow of a warp MMA GEMM kernel, for the most common case where A and B matrices are stored in GMEM and staged through SMEM via `cp.async`, and the output matrix --accumulated in RMEM-- is written back to GMEM through an SMEM staging buffer for coalesced vectorized stores.

There are 3 parallel tracks where each has 2 sub-tracks. Three parallel tracks are for operands A, B, and C/D, respectively. The two sub-tracks are for copying data between different memory spaces and for MMA execution.

- **Operand A** (and symmetrically **Operand B**):
  - First, we need to create SMEM tensors for A and B matrices: `sA` and `sB`. These tensors are physically allocated tensors that are the staging destination of `cp.async` and the source of `ldmatrix` for the warp MMA instructions.
  - Next the **data copy flow** creates the tensor views for copying data from GMEM to SMEM. It starts with `mA` tensor that represents the matrix A in global memory. Then `mA` → `local_tile` → `gA` operation creates the local tile view of A that is the slice of A matrix needed to compute the given CTA's output tile. A copy partition maps this tile to per-thread copy views (`tAgA`, `tAsA`), and the multi-stage `cp.async` pipeline performs `copy(tiled_copy_A, tAgA[k], tAsA[stage])`.
  - In parallel, the **MMA flow** turns the staged SMEM tensor into register fragments consumed by the warp MMA. From the SMEM allocation `sA`, MMA partitioning produces the SMEM operand view `tCsA = partition_A(sA)` and the register-fragment layout `tCrA = make_fragment_A(tCsA)`. A dedicated S2R/`ldmatrix` path then retiles the source and destination (`partition_S` on SMEM, `retile` on RMEM) and executes `copy(s2r_A, tCsA_copy_view[k_blk], tCrA_copy_view[k_blk])` per k-block, filling the `tCrA` registers read by `cute.gemm()`.
- **Accumulator C/D**:
  - **RMEM accumulator flow** (MMA input/output): output tile views are formed by `mC` → `local_tile` → `gC` → `partition_C` → `tCgC`, then `make_fragment_C(tCgC)` creates the register accumulator `tCrC`. Warp MMA keeps C/D entirely in RMEM, and `tCrC` is both the input C and output D of `cute.gemm()`.
  - **Epilogue flow** (RMEM → SMEM → RMEM → GMEM): the epilogue converts accumulator values (for example `tCrD = epilogue_op(tCrC)`), stages them through SMEM (`autovec_copy(tCrD, tCsC)`), reloads them into registers with the epilogue copy layout, and performs coalesced vectorized GMEM stores via `copy(tiled_copy_C, tCrC_epi, tCgC_epi)`.

``` text
Operand A Dataflow Path                 Operand B Dataflow Path                 Accumulator C/D Dataflow Path
───────────────────────                 ───────────────────────                 ─────────────────────────────

mA: (M, K)           [GMEM]             mB: (N, K)            [GMEM]             ┌──── RMEM ──────────┐
│                                       │                                       │ make_fragment_C()  │
│ local_tile(mA, cta_tiler, coord)      │ local_tile(mB, cta_tiler, coord)      │ tCrC: accumulator  │
▼                                       ▼                                       └───────┬────────────┘
gA: (BM, BK, k)      [GMEM]             gB: (BN, BK, k)       [GMEM]                     │
│                                       │                                       tCrC:(MMA,MMA_M,MMA_N) [RMEM]
│  ┌──── SMEM ─────────┐                │  ┌──── SMEM ─────────┐                         │
│  │ sA: (BM,BK,PIPE)  │                │  │ sB: (BN,BK,PIPE)  │                         │        mC: (M, N)     [GMEM]
│  └──┬────────┬───────┘                │  └──┬────────┬───────┘                         │        │
│     │        │                        │     │        │                                 │        │ local_tile
│     │  thr_mma.partition_A(sA)        │     │  thr_mma.partition_B(sB)                 │        ▼
│     │        ▼                        │     │        ▼                                 │        gC: (BM, BN)   [GMEM]
│     │  tCsA:(MMA,MMA_M,               │     │  tCsB:(MMA,MMA_N,                        │        │ partition_C
│     │        MMA_K,PIPE) [SMEM]       │     │        MMA_K,PIPE) [SMEM]                │        ▼
│     │        │                        │     │        │                                 │        tCgC:(MMA,MMA_M,
│     │  make_fragment_A(tCsA)          │     │  make_fragment_B(tCsB)                   │              MMA_N)
│     │        ▼                        │     │        ▼                                 │        [GMEM] (epi dest)
│     │  tCrA:(MMA,MMA_M,               │     │  tCrB:(MMA,MMA_N,                        │        │
│     │        MMA_K) [RMEM]            │     │        MMA_K) [RMEM]                     │        │
│     │        │                        │     │        │                                 │        │
│     │  S2R retiling (ldmatrix):       │     │  S2R retiling (ldmatrix):                │        │
│     │   s2r_A = make_tiled_copy_A(    │     │   s2r_B = make_tiled_copy_B(             │        │
│     │             ldmatrix, mma)      │     │             ldmatrix, mma)               │        │
│     │   tCsA_copy_view =              │     │   tCsB_copy_view =                       │        │
│     │     s2r_A.partition_S(sA)       │     │     s2r_B.partition_S(sB)                │        │
│     │   tCrA_copy_view = retile(tCrA) │     │   tCrB_copy_view = retile(tCrB)          │        │
│     │        └─────────────┐          │     │        └─────────────┐                   │        │
╰─────┤                      │          ╰─────┤                      │                   │        │
      ▼                      │                ▼                      │                   │        │
tAgA = thr_copy_A.           │              tBgB = thr_copy_B.       │                   │        │
         partition_S(gA)     │                       partition_S(gB) │                   │        │
tAsA = thr_copy_A.           │              tBsB = thr_copy_B.       │                   │        │
         partition_D(sA)     │                       partition_D(sB) │                   │        │
      |                      │                    |                  │                   │        │
      ▼                      │                    ▼                  │                   │        │
  ┌───┴────────────────────┐ │             ┌──────┴─────────────────┐│                   │        │
  │ cp.async loop (k-tile):│ │             │ cp.async loop (k-tile):││                   │        │
  │ copy(tiled_copy_A,     │ │             │ copy(tiled_copy_B,     ││                   │        │
  │      tAgA[k],          │ │             │      tBgB[k],          ││                   │        │
┌─▶│      tAsA[stage])      │ │         ┌──▶│      tBsB[stage])      ││                   │        │
│  │ (writes into sA;       │ │         │   │ (writes into sB;       ││                   │        │
│  │  ldmatrix reads sA)    │ │         │   │  ldmatrix reads sB)    ││                   │        │
│  │ repeat for next k/stage│ │         │   │ repeat for next k/stage││                   │        │
│  └────────────────────────┘ │         │   └────────────────────────┘│                   │        │
│        │                    │         │         │                   │                   │        │
└────────┘                    ▼         └─────────┘                   ▼                   ▼        │
                             └───────┬───────────────────────────────┴───────────────────┘        │
                                     │                                                            │
                                     ▼                                                            │
                       ┌────────────────────────────────────────────────────────┐                 │
                       │ MMA loop (k_blk):                                      │                 │
                       │ S2R: copy(s2r_A, tCsA_copy_view[k_blk],                │                 │
                       │                  tCrA_copy_view[k_blk])                │                 │
                       │ S2R: copy(s2r_B, tCsB_copy_view[k_blk],                │                 │
                       │                  tCrB_copy_view[k_blk])                │                 │
                       │      [SMEM → RMEM via ldmatrix; fills tCrA/tCrB]       │                 │
                       │                                                        │                 │
                       │ cute.gemm(tiled_mma,                                   │                 │
                  ┌──▶ │  tCrC,         D (output, RMEM),                       │                 │
                  │    │  tCrA[k_blk],  A (RMEM),                               │                 │
                  │    │  tCrB[k_blk],  B (RMEM),                               │                 │
                  │    │  tCrC)         C (accumulator, RMEM)                   │                 │
                  │    └────────────────────────────────────────────────────────┘                 │
                  │       │     │                                                                 │
                  └───────┘     |                                                                 │
                                ▼                                                                 │
                          Epilogue:                                                               │
                          tCrD = epilogue_op(tCrC)       [RMEM]                                   │
                                │                                                                 │
                                ▼                                                                 │
                          sC = alloc(sC_layout) [SMEM]                                            │
                          tCsC = thr_mma.partition_C(sC)                                          │
                          R2S: autovec_copy(tCrD, tCsC)                                           │
                          [RMEM → SMEM]                                                           │
                                │                                                                 │
                                ▼                                                                 │
                          tCsC_epi = thr_copy_C.partition_S(sC)                                   │
                          tCgC_epi = thr_copy_C.partition_D(gC) ◀─────────────────────────────────┘
                          tCrC_epi = make_fragment_like(...)
                          S2R: autovec_copy(tCsC_epi, tCrC_epi)
                          [SMEM → RMEM]
                                │
                                ▼
                          Store: copy(tiled_copy_C, tCrC_epi, tCgC_epi)
                          [RMEM → GMEM]
```

**Naming convention:**

- `mma_tiler` = `(BM, BN, BK)` (CTA tiler dimensions)
- `mX` = global tensor (for example A as `(M, K)`)
- `gX` = CTA-tiled GMEM slice (for example `(BM, BK, k)` for A)
- `sX` = SMEM allocation (for example `(BM, BK, PIPE)`)
- `tAgA` / `tAsA` = `cp.async` source/destination partitions (`CPY, CPY_M, CPY_K, ...`)
- `tCsX` = MMA-partitioned SMEM view (for example `(MMA, MMA_M, MMA_K, PIPE)`)
- `tCrX` = register fragment (for example `(MMA, MMA_M, MMA_K)`)
- `tCrC` = RMEM accumulator (`MMA, MMA_M, MMA_N`)
- `tCgC` = MMA-partitioned GMEM view for output (`MMA, MMA_M, MMA_N`)
- `tCsA_copy_view` / `tCrA_copy_view` = `ldmatrix` retile views for SMEM→RMEM copy (from `partition_S(sA)` and `retile(tCrA)` on the S2R tiled copy; C++ equivalents: `tXsA` / `tXrA`)
- `MMA` = atom thread-value layout; `MMA_M/MMA_N/MMA_K` = repeat counts (for example `BM/inst_M`), `k` = outer K-tiles, `PIPE` = pipeline stages

### Setting up the TiledMMA, MMA Ops

As shown in the data flow overview, CuTe DSL provides many utilities to tile/partition the global memory tensors, and create fragment views of SMEM and register tensors for MMA instructions.

To utilize these functions, we need to setup the TiledMMA, MMA Ops first.

#### Creating a Warp MMA Op

A warp MMA op describes the hardware `mma.sync.aligned` instruction to use, it has parameters like data types and instruction shape. The operand layout is fixed (A = row-major, B = col-major).

``` python
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warp

op = warp.MmaF16BF16Op(
    cutlass.Float16,     # A/B element type
    cutlass.Float32,     # accumulator type
    (16, 8, 16),         # instruction shape (M, N, K)
)
```

The key parameters are:

- **Instruction shape** `(M, N, K)`: determines the size of one hardware MMA instruction. Valid shapes depend on the data type (see ops table below).
- **A/B element type** (`ab_dtype`) and **accumulator type** (`acc_dtype`): `Float32` is always a valid accumulator; `Float16` is only valid for F16 inputs. Each op restricts `ab_dtype` to a specific family (F16/BF16, FP8, MXF4, etc.).
- **Operand layout**: fixed to A = row-major (K-major), B = col-major (K-major). Transpose is not supported. All 32 threads in a warp cooperate on each instruction.

CuTe DSL provides implementation of many warp-level MMA ops:

| PTX name | Python class | Constructor parameters | SM Arch |
|----|----|----|----|
| `mma.sync.aligned.m16n8k{K}.row.col.{acc}.f16.f16` / `.bf16.bf16` | `warp.MmaF16BF16Op` | `ab_dtype, acc_dtype, shape_mnk` | `sm_80+` |
| `mma.sync.aligned.m16n8k{K}.row.col.{acc}.{e4m3|e5m2}.{e4m3|e5m2}` | `warp.MmaFP8Op` | `ab_dtype, acc_dtype, shape_mnk` | `sm_89+` |
| `mma.sync.aligned.kind::mxf4.block_scale.m16n8k64` | `warp.MmaMXF4Op` | `ab_dtype, acc_dtype, sf_type` | `sm_120a+` |
| `mma.sync.aligned.kind::mxf4nvf4.block_scale.m16n8k64` | `warp.MmaMXF4NVF4Op` | `ab_dtype, acc_dtype, sf_type` | `sm_120a+` |

warp-level MMA ops

Internal builds additionally provide:

| PTX name | Python class | Constructor parameters | SM Arch |
|----|----|----|----|
| `mma.sp.sync.aligned.m16n8k{K}.row.col.{acc}.f16.f16` / `.bf16.bf16` | `warp.MmaF16BF16SparseOp` | `ab_dtype, acc_dtype, shape_mnk, sparse_metadata_format` | `sm_80+` |

Internal warp-level MMA ops

#### Creating a Tiled MMA

A `TiledMma` tiles the MMA atom across the thread block so that multiple warps cooperate on a larger tile. You can pass the op directly or create an explicit atom first:

``` python
# Option 1: directly from op (common shorthand)
tiled_mma = cute.make_tiled_mma(op)

# Option 2: explicit atom creation
atom = cute.make_mma_atom(op)
tiled_mma = cute.make_tiled_mma(atom)
```

With no extra arguments this wraps a single atom — one warp, one `(16, 8, K)` tile. The optional `atom_layout_mnk` and `permutation_mnk` parameters (described in the subsections below) control multi-warp tiling and per-thread value layout respectively.

#### Spatial tiling with a repeat count

A repeat tuple `(M_rep, N_rep, K_rep)` passed as `atom_layout_mnk` replicates the warp MMA atom across the M, N, and K dimensions, producing a larger tiled MMA that is executed cooperatively by `M_rep * N_rep * K_rep` warps in a single `cute.gemm` call. Each entry in the repeat tuple corresponds to one **warp** (32 threads), so `(2, 2, 1)` uses four warps — a common configuration for warp-specialized SM80/SM89 kernels:

``` python
atom = cute.make_mma_atom(op)     # op shape: (16, 8, 16)
tiled_mma = cute.make_tiled_mma(
    atom,
    atom_layout_mnk=(2, 2, 1),    # 4 warps: 2 in M, 2 in N
)                                 # total tiled-MMA tile = (32, 16, 16)
```

The coordinates of atoms could be thought as a 3D coordinate: `(m, n, k)`. `m` is the M repeat index, `n` is the N repeat index, and `k` is the K repeat index. Each warp MMA atom is executed by a single warp within a single CTA.

``` text
Warp MMA Atom (16x8x16)                make_tiled_mma(atom, (2, 2, 1))
+----------------+                     +----------------+----------------+
|                |                     |                |                | ^
|   16 x 8       |                     | Atom (0,0,0)   | Atom (0,1,0)   | |
|     x 16       |   --(2,2,1)-->      |   16 x 8       |   16 x 8       | | 2 x inst_M
|                |      repeat         |     x 16       |     x 16       | |  = 32
|                |                     | [Warp 0]       | [Warp 2]       | |
+----------------+                     +----------------+----------------+ |
                                      |                |                | |
                                      | Atom (1,0,0)   | Atom (1,1,0)   | |
                                      |   16 x 8       |   16 x 8       | |
                                      |     x 16       |     x 16       | |
                                      | [Warp 1]       | [Warp 3]       | v
                                      +----------------+----------------+
                                      <--- 2 x inst_N = 16 --->
                                      K unchanged = 16
```

#### Custom tile permutation with `permutation_mnk`

`permutation_mnk` is an optional third argument to `make_tiled_mma`. Each of its three entries is a **per-mode permutation** of the M, N, and K coordinates inside the tiled MMA. In the common case shown in this section, each entry is just a size, which is the identity permutation of that size; in that case `permutation_mnk` simply sets the **total tile footprint** of the tiled MMA along each dimension. When a mode's size is larger than the atom layout's natural coverage (`atom_layout x inst_shape`), each thread receives additional values to fill the extended region — the thread count stays the same, but every thread holds more data. The general form, where an entry is a `Layout` that reorders coordinates inside a mode, is covered in the subsection below.

The standard convention for warp MMA (used in `tensorop_gemm.py` and throughout the Ampere examples) doubles the N dimension:

``` python
# From examples/cute/ampere/kernel/dense_gemm/tensorop_gemm.py
permutation_mnk = (
    atom_layout_mnk[0] * mma_inst_shape[0],      # M: matches atom coverage
    atom_layout_mnk[1] * mma_inst_shape[1] * 2,   # N: 2x atom coverage
    atom_layout_mnk[2] * mma_inst_shape[2],        # K: matches atom coverage
)

tC = cute.make_layout(atom_layout_mnk)
tiled_mma = cute.make_tiled_mma(
    op,
    tC,
    permutation_mnk=permutation_mnk,
)
```

**Why double N?** The atom's N dimension is only 8 (inst_N = 8). Without a permutation, each thread's B-operand values span a single 8-wide N-range, which may not align well with SMEM load widths. The `* 2` on N gives each thread's B fragment two 8-wide N-ranges instead of one, aligning the access pattern with wider contiguous SMEM regions for more efficient loads.

For `atom_layout_mnk = (2, 2, 1)` and `inst_shape = (16, 8, 16)`:

- Atom coverage = `(2x16, 2x8, 1x16) = (32, 16, 16)`
- `permutation_mnk = (32, 32, 16)` — N extended from 16 to 32

``` text
Without permutation — natural atom coverage (M = 32, N = 16):

C tile (M=32, N=16)
+----------------+----------------+
|                |                | ^
|   [Warp 0]     |   [Warp 2]     | |
|    16 x 8      |    16 x 8      | | 2 x inst_M
|                |                | |  = 32
+----------------+----------------+ |
|                |                | |
|   [Warp 1]     |   [Warp 3]     | |
|    16 x 8      |    16 x 8      | |
|                |                | v
+----------------+----------------+
<------------- N = 16 ---------->
(each warp owns one (16, 8) atom;
 thread T0 of Warp 0 holds 4 C values in its 16x8 block)

With permutation_mnk = (32, 32, 16) — N extended from 16 to 32:

C tile (M=32, N=32)
+----------------+----------------+----------------+----------------+
|                |                |                |                | ^   N = 16 → 32:
|   [Warp 0]     |   [Warp 2]     |   [Warp 0]     |   [Warp 2]     | |   atom pattern repeats
|    16 x 8      |    16 x 8      |    16 x 8      |    16 x 8      | |   along N. Each thread
|                |                |                |                | |   now holds 2x the
+----------------+----------------+----------------+----------------+ |   values along N
|                |                |                |                | |   (same threads, more
|   [Warp 1]     |   [Warp 3]     |   [Warp 1]     |   [Warp 3]     | |   values per thread).
|    16 x 8      |    16 x 8      |    16 x 8      |    16 x 8      | |
|                |                |                |                | v
+----------------+----------------+----------------+----------------+
<---------------------------- N = 32 ---------------------------->
|        atom coverage            |          value repeat           |
```

##### Reordering coordinates with a per-mode `Layout`

So far each entry of `permutation_mnk` has been an integer, which is shorthand for the identity layout `Layout<Shape<S>, Stride<_1>>` — the atom pattern simply tiles to fill an `S`-wide footprint. The general form lets each entry be a `Layout` that **reorders coordinates inside that mode** while keeping the same total size. That reordering is what gives the parameter its name; the integer-only cases used earlier are just the identity permutation.

The canonical illustration is the SM70 example from [0t_mma_atom.md](https://github.com/NVIDIA/cutlass/blob/ad9bd53bdaec27a2e88053d57322ccf74efe525e/media/docs/cpp/cute/0t_mma_atom.md). Take a 2x2 tiled MMA of `SM70_8x8x4_F32F16F16F32_NT` atoms with a `32x32x4` footprint. Without any M-mode permutation, thread `T0`'s 8 A-values land at the following `(m, k)` coordinates:

    T0V0 => (0, 0)     T0V4 => (16, 0)
    T0V1 => (1, 0)     T0V5 => (17, 0)
    T0V2 => (2, 0)     T0V6 => (18, 0)
    T0V3 => (3, 0)     T0V7 => (19, 0)

— two separate runs of 4 along M, with a gap from m=4 to m=15. We may prefer those 8 values to sit in **one contiguous run** in the logical M-coordinates (e.g. so register or SMEM layouts pack cleanly). Passing the M-mode layout `(4, 4, 2):(1, 8, 4)` does exactly that: it is a scatter permutation telling each old m-coord where to go in the new image.

``` text
old m-coord:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
new m-coord:  0  1  2  3  8  9 10 11 16 17 18 19 24 25 26 27  4  5  6  7 12 13 14 15 20 21 22 23 28 29 30 31
```

After the permutation, `T0`'s 8 A-values occupy `m = 0..7` — one contiguous run — and every other thread's M-values become equally contiguous. Thread-data ownership and value counts are unchanged; only the **mapping from values to m-coordinates** is permuted.

In CuTeDSL the permuted entry is built with `cute.make_layout`; identity entries stay as integers:

``` python
m_perm = cute.make_layout((4, 4, 2), stride=(1, 8, 4))
tiled_mma = cute.make_tiled_mma(
    op,                                # SM70_8x8x4 NT atom
    atom_layout_mnk=(2, 2, 1),
    permutation_mnk=(m_perm, 32, 4),   # M: scatter, N/K: identity sizes
)
```

The same mechanism applies to the N and K modes — any subset of the three entries can be an integer (identity) or a `Layout` (real permutation). For warp MMAs the most common case in practice is still the integer-only form shown earlier in this section; the `Layout` form is the tool you reach for when a register or SMEM layout wants each thread's fragment to be contiguous in logical coordinates.

### Partitioning Tensors

Before computing, partition the CTA-tiled tensors according to the tiled MMA layout. Warp MMA partitioning is **per-thread**: each of the 32 threads in a warp (or 128 threads across 4 warps) receives its own slice of the data, sized to match the register fragments the MMA instruction expects.

Example: `GEMM (M, N, K) = (512, 512, 256)`, `cta_tiler = (128, 128, 32)`, `atom_layout_mnk = (2, 2, 1)`, F16 atom = m16n8k16, `permutation_mnk = (32, 32, 16)`, `num_stages = 4`, 4 warps = 128 threads.

Global matrices:

``` text
mA: (M, K) = (512, 256)       mB: (N, K) = (512, 256)       mC: (M, N) = (512, 512)

     K=256                          K=256                       N=512
   |<--------->|                |<--------->|              |<---------------->|
   +-----------+                +-----------+              +----+----+----+---+
   |           | ^              |           | ^            |    |    |    |   | ^
   |    mA     | | M=512        |    mB     | | N=512      |    |    |    |   | | M=512
   |           | v              |           | v            |    |    |    |   | v
   +-----------+                +-----------+              +----+----+----+---+
```

Tiling with `cta_tiler = (BM, BN, BK) = (128, 128, 32)` gives M/BM = 4 tiles, N/BN = 4 tiles, K/BK = 8 tiles:

``` text
mA tiled into (M/BM x K/BK)    mB tiled into (N/BN x K/BK)    mC tiled into (M/BM x N/BN)
= (4 x 8) blocks               = (4 x 8) blocks               = (4 x 4) blocks

  BK=32  x8                       BK=32  x8                       BN=128  x4
|<-->|                          |<-->|                          |<------>|
+----+----+-- --+               +----+----+-- --+               +--------+--------+-- --+
|    |    |..|  | ^  BM=128     |    |    |..|  | ^  BN=128     | (0,0)  | (0,1)  |..   | ^  BM=128
+----+----+-- --+ v             +----+----+-- --+ v             +--------+--------+     + v
|    |    |..|  | ^  BM=128     |    |    |..|  | ^  BN=128     | (1,0)  | (1,1)  |..   | ^  BM=128
+----+----+-- --+ v             +----+----+-- --+ v             +--------+--------+     + v
|    |    |..|  | ^             |    |    |..|  | ^             |  ...   |  ...   |..   | ^
+----+----+-- --+ v             +----+----+-- --+ v             +--------+--------+-- --+ v
|    |    |..|  | ^             |    |    |..|  | ^             | (3,0)  | (3,1)  |..   | ^
+----+----+-- --+ v             +----+----+-- --+ v             +--------+--------+-- --+ v
```

Each CTA picks one (M-tile, N-tile) coordinate. For example, CTA at `tiler_coord = (0, 1, :)`.

After `local_tile` — one CTA's tile (`k = K/BK = 256/32 = 8`):

``` text
gA: (BM, BK, k) = (128, 32, 8)   gB: (BN, BK, k) = (128, 32, 8)   gC: (BM, BN) = (128, 128)

     BK=32                             BK=32                       BN=128
   |<----->|                        |<----->|                  |<-------->|
   +-------+--                      +-------+--                +----------+
   |       |..                      |       |..                |          | ^
BM=  |  gA   | k=8                BN= |  gB   | k=8          BM= |    gC    | | 128
128 |       |                    128 |       |              128 |          | v
   +-------+                        +-------+                  +----------+
```

SMEM tensors `sA` and `sB` have a pipeline staging dimension:

``` text
sA: (BM, BK, PIPE) = (128, 32, 4)         sB: (BN, BK, PIPE) = (128, 32, 4)
```

`get_slice(tidx)` — each thread receives its own per-thread partition. The tiled MMA footprint is `permutation_mnk = (32, 32, 16)`, so BM, BN, and BK are each subdivided into MMA-sized blocks:

``` text
sA: partition into (MMA, MMA_M, MMA_K, PIPE)

Each SMEM stage (BM=128, BK=32):

perm_K perm_K                               perm_M=32
  =16    =16                                |<---->|
|<--->|<--->|                               +------+------+------+------+
+-----+-----+  ^                            |      |      |      |      | ^
|  0  |  1  |  |  perm_M=32                 |  0   |  1   |  2   |  3   | | perm_N
+-----+-----+  v                            |      |      |      |      | v  =32
|  0  |  1  |  ^                            +------+------+------+------+
|     |     |  |  perm_M=32                   MMA_N = BN/perm_N = 4
+-----+-----+  v
|  0  |  1  |  ^                           sB: partition into (MMA, MMA_N, MMA_K, PIPE)
|     |     |  |
+-----+-----+  v                           gC: partition into (MMA, MMA_M, MMA_N)
|  0  |  1  |  ^
|     |     |  |
+-----+-----+  v
  MMA_K = BK/perm_K = 2
  MMA_M = BM/perm_M = 4
```

After partition (per thread, e.g. thread `tidx`):

- `tCsA: (MMA, MMA_M, MMA_K, PIPE) = (MMA, 4, 2, 4)` — MMA_M = BM/perm_M = 128/32 = 4, MMA_K = BK/perm_K = 32/16 = 2
- `tCsB: (MMA, MMA_N, MMA_K, PIPE) = (MMA, 4, 2, 4)` — MMA_N = BN/perm_N = 128/32 = 4, MMA_K = BK/perm_K = 32/16 = 2
- `tCgC: (MMA, MMA_M, MMA_N) = (MMA, 4, 4)` — MMA_M = 128/32 = 4, MMA_N = 128/32 = 4

The first mode `MMA` contains the atom's **thread × value** layout — it encodes which registers within a single thread hold which matrix elements. The remaining modes are repeat counts that tile the atom across the full CTA tile.

``` python
@cute.kernel
def kernel(tiled_mma: cute.TiledMma, ...):
    tidx, _, _ = cute.arch.thread_idx()

    # CTA-tiled global tensors
    gA = cute.local_tile(mA, cta_tiler, tiler_coord, proj=(1, None, 1))
    gB = cute.local_tile(mB, cta_tiler, tiler_coord, proj=(None, 1, 1))
    gC = cute.local_tile(mC, cta_tiler, tiler_coord, proj=(1, 1, None))

    # Per-thread partition via the thread index
    thr_mma = tiled_mma.get_slice(tidx)

    # SMEM partitions (used by make_fragment_A/B and ldmatrix retiling)
    tCsA = thr_mma.partition_A(sA)   # (MMA, MMA_M, MMA_K, PIPE)
    tCsB = thr_mma.partition_B(sB)   # (MMA, MMA_N, MMA_K, PIPE)

    # C partitions for epilogue staging (SMEM) and destination (GMEM)
    tCsC = thr_mma.partition_C(sC)   # (MMA, MMA_M, MMA_N)
    tCgC = thr_mma.partition_C(gC)   # (MMA, MMA_M, MMA_N)
```

> [!NOTE]
> The `tCsA` / `tCsB` SMEM partitions are not read directly by the GEMM — they establish the **shape** that `make_fragment_A` / `make_fragment_B` use to allocate register fragments. Actual SMEM→RMEM data movement goes through the S2R `ldmatrix` retiling path (see [Making Fragments](#making-fragments)).

### Pre and Post-Conditions for Partitioning

- The inputs of the partition should be at least rank-2 tensors.
- The output of the partition will have the layout that is compatible with the MMA atom's operand:
  - For A, the output will have the layout `(MMA, MMA_M, MMA_K, ...)`.
  - For B, the output will have the layout `(MMA, MMA_N, MMA_K, ...)`.
  - For C, the output will have the layout `(MMA, MMA_M, MMA_N, ...)`.
- Note that the partition doesn't enforce any rules on the tensor's memory space or the tensor's data type. It only cares about the layout.

### Making Fragments

Fragments are the tensors that the warp MMA instruction operates on. For warp MMA:

- **Fragment A**: per-thread register fragment holding one operand-A K-block.
- **Fragment B**: per-thread register fragment holding one operand-B K-block.
- **Fragment C (accumulator)**: per-thread register fragment that lives in RMEM and serves as both the input C and output D of `cute.gemm()`.

#### Creating register fragments and `ldmatrix` copy views

Warp MMA fragments are actual per-thread register tensors, not descriptors. Fragment creation has three parts:

**1. A and B fragments**

`make_fragment_A` and `make_fragment_B` take one stage of the MMA-partitioned SMEM views (`tCsA` / `tCsB`) and allocate register fragments with a matching thread-local layout. This establishes the shape only; no data is loaded yet.

``` python
# Per-thread MMA partitions
# (sA/sB are the staged SMEM tensors — see "Creating SMEM layouts for A and B")
tCsA = thr_mma.partition_A(sA)   # (MMA, MMA_M, MMA_K, PIPE)
tCsB = thr_mma.partition_B(sB)   # (MMA, MMA_N, MMA_K, PIPE)

# Register fragments for one pipeline stage
tCrA = tiled_mma.make_fragment_A(
    tCsA[None, None, None, 0]
)  # (MMA, MMA_M, MMA_K)
tCrB = tiled_mma.make_fragment_B(
    tCsB[None, None, None, 0]
)  # (MMA, MMA_N, MMA_K)
```

Continuing the running example from [Partitioning Tensors](#partitioning-tensors) (F16 `m16n8k16`, `cta_tiler = (128, 128, 32)`, `permutation_mnk = (32, 32, 16)`, `num_stages = 4`):

``` text
tCsA: (MMA, MMA_M=4, MMA_K=2, PIPE=4)
tCsB: (MMA, MMA_N=4, MMA_K=2, PIPE=4)

make_fragment_A(tCsA[..., stage]) -> tCrA: (MMA, 4, 2)
make_fragment_B(tCsB[..., stage]) -> tCrB: (MMA, 4, 2)
```

Each element of `tCrA` / `tCrB` is a register value owned by the current thread. Together, the 32 threads in the warp hold the full operand fragment that one `mma.sync.aligned` instruction consumes.

**2. C fragment (accumulator)**

`make_fragment_C` allocates the accumulator registers for the CTA tile slice owned by the current thread. The accumulator usually starts at zero before the K loop and is updated in-place by each `cute.gemm()` call.

``` python
tCgC = thr_mma.partition_C(gC)   # (MMA, MMA_M, MMA_N)
tCrC = tiled_mma.make_fragment_C(tCgC)
tCrC.fill(0.0)
```

For the same running example:

``` text
tCgC: (MMA, MMA_M=4, MMA_N=4)
make_fragment_C(tCgC) -> tCrC: (MMA, 4, 4)
```

`tCrC` stays in registers for the entire main loop and serves as both the input C and output D argument of `cute.gemm()`.

**3. SMEM → RMEM load (\`\`ldmatrix\`\` retiling)**

The register fragments above are storage only — before `cute.gemm()` can consume `tCrA` and `tCrB`, each K-block must be loaded from shared memory into those registers. This is done via a separate tiled copy built from an `ldmatrix` copy atom and linked to the tiled MMA with `make_tiled_copy_A` / `make_tiled_copy_B`. The copy's `retile()` call remaps the MMA fragment's register layout to match what the `ldmatrix` instruction writes.

``` python
# 1. Create ldmatrix copy atom → tiled copy tied to the MMA layout
s2r_atom_A = cute.make_copy_atom(LdMatrix8x8x16bOp(...), dtype)
s2r_tiled_A = cute.make_tiled_copy_A(s2r_atom_A, tiled_mma)

# 2. Build SMEM-side and RMEM-side views for the copy
thr_s2r_A = s2r_tiled_A.get_slice(tidx)
tCsA_copy_view = thr_s2r_A.partition_S(sA)   # SMEM source
tCrA_copy_view = thr_s2r_A.retile(tCrA)      # RMEM dest (retiled)

# 3. Load one k-block from SMEM into the MMA fragment (in the main loop)
cute.copy(s2r_tiled_A, tCsA_copy_view[None, None, k_block],
          tCrA_copy_view[None, None, k_block])
```

See `tensorop_gemm.py` for the complete implementation including the `ldmatrix` transpose flag, FP8 variants, and operand B.

#### Creating SMEM layouts for A and B

The SMEM layouts define how A and B tiles are staged in shared memory before the `ldmatrix` loads. For warp MMA, these layouts must satisfy two goals at the same time:

- **Efficient GMEM -\> SMEM copy:** `cp.async` should write contiguous 16-byte regions for each thread.
- **Bank-conflict-free SMEM -\> RMEM load:** the later `ldmatrix` loads should see a swizzled layout that matches the warp MMA operand access pattern.

The Ampere dense GEMM example (`examples/cute/ampere/kernel/dense_gemm/tensorop_gemm.py`) builds these layouts inline with a helper named `_make_smem_layout_AB`.

**Host side** (`@cute.jit`):

``` python
# 16 bytes per thread for GMEM -> SMEM copies
ab_copy_bits = 128

sA_layout, sA_swizzle = self._make_smem_layout_AB(
    mA.element_type,       # dtype (e.g. Float16)
    self.a_major_mode,     # row-major or col-major
    ab_copy_bits,          # copy width in bits (128 = 16 bytes)
    (self.cta_tiler[0],    # BM
     self.cta_tiler[2],    # BK
     self.num_stages),     # PIPE
)
sB_layout, sB_swizzle = self._make_smem_layout_AB(
    mB.element_type,
    self.b_major_mode,
    ab_copy_bits,
    (self.cta_tiler[1],    # BN
     self.cta_tiler[2],    # BK
     self.num_stages),     # PIPE
)
```

Here `smem_tiler` is `(M_or_N, K, PIPE)`: `(BM, BK, PIPE)` for A and `(BN, BK, PIPE)` for B. The helper returns:

- `sX_layout`: the logical SMEM layout with shape `(BM_or_BN, BK, PIPE)`.
- `sX_swizzle`: the swizzle applied when the tensor is materialized in SMEM.

The helper from `tensorop_gemm.py` implements the following four steps:

1.  **Pick the major-mode size.** For a row-major operand, the contiguous dimension is K, so the helper uses `smem_tiler[1]`. For a col-major operand, the contiguous dimension is M or N, so it uses `smem_tiler[0]`.
2.  **Cap the contiguous span at 128 bytes.** This keeps the layout atom within the swizzle span used by the example. The cap is 64 elements for F16/BF16 and 128 elements for FP8.
3.  **Build the swizzle.** With `copy_bits = 128` (16 bytes), the helper derives three arguments for `make_swizzle`:
    - `swizzle_bits = log2(major_mode_size * dtype.width / copy_bits)`, capped at 3. This is the number of address bits that get XOR'd.
    - `base_bits = log2(copy_bits / 8)` — log2 of the copy width in bytes (= 4 for 16-byte copies).
    - `shift_bits = log2(copy_bits / dtype.width)` — log2 of the copy width in elements (= 3 for F16 with 128-bit copies, i.e. 8 elements).
4.  **Build an 8-row layout atom and tile it.** The constant 8 comes from `ldmatrix`: each warp-level load touches 8 rows of shared memory (32 threads, 4 matrices per load). Row-major uses an atom `(8, major_mode_size):(major_mode_size, 1)` — 8 rows of contiguous K-elements. Col-major uses `(major_mode_size, 8):(1, major_mode_size)` — contiguous MN-elements across 8 K-rows. `tile_to_shape` then broadcasts that atom across the full `(M_or_N, K, PIPE)` SMEM tensor.

For the running F16 example (`cta_tiler = (128, 128, 32)`, `num_stages = 4`, `copy_bits = 128`):

``` text
A operand (row-major, smem_tiler = (128, 32, 4)):
  major_mode_size = 32
  atom = (8, 32):(32, 1)
  swizzle = make_swizzle(2, 4, 3)
  tiled layout -> sA: (128, 32, 4)

B operand (col-major, smem_tiler = (128, 32, 4)):
  major_mode_size = min(128, 64) = 64
  atom = (64, 8):(1, 64)
  swizzle = make_swizzle(3, 4, 3)
  tiled layout -> sB: (128, 32, 4)
```

**Kernel side** (`@cute.kernel`):

The layout and swizzle are passed to shared-memory allocation:

``` python
@cute.struct
class SharedStorageAB:
    a: cute.struct.Align[
        cute.struct.MemRange[mA.element_type, cute.cosize(sA_layout)],
        16,
    ]
    b: cute.struct.Align[
        cute.struct.MemRange[mB.element_type, cute.cosize(sB_layout)],
        16,
    ]

sA = SharedStorageAB(storage).a.get_tensor(sA_layout, swizzle=sA_swizzle)
sB = SharedStorageAB(storage).b.get_tensor(sB_layout, swizzle=sB_swizzle)
```

After allocation:

- `sA` has shape `(BM, BK, PIPE)`.
- `sB` has shape `(BN, BK, PIPE)`.

These are the staged SMEM tensors written by `cp.async` and later consumed by `partition_A` / `partition_B`, `make_fragment_A` / `make_fragment_B`, and the `ldmatrix` copy views described in [Making Fragments](#making-fragments).

### Executing the GEMM (Main Loop)

The main loop iterates over K-tiles and, within each tile, over k-blocks (`num_k_block = BK / perm_K`). Each k-block loads A and B from SMEM into registers via `ldmatrix`, then issues `cute.gemm`.

``` python
tCrC.fill(0.0)

for k_tile in range(k_tile_count):
    for k_block in cutlass.range(num_k_block, unroll_full=True):
        # Wait for next SMEM stage at the tile boundary
        if k_block == num_k_block - 1:
            cute.arch.cp_async_wait_group(num_smem_stages - 2)
            cute.arch.sync_threads()

        # ldmatrix: prefetch next k-block from SMEM → RMEM
        k_block_next = (k_block + 1) % num_k_block
        cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, k_block_next],
                  tCrA_copy_view[None, None, k_block_next])
        cute.copy(tiled_copy_s2r_B, tCsB_p[None, None, k_block_next],
                  tCrB_copy_view[None, None, k_block_next])

        # cp.async: issue GMEM → SMEM for next K-tile
        # ... (see tensorop_gemm.py for pipeline pointer management)

        # MMA: tCrC += tCrA * tCrB
        cute.gemm(tiled_mma, tCrC, tCrA[None, None, k_block], tCrB[None, None, k_block], tCrC)

cute.arch.cp_async_wait_group(0)
cute.arch.sync_threads()
```

Key points:

- `cute.gemm` is **synchronous** — it emits `mma.sync.aligned` instructions. There is no accumulate-mode flag; the accumulator (`tCrC`) is always read and written.
- All operands must be in **registers** before `cute.gemm` is called. The `ldmatrix` copies above prefetch the next k-block into `tCrA` / `tCrB` from SMEM each iteration.
- The `cp.async` / `cp_async_wait_group` calls manage the GMEM→SMEM pipeline; see `tensorop_gemm.py` for predication, K-residue handling, and pipeline pointer management.

### Complete Workflow

Putting it all together, a typical Ampere warp MMA GEMM has this structure:

**Host function** (`@cute.jit`):

``` python
import cutlass
import cutlass.cute as cute

@cute.jit
def host_function(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, stream):
    # 1. Create the MMA op and tiled MMA
    op = cute.nvgpu.warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (16, 8, 16))
    atom_layout_mnk = (2, 2, 1)
    permutation_mnk = (
        atom_layout_mnk[0] * 16,
        atom_layout_mnk[1] * 8 * 2,
        atom_layout_mnk[2] * 16,
    )
    tC = cute.make_layout(atom_layout_mnk)
    tiled_mma = cute.make_tiled_mma(op, tC, permutation_mnk=permutation_mnk)

    # 2. Create SMEM layouts
    ab_copy_bits = 128
    sA_layout, sA_swizzle = _make_smem_layout_AB(
        mA.element_type, a_major_mode, ab_copy_bits,
        (cta_tiler[0], cta_tiler[2], num_stages),
    )
    sB_layout, sB_swizzle = _make_smem_layout_AB(
        mB.element_type, b_major_mode, ab_copy_bits,
        (cta_tiler[1], cta_tiler[2], num_stages),
    )

    # 3. Launch the kernel
    kernel(mA, mB, mC, ..., tiled_mma, sA_layout, sA_swizzle,
           sB_layout, sB_swizzle).launch(
        grid=grid, block=[128, 1, 1], stream=stream,
    )
```

**Kernel function** (`@cute.kernel`):

``` python
@cute.kernel
def kernel(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
          ..., tiled_mma: cute.TiledMma):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()

    # -- CTA-tiled global tensors --
    gA = cute.local_tile(mA[None, None, bidz], cta_tiler, (bidx, bidy, None), proj=(1, None, 1))
    gB = cute.local_tile(mB[None, None, bidz], cta_tiler, (bidx, bidy, None), proj=(None, 1, 1))
    gC = cute.local_tile(mC[None, None, bidz], cta_tiler, (bidx, bidy, None), proj=(1, 1, None))

    # -- Allocate SMEM --
    @cute.struct
    class SharedStorageAB:
        a: cute.struct.Align[cute.struct.MemRange[mA.element_type, cute.cosize(sA_layout)], 16]
        b: cute.struct.Align[cute.struct.MemRange[mB.element_type, cute.cosize(sB_layout)], 16]

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorageAB)
    sA = SharedStorageAB(storage).a.get_tensor(sA_layout, swizzle=sA_swizzle)  # (BM, BK, PIPE)
    sB = SharedStorageAB(storage).b.get_tensor(sB_layout, swizzle=sB_swizzle)  # (BN, BK, PIPE)
    sC = ...  # (BM, BN) SMEM for epilogue (non-MMA, see tensorop_gemm.py)

    # -- GMEM → SMEM copy partitions (cp.async) --
    # ... setup tAgA, tAsA, tBgB, tBsB (see tensorop_gemm.py)

    # -- MMA partitions and fragments --
    thr_mma = tiled_mma.get_slice(tidx)
    tCsA = thr_mma.partition_A(sA)   # (MMA, MMA_M, MMA_K, PIPE)
    tCsB = thr_mma.partition_B(sB)   # (MMA, MMA_N, MMA_K, PIPE)
    tCsC = thr_mma.partition_C(sC)   # (MMA, MMA_M, MMA_N)
    tCgC = thr_mma.partition_C(gC)   # (MMA, MMA_M, MMA_N)
    tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])  # (MMA, MMA_M, MMA_K)
    tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])  # (MMA, MMA_N, MMA_K)
    tCrC = tiled_mma.make_fragment_C(tCgC)                       # (MMA, MMA_M, MMA_N)
    tCrC.fill(0.0)

    # -- ldmatrix retiling (see "Making Fragments" § SMEM → RMEM load) --
    # ... build tiled_copy_s2r_A/B from LdMatrix8x8x16bOp + make_tiled_copy_A/B
    # ... then: tCsA_copy_view = partition_S(sA), tCrA_copy_view = retile(tCrA), etc.

    # -- Prologue: cp.async fills num_stages-1 SMEM buffers --
    # -- Prefetch first k-block into registers via ldmatrix --
    # ... (see tensorop_gemm.py for predication, residual_k, and pipeline setup)

    # -- Main loop --
    for k_tile in range(k_tile_count):
        for k_block in cutlass.range(num_k_block, unroll_full=True):
            if k_block == num_k_block - 1:
                cute.arch.cp_async_wait_group(num_smem_stages - 2)
                cute.arch.sync_threads()

            # ldmatrix: prefetch next k-block from SMEM → RMEM
            # tCsA_p / tCsB_p are per-pipeline-stage slices, e.g.:
            #   tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
            k_block_next = (k_block + 1) % num_k_block
            cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, k_block_next],
                      tCrA_copy_view[None, None, k_block_next])
            cute.copy(tiled_copy_s2r_B, tCsB_p[None, None, k_block_next],
                      tCrB_copy_view[None, None, k_block_next])

            # cp.async: issue GMEM → SMEM for next K-tile
            # ... (see tensorop_gemm.py for pipeline pointer management)

            # MMA
            cute.gemm(tiled_mma, tCrC, tCrA[None, None, k_block],
                      tCrB[None, None, k_block], tCrC)

    # -- Epilogue: RMEM → SMEM → RMEM → GMEM --
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()
    tCrD = cute.make_fragment_like(tCrC, c_dtype)
    tCrD[None] = epilogue_op(tCrC.load()).to(c_dtype)
    cute.autovec_copy(tCrD, tCsC)   # RMEM → SMEM
    cute.arch.sync_threads()
    # ... reload with epilogue thread layout, then vectorized store to GMEM
```

### Beyond Simple Dense MMAs

The warp MMA DSL supports more complex MMA operations beyond simple dense MMA:

- Block-scaled MMA

Internal builds additionally provide:

- Sparse MMA

#### Sparse MMA

Sparse MMA exploits **2:4 structured sparsity** in operand A: out of every 4 consecutive K-elements, exactly 2 are non-zero. The hardware consumes a compressed A operand together with a compact **metadata** tensor `E` that encodes which 2 of 4 positions are non-zero.

Compared to dense MMA, the MMA API differences are:

**1. MMA op creation** — use `MmaF16BF16SparseOp` with an extra `sparse_metadata_format` parameter. The sparse instruction K is doubled relative to dense (dense `m16n8k8` → sparse `m16n8k16`, dense `m16n8k16` → sparse `m16n8k32`) because operand A is 2:4 compressed:

``` python
from cutlass.cute.nvgpu.warp.mma import SparseMetadataFormat

# Dense F16 (for comparison): inst_K = 16
dense_op = cute.nvgpu.warp.MmaF16BF16Op(
    cutlass.Float16, cutlass.Float32, (16, 8, 16),
)

# Sparse F16: inst_K = 32 (2× dense, since A is 2:4 compressed)
sparse_op = cute.nvgpu.warp.MmaF16BF16SparseOp(
    cutlass.Float16,                         # A/B element type
    cutlass.Float32,                         # accumulator type
    (16, 8, 32),                             # instruction shape (M, N, K)
    SparseMetadataFormat.TID,                # metadata format
)
tiled_mma = cute.make_tiled_mma(sparse_op, cute.make_layout((1, 1, 1)))
```

``` text
Supported instruction shapes for MmaF16BF16SparseOp:

| A/B Type | Acc Type  | Inst Shape     |
|----------|-----------|----------------|
| F16      | F16, F32  | (16,8,16), (16,8,32) |
| BF16     | F32       | (16,8,16), (16,8,32) |
```

**2. Compressed A tensor and metadata E** — operand A stores only the two non-zero values per group of 4 K-elements (half the storage). The metadata tensor `E` records which 2 of 4 positions are non-zero. The exact bit encoding depends on `SparseMetadataFormat` and on how the implementation packs metadata. In this repository, helper code that generates 2:4 test inputs packs two 4-bit metadata entries into each `uint8` value:

``` python
# Example metadata values used by examples/CuTeDSL/helpers/sparse_utils.py
# Each nibble selects which 2 of 4 positions are non-zero.
metadata_values = [0x4, 0x8, 0x9, 0xC, 0xD, 0xE]
```

``` text
Dense A: (M, K)                    Sparse operands:
+--+--+--+--+--+--+--+--+         +--+--+--+--+
| a| 0| b| 0| c| 0| d| 0|   →     | a| b| c| d|   (compressed A values)
+--+--+--+--+--+--+--+--+         +--+--+--+--+

                                  E stores the non-zero positions
                                  for each 2:4 group.
```

**3. Fragments** — the dense-style fragment APIs for A, B, and C still apply to the sparse atom:

``` python
# A/B/C fragments — same public API shape as dense
tCsA = thr_mma.partition_A(sA)
tCsB = thr_mma.partition_B(sB)
tCgC = thr_mma.partition_C(gC)

tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
tCrC = tiled_mma.make_fragment_C(tCgC)
tCrC.fill(0.0)
```

Sparse metadata `E` is an auxiliary operand associated with A. The public warp API and tests in this repository verify op construction and the `cute.gemm(..., [A, E], B, ...)` calling convention, but they do not provide an end-to-end warp sparse kernel showing the exact `partition` / `copy` / `make_fragment` sequence for `E`. For that reason, this document intentionally does not spell out an `E` fragment construction sequence that has no example backing it.

**4. Modified gemm call** — the metadata E is passed alongside operand A as a list. This part of the API is verified by `cutlass.cute.algorithm.gemm`:

``` python
# Schematic only: E_k is the metadata operand for the same k-slice as A_k.
A_k = tCrA[None, None, k_block]
E_k = metadata_k
B_k = tCrB[None, None, k_block]

cute.gemm(
    tiled_mma,
    tCrC,
    [A_k, E_k],   # [A, E]
    B_k,
    tCrC,
)
```

``` text
Dense gemm call:
  cute.gemm(tiled_mma, tCrC, A_k, B_k, tCrC)

Sparse gemm call:
  cute.gemm(tiled_mma, tCrC, [A_k, E_k], B_k, tCrC)
                            ^^^^  ^^^
                            A     metadata
```

The epilogue (RMEM → SMEM → GMEM) is identical to a dense kernel.

> [!NOTE]
> An end-to-end warp sparse GEMM example is not yet available in the examples directory. The closest verified references in this repository are `cutlass_ir/compiler/test/python/not_pytest/sm_80/test_mma_atom.py` for op construction, `cutlass_ir/compiler/test/python/api/sm_120a/test_nvgpu_warp_mma.py` for tiled sparse MMA construction, and `examples/CuTeDSL/helpers/sparse_utils.py` for 2:4 metadata packing.

#### Block-scaled MMA

Block-scaled MMA multiplies narrow-type matrices (FP4) while applying **per-block scale factors** along the GEMM-K dimension. Each vector of `sf_vec_size` consecutive K-elements shares a single scale factor, so the hardware computes `D = (SFA · A) * (SFB · B) + C`. The scale factors live in **registers** alongside the operands and must be loaded from SMEM before each `gemm` call.

Supported ops: `MmaMXF4Op` (SM120a+), `MmaMXF4NVF4Op` (SM120a+).

Compared to a dense MMA kernel, a block-scaled kernel has four additional concerns:

**1. MMA op creation** — block-scaled ops fix the data type to FP4 (E2M1) and the accumulator to FP32. The scale-factor type and vector size distinguish the two ops:

``` python
# MXF4: UE8M0 scales, sf_vec_size = 32
op = cute.nvgpu.warp.MmaMXF4Op(
    cutlass.Float4E2M1FN,     # A/B element type (fixed: E2M1)
    cutlass.Float32,          # accumulator type (fixed: F32)
    cutlass.Float8E8M0FNU,    # scale-factor type
)  # instruction shape = (16, 8, 64), sf_vec_size = 32

# MXF4NVF4: UE4M3 scales, sf_vec_size = 16
op = cute.nvgpu.warp.MmaMXF4NVF4Op(
    cutlass.Float4E2M1FN,     # A/B element type (fixed: E2M1)
    cutlass.Float32,          # accumulator type (fixed: F32)
    cutlass.Float8E4M3FN,     # scale-factor type
)  # instruction shape = (16, 8, 64), sf_vec_size = 16
```

``` text
| Op            | A/B Type | SF Type | Acc  | Inst Shape  | SF Vec Size |
|---------------|----------|---------|------|-------------|-------------|
| MmaMXF4Op     | E2M1     | UE8M0   | F32  | (16,8,64)   | 32          |
| MmaMXF4NVF4Op | E2M1     | UE4M3   | F32  | (16,8,64)   | 16          |
```

**2. Extra global tensors and SMEM layouts for scale factors** — the host function creates SFA/SFB tensors and allocates SMEM layouts for them alongside A and B:

``` python
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.blackwell_helpers as sm120_utils

# Scale-factor global tensors (host side)
sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a.shape, sf_vec_size)
sfa_tensor = cute.make_tensor(sfa.iterator, sfa_layout)
sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b.shape, sf_vec_size)
sfb_tensor = cute.make_tensor(sfb.iterator, sfb_layout)

# SMEM layouts for scale factors (SM120-specific helper)
sfa_smem_layout = blockscaled_utils.sm120_make_smem_layout_sfa(
    tiled_mma, tile_shape_mnk, sf_vec_size, num_stages,
)
sfb_smem_layout = blockscaled_utils.sm120_make_smem_layout_sfb(
    tiled_mma, tile_shape_mnk, sf_vec_size, num_stages,
)
```

**3. SF fragment creation and SMEM→RMEM retiling** — scale-factor fragments use a `CopyUniversalOp` with thread-value layouts derived from the tiled MMA, rather than the `ldmatrix`-based path used for A and B:

``` python
# A/B fragments (same as dense)
tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])

# SF fragments (SM120-specific partition helpers)
tCrSFA = sm120_utils.partition_fragment_SFA(sSFA[None, None, 0], thr_mma, tidx)
tCrSFB = sm120_utils.partition_fragment_SFB(sSFB[None, None, 0], thr_mma, tidx)

# A/B: ldmatrix retiling (same as dense)
atom_copy_A = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(...), a_dtype)
smem_tiled_copy_A = cute.make_tiled_copy_A(atom_copy_A, tiled_mma)

# SF: CopyUniversal with SF-specific thread-value layout
atom_copy_SF = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), sf_dtype)
smem_tiled_copy_SFA = cute.make_tiled_copy(
    atom_copy_SF,
    sm120_utils.get_layoutSFA_TV(tiled_mma),
    (cute.size(tiled_mma.permutation_mnk[0]), cute.size(tiled_mma.permutation_mnk[2])),
)
smem_tiled_copy_SFB = cute.make_tiled_copy(
    atom_copy_SF,
    sm120_utils.get_layoutSFB_TV(tiled_mma),
    (cute.size(tiled_mma.permutation_mnk[1]), cute.size(tiled_mma.permutation_mnk[2])),
)
```

**4. Modified main loop** — each k-block loads A, B, SFA, and SFB from SMEM into registers. The `cute.gemm` call passes `[A, SFA]` and `[B, SFB]` as operand lists:

``` python
for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
    # ldmatrix: load A and B from SMEM → RMEM (same as dense)
    cute.copy(smem_tiled_copy_A, tCsA_p[None, None, k_block_next],
              tCrA_copy_view[None, None, k_block_next])
    cute.copy(smem_tiled_copy_B, tCsB_p[None, None, k_block_next],
              tCrB_copy_view[None, None, k_block_next])

    # CopyUniversal: load SFA and SFB from SMEM → RMEM            # NEW
    cute.copy(smem_tiled_copy_SFA,
              cute.filter_zeros(tCsSFA_p)[None, None, k_block_next],
              cute.filter_zeros(tCrSFA_copy_view)[None, None, k_block_next])
    cute.copy(smem_tiled_copy_SFB,
              cute.filter_zeros(tCsSFB_p)[None, None, k_block_next],
              cute.filter_zeros(tCrSFB_copy_view)[None, None, k_block_next])

    # MMA with scale factors passed as [value, scale] pairs
    cute.gemm(
        tiled_mma,
        accumulators,
        [tCrA[None, None, k_block_idx], tCrSFA[None, None, k_block_idx]],  # [A, SFA]
        [tCrB[None, None, k_block_idx], tCrSFB[None, None, k_block_idx]],  # [B, SFB]
        accumulators,
    )
```

``` text
Dense gemm call:
  cute.gemm(tiled_mma, acc, tCrA[k], tCrB[k], acc)

Block-scaled gemm call:
  cute.gemm(tiled_mma, acc, [tCrA[k], tCrSFA[k]], [tCrB[k], tCrSFB[k]], acc)
                            ^^^^^^^^  ^^^^^^^^^    ^^^^^^^^  ^^^^^^^^^
                            value     scale        value     scale
                            (RMEM)    (RMEM)       (RMEM)    (RMEM)
```

Note that `cute.filter_zeros` is applied to the SF copy views because the scale-factor SMEM layouts may contain padding zeros from the TMA tiling. This strips the padded entries so the copy operates only on valid elements.

The epilogue (RMEM → SMEM → GMEM) is identical to a dense kernel.

See also:

- Dense GEMM example (Ampere): `examples/cute/ampere/kernel/dense_gemm/tensorop_gemm.py`
- Block-scaled GEMM example (SM120a): `examples/cute/blackwell_geforce/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent_pingpong.py`
- Block-scaled layout utilities: `cutlass.utils.blockscaled_layout`
- SM120 helper utilities: `cutlass.utils.blackwell_helpers`

---

<!-- source: mma_docs/wgmma_programming.rst -->

## Warpgroup MMA Programming Guide

Hopper (SM90a) introduced the **warpgroup-level MMA** PTX instruction family `wgmma.mma_async.sync.aligned`. A warpgroup (128 threads / 4 warps) cooperates on one asynchronous `D = A * B + C` matrix multiply-accumulate.

Key architectural characteristics:

- **Warpgroup scope:** One MMA is issued collectively by a 128-thread warpgroup rather than by a single warp.
- **Asynchronous issue model:** WGMMA instructions are ordered with `cute.nvgpu.warpgroup.fence()`, `commit_group()`, and `wait_group()`.
- **Descriptor-based operand path:** Operand B is sourced from staged shared memory. Operand A can be sourced either from shared memory descriptors or from registers via `OperandSource`.
- **Register accumulator:** The accumulator lives in RMEM and serves as both the input C and output D of `cute.gemm()`.
- **Architecture-specific operand layouts:** F16/BF16 supports K-major and MN-major dense layouts when A comes from SMEM. FP8 and INT8 variants are K-major only.

The dense DSL op classes currently exposed are `MmaF16BF16Op` (F16/BF16), `MmaF8Op` (FP8 E4M3/E5M2), and `MmaI8Op` (INT8/UINT8); see [Setting up the TiledMMA, MMA Ops](#setting-up-the-tiledmma-mma-ops) for their full constructor parameters, instruction K extents, and major-mode constraints.

This guide outlines the CuTe Python DSL programming model for WGMMA kernels: stage operands in SMEM, build fragment descriptors, launch asynchronous warpgroup MMAs, and stage the RMEM accumulator back to GMEM in the epilogue.

### Global Memory (GMEM) to MMA data flow overview

WGMMA instructions require us to stage B input operands in Shared Memory (SMEM), while A input operands can be sourced from either SMEM or registers (RMEM). SMEM operands are read asynchronously by the hardware via SMEM descriptors. The accumulator is always kept in registers (RMEM) of the warpgroup.

The diagram below traces the full data flow of a WGMMA GEMM kernel, for the most common case where A and B matrices are stored in GMEM and both are staged through SMEM (`a_src=SMEM`), and the output matrix --accumulated in RMEM-- is written back to GMEM through an SMEM staging buffer.

There are 3 parallel tracks where each has 2 sub-tracks. Three parallel tracks are for operands A, B, and C/D, respectively. The two sub-tracks are for copying data between different memory spaces and for MMA execution.

- **Operand A** (and symmetrically **Operand B**):
  - First, we need to create SMEM tensors for A and B matrices: `sA` and `sB`. These tensors are physically allocated tensors that are the destination of TMA copy and the source operands for the WGMMA instructions.
  - Next the **data copy flow** creates the tensor views for copying data from GMEM to SMEM. It starts with `mA` tensor that represents the matrix A in global memory. Then `mA` → `local_tile` → `gA` operation creates the local tile view of A that is the slice of A matrix needed to compute the given CTA's output tile. Then `tma_partition(tma, sA, gA)` produces TMA views `tAsA`, `tAgA`, and the loop copies tiles from GMEM into SMEM via `copy(tma, tAgA[k], tAsA[stage])`.
  - In parallel, the **MMA flow** turns the SMEM tensor into an iterable tensor of SMEM descriptors for the WGMMA instructions. `sA` (the same shared-memory allocation written by TMA) → `partition_A` → `tCsA` (MMA-partitioned SMEM view) → `make_fragment_A` → `tCrA` (SMEM descriptor passed to `cute.gemm()`). Note that the SMEM descriptor is a view created from the SMEM tensor that is interpretable by the WGMMA instructions.
- **Accumulator C/D**:
  - **RMEM accumulator flow** (gemm input/output): `partition_C(gC)` → `tCgC` → `make_rmem_tensor(tCgC.shape)` → `acc`, which serves as both the accumulator input (C) and output (D) of `cute.gemm()` (and the WGMMA instruction).
  - **Output flow** (RMEM → SMEM → GMEM): After the main loop, the accumulator is type-converted and copied from registers to SMEM via `stmatrix` (R2S copy), then stored to global memory via TMA store (S2G copy): `mC` → `local_tile` → `gC` → `partition_C` → `tCgC` on the destination side, and `tRS_rAcc`/`tRS_sD` / `bSG_sD`/`bSG_gD` views drive the two copy stages.

``` text
Operand A Dataflow Path               Operand B Dataflow Path                 Accumulator C/D Dataflow Path
───────────────────────               ───────────────────────                 ─────────────────────────────

mA: (M, K)           [GMEM]             mB: (N, K)            [GMEM]             ┌──── RMEM ──────────┐
│                                       │                                       │ make_rmem_tensor() │
│ local_tile(mA, cta_tiler, coord)      │ local_tile(mB, cta_tiler, coord)      │ acc: accumulator   │
▼                                       ▼                                       └───────┬────────────┘
gA: (BM, BK, k)      [GMEM]             gB: (BN, BK, k)       [GMEM]                     │
│                                       │                                       acc:(MMA,MMA_M,MMA_N) [RMEM]
│  ┌──── SMEM ─────────┐                │  ┌──── SMEM ─────────┐                         │
│  │ sA = alloc(layout)│                │  │ sB = alloc(layout)│                         │        mC: (M, N)     [GMEM]
│  └──┬────────┬───────┘                │  └──┬────────┬───────┘                         │        │
│     │        │                        │     │        │                                 │        │ local_tile
│     │  thr_mma.partition_A(sA)        │     │  thr_mma.partition_B(sB)                 │        ▼
│     │        ▼                        │     │        ▼                                 │        gC: (BM, BN)   [GMEM]
│     │  tCsA:(MMA,MMA_M,               │     │  tCsB:(MMA,MMA_N,                        │        │ partition_C
│     │        MMA_K,PIPE) [SMEM]       │     │        MMA_K,PIPE) [SMEM]                │        ▼
│     │        │                        │     │        │                                 │        tCgC:(MMA,MMA_M,
│     │  make_fragment_A(tCsA)          │     │  make_fragment_B(tCsB)                   │              MMA_N)
│     │        ▼                        │     │        ▼                                 │        [GMEM] (epi dest)
│     │  tCrA:(MMA,MMA_M,               │     │  tCrB:(MMA,MMA_N,                        │        │
│     │        MMA_K,PIPE)              │     │        MMA_K,PIPE)                       │        │
│     │  [SMEM descriptors]             │     │  [SMEM descriptors]                      │        │
│     │        └─────────────┐          │     │        └─────────────┐                   │        │
╰─────┤                      │          ╰─────┤                      │                   │        │
      ▼                      │                ▼                      │                   │        │
tma_partition(tma,           │              tma_partition(tma,       │                   │        │
 sA, gA)                     │               sB, gB)                 │                   │        │
 → tAsA, tAgA                │               → tBsB, tBgB            │                   │        │
      ▼                      │                    ▼                  │                   │        │
  ┌───┴────────────────────┐ │             ┌──────┴─────────────────┐│                   │        │
  │ TMA copy loop (A path):│ │             │ TMA copy loop (B path):││                   │        │
  │ copy(tma, tAgA[k],     │ │             │ copy(tma, tBgB[k],     ││                   │        │
  │      tAsA[stage])      │ │             │      tBsB[stage])      ││                   │        │
┌─▶│ (writes into sA;       │ │         ┌──▶│ (writes into sB;       ││                   │        │
│  │  tCrA reads same sA)   │ │         │   │  tCrB reads same sB)   ││                   │        │
│  │ repeat for next k/stage│ │         │   │ repeat for next k/stage││                   │        │
│  └────────────────────────┘ │         │   └────────────────────────┘│                   │        │
│        │                    │         │         │                   │                   │        │
└────────┘                    ▼         └─────────┘                   ▼                   ▼        │
                             └───────┬───────────────────────────────┴───────────────────┘        │
                                     │                                                            │
                                     ▼                                                            │
                            ┌──────────────────────────────────────────────┐                      │
                            │ GEMM Loop:                                   |                      │
                            │ warpgroup.fence()                            │                      │
                            │ cute.gemm(tiled_mma,                         │                      │
                            │  acc,          D (output, RMEM),             │                      │
                       ┌──▶ │  tCrA[stage],  A (SMEM desc -> sA),          │                      │
                       │    │  tCrB[stage],  B (SMEM desc -> sB),          │                      │
                       │    │  acc)          C (accumulator, RMEM)         │                      │
                       │    │ warpgroup.commit_group()                     │                      │
                       │    │ warpgroup.wait_group(n)                      │                      │
                       │    └──────────────────────────────────────────────┘                      │
                       │       │     │                                                            │
                       └───────┘     |                                                            │
                                     ▼                                                            │
                               Epilogue:                                                          │
                               tRS_rAcc = retile(acc)                                             │
                               tRS_rD   = type_convert(tRS_rAcc)                                  │
                                     │                                                            │
                                     ▼                                                            │
                               R2S: copy(tiled_copy_r2s, tRS_rD, tRS_sD)                          │
                               [RMEM → SMEM via stmatrix]                                         │
                                     │                                                            │
                                     ▼                                                            │
                               sC = alloc(epi_layout) [SMEM]                                      │
                               bSG_sD, bSG_gD = tma_partition(tma_c, sC, gC) ◀───────────────────┘
                                     │
                                     ▼
                               S2G: copy(tma_c, bSG_sD[stage], bSG_gD[coord])
                               [SMEM → GMEM via TMA store]
```

**Naming convention:**

- cta_tiler = (BM, BN, BK) = CTA-wide tiler dimensions
- `mX` = a global tensor, e.g., (M, K) for A
- `gX` = CTA-tiled GMEM slice, e.g., (BM, BK, k) for A
- `sX` = SMEM allocation, e.g., (BM, BK, PIPE) for A
- `tAsA`/`tBsB` = TMA-partitioned SMEM views
- `tAgA`/`tBgB` = TMA-partitioned GMEM views
- `tCsX` = MMA-partitioned SMEM view, e.g., (MMA, MMA_M, MMA_K, PIPE) for A
- `tCrX` = SMEM descriptor fragment, e.g., (MMA, MMA_M, MMA_K, PIPE) for A
- `acc` = RMEM accumulator, (MMA, MMA_M, MMA_N)
- `tCgC` = MMA-partitioned GMEM, (MMA, MMA_M, MMA_N)
- `tRS_rAcc`/`tRS_sD` = epilogue retile views for R2S (RMEM → SMEM) copy
- `bSG_sD`/`bSG_gD` = TMA-partitioned SMEM/GMEM views for epilogue store
- MMA = warpgroup atom thread-value layout; MMA_M/MMA_N/MMA_K = repeat counts (e.g., BM/inst_M), k = outer K-tiles, PIPE = pipeline stages

### Setting up the TiledMMA, MMA Ops

As shown in the data flow overview, CuTe DSL provides many utilities to tile/partition the global memory tensors, and create fragment views of SMEM tensors for MMA instructions.

To utilize these functions, we need to setup the TiledMMA, MMA Ops first.

#### Creating a WGMMA Op

A WGMMA op describes the hardware instruction to use, it has parameters like data types, instruction shape, operand A source (SMEM or RMEM), and operand major modes.

``` python
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import OperandMajorMode
import cutlass.cute.nvgpu.warpgroup as warpgroup

op = warpgroup.MmaF16BF16Op(
    cutlass.Float16,                   # A/B element type
    cutlass.Float32,                   # accumulator type
    (64, 128, 16),                     # instruction shape (M, N, K)
    warpgroup.OperandSource.SMEM,      # A operand from shared memory
    OperandMajorMode.K,                # A is K-major
    OperandMajorMode.K,                # B is K-major
)
```

The key parameters are:

- **Instruction shape** `(M, N, K)`: determines the size of one hardware MMA instruction. WGMMA requires `M = 64` and `8 <= N <= 256` in steps of 8. K is fixed by the op class (16 for F16/BF16, 32 for FP8 and INT8).
- **OperandSource**: `SMEM` reads A from a shared memory descriptor; `RMEM` reads A directly from registers.
- **OperandMajorMode**: `K` for K-major (default), `MN` for transposed layout. F16/BF16 supports both K-major and MN-major for A and B when `a_src=SMEM`; when `a_src=RMEM`, only B can be transposed. FP8 and INT8 are K-major only.

CuTe DSL provides implementation of the following WGMMA ops:

| PTX name | Python class | Constructor parameters |
|----|----|----|
| `wgmma.mma_async.m64n{N}k16.{acc}.f16.f16` / `.bf16.bf16` | `warpgroup.MmaF16BF16Op` | `ab_dtype, acc_dtype, instruction_shape, a_src, a_major_mode, b_major_mode` |
| `wgmma.mma_async.m64n{N}k32.{acc}.{e4m3|e5m2}.{e4m3|e5m2}` | `warpgroup.MmaF8Op` | `a_dtype, b_dtype, acc_dtype, instruction_shape, a_src, a_major_mode, b_major_mode` |
| `wgmma.mma_async.m64n{N}k32.s32.{s8|u8}.{s8|u8}` | `warpgroup.MmaI8Op` | `a_dtype, b_dtype, acc_dtype, instruction_shape, a_src, a_major_mode, b_major_mode` |

WGMMA ops

#### Creating a Tiled MMA

A `TiledMma` tiles the WGMMA atom across the CTA tile. You can pass the op directly or create an explicit atom first.

``` python
# Option 1: directly from op (common shorthand)
tiled_mma = cute.make_tiled_mma(op)

# Option 2: explicit atom creation
atom = cute.make_mma_atom(op)
tiled_mma = cute.make_tiled_mma(atom)
```

#### Spatial tiling with a repeat count

A repeat tuple `(M_rep, N_rep, K_rep)` replicates the WGMMA atom across the M, N, and K dimensions, producing a larger tiled MMA that covers a bigger CTA tile with a single `cute.gemm` call. Each entry in the repeat tuple corresponds to one **warpgroup** (128 threads / 4 warps), so `(2, 1, 1)` uses two warpgroups — the standard configuration for large Hopper tiles:

``` python
atom = cute.make_mma_atom(op)   # op shape: (64, 128, 16)
tiled_mma = cute.make_tiled_mma(
    atom,
    atom_layout_mnk=(2, 1, 1),  # 2 warpgroups in M
)
```

``` text
WGMMA Atom                             make_tiled_mma(atom, (2, 1, 1))
+---------------+                      +----------------+
|               |                      |                | ^
|   64 x 128    |                      |  Atom (0,0,0)  | |
|     x 16      |    --(2,1,1)-->      |   64 x 128     | | 2 x M_atom
|               |       repeat         |     x 16       | |  = 128
|               |                      | [Warpgroup 0]  | |
+---------------+                      +----------------+ |
                                       |                | |
                                       |  Atom (1,0,0)  | |
                                       |   64 x 128     | |
                                       |     x 16       | |
                                       | [Warpgroup 1]  | v
                                       +----------------+
                                       <-- N_atom = 128 -->
                                       K unchanged = 16
```

The Hopper dense GEMM examples (`examples/cute/hopper/kernel/dense_gemm/dense_gemm.py`) use this pattern. The helper `sm90_utils.make_trivial_tiled_mma(...)` selects the repeat count automatically:

- `atom_layout_mnk = (2, 1, 1)` when both `tile_M > 64` and `tile_N > 128` (two warpgroups reduce register pressure).
- `atom_layout_mnk = (1, 1, 1)` otherwise (a single warpgroup suffices).

``` python
import cutlass.utils.hopper_helpers as sm90_utils

tiled_mma = sm90_utils.make_trivial_tiled_mma(
    a_dtype,
    b_dtype,
    a_major_mode,
    b_major_mode,
    acc_dtype,
    atom_layout_mnk=(2, 1, 1),
    tiler_mn=(64, 128),         # atom instruction shape (M, N)
)
```

#### Custom tile permutation with `permutation_mnk`

`make_tiled_mma` also accepts an optional `permutation_mnk` argument that controls how the tiled atom footprint is laid out across M, N, and K. At a high level:

- `atom_layout_mnk` tells CuTe how many atoms (warpgroups) to replicate.
- `permutation_mnk` tells CuTe how the final tiled footprint is ordered.

`permutation_mnk` is a tuple of layouts or integers that represent the tile size and ordering of values along each dimension. When a mode's permutation size is larger than the atom layout's natural coverage (`atom_layout x inst_shape`), each warpgroup receives additional values to fill the extended region — the warpgroup count stays the same, but each warpgroup holds more data.

``` python
atom = cute.make_mma_atom(op)   # op shape: (64, 128, 16)
tiled_mma = cute.make_tiled_mma(
    atom,
    atom_layout_mnk=(2, 1, 1),
    permutation_mnk=(128, 256, 16),   # extend N from 128 to 256
)
```

``` text
Without permutation — natural atom coverage (M = 128, N = 128):

C tile (M=128, N=128)
+----------------+
|                | ^
| [Warpgroup 0]  | |
|    64 x 128    | | 2 x inst_M
|                | |  = 128
+----------------+ |
|                | |
| [Warpgroup 1]  | |
|    64 x 128    | |
|                | v
+----------------+
<--- N = 128 --->
(each warpgroup owns one (64, 128) atom)

With permutation_mnk = (128, 256, 16) — N extended to 256:

C tile (M=128, N=256)
+----------------+----------------+
|                |                | ^    N = 128 → 256:
| [Warpgroup 0]  | [Warpgroup 0]  | |    atom pattern repeats
|    64 x 128    |    64 x 128    | |    along N. Each warpgroup
|                |                | |    now holds 2x the values
+----------------+----------------+ |    along N (same threads,
|                |                | |    more data).
| [Warpgroup 1]  | [Warpgroup 1]  | |
|    64 x 128    |    64 x 128    | |
|                |                | v
+----------------+----------------+
<------------ N = 256 ------------>
| atom coverage  |  value repeat  |
```

**Why WGMMA typically does not need permutation_mnk:** The WGMMA instruction already has a large N dimension (64, 128, or 256), so the natural atom coverage is wide enough that no permutation is needed to align with SMEM swizzle widths. The Hopper dense GEMM examples (`dense_gemm.py`, `dense_gemm_persistent.py`) use `atom_layout_mnk` alone without `permutation_mnk`.

When `permutation_mnk` is not provided (default), the tile ordering is sequential and no permutation is applied.

### Partitioning Tensors

Before computing, partition the CTA-tiled tensors according to the tiled MMA layout. WGMMA partitioning is **warpgroup-oriented**: each warpgroup (128 threads / 4 warps) receives its own slice of the CTA tile, sized to match the SMEM descriptors and register accumulators that the WGMMA instruction expects.

**2-warpgroup example**

`GEMM (M, N, K) = (512, 768, 256)`, `tile_shape_mnk = (128, 256, 64)`, F16 WGMMA atom = (64, 256, 16), `atom_layout_mnk = (2, 1, 1)`, `num_stages = 4`, 2 warpgroups = 256 threads.

Global matrices:

``` text
mA: (M, K) = (512, 256)       mB: (N, K) = (768, 256)       mC: (M, N) = (512, 768)

     K=256                          K=256                       N=768
   |<--------->|               |<--------->|             |<----------------->|
   +-----------+               +-----------+             +---+---+---+-------+
   |           | ^             |           | ^           |   |   |   |       | ^
   |    mA     | | M=512       |    mB     | | N=768     |   |   |   |       | | M=512
   |           | v             |           | v           |   |   |   |       | v
   +-----------+               +-----------+             +---+---+---+-------+
```

Tiling with `tile_shape_mnk = (BM, BN, BK) = (128, 256, 64)` gives M/BM = 4 tiles, N/BN = 3 tiles, K/BK = 4 tiles:

``` text
mA tiled into (M/BM x K/BK)    mB tiled into (N/BN x K/BK)   mC tiled into (M/BM x N/BN)
= (4 x 4) blocks               = (3 x 4) blocks              = (4 x 3) blocks

  BK=64  x4                       BK=64  x4                      BN=256  x3
|<--->|                         |<--->|                        |<------>|
+-----+-----+-----+-----+       +-----+-----+-----+-----+      +--------+--------+--------+
|     |     |     |     | ^     |     |     |     |     | ^    | (0,0)  | (0,1)  | (0,2)  | ^
|     |     |     |     | |128  |     |     |     |     | |256 |        |        |        | |128
+-----+-----+-----+-----+ v     +-----+-----+-----+-----+ v    +--------+--------+--------+ v
|     |     |     |     | ^     |     |     |     |     | ^    | (1,0)  | (1,1)  | (1,2)  | ^
|     |     |     |     | |128  |     |     |     |     | |256 |        |        |        | |128
+-----+-----+-----+-----+ v     +-----+-----+-----+-----+ v    +--------+--------+--------+ v
|     |     |     |     |       |     |     |     |     |      | (2,0)  | (2,1)  | (2,2)  |
+-----+-----+-----+-----+       +-----+-----+-----+-----+      +--------+--------+--------+
|     |     |     |     |                                      | (3,0)  | (3,1)  | (3,2)  |
+-----+-----+-----+-----+                                      +--------+--------+--------+
```

Each CTA picks one (M-tile, N-tile) coordinate. For example, CTA at `tile_coord = (1, 0, :)`.

After `local_tile` — one CTA's tile (`k = K/BK = 256/64 = 4`):

``` text
gA: (BM, BK, k) = (128, 64, 4)   gB: (BN, BK, k) = (256, 64, 4)   gC: (BM, BN) = (128, 256)

     BK=64                              BK=64                       BN=256
   |<----->|                         |<----->|                 |<--------->|
   +-------+--                       +-------+--               +-----------+
   |       |..                       |       |..               |           | ^
BM=  |  gA   | k=4                BN=  |  gB   | k=4         BM= |    gC     | | 128
128 |       |                    256  |       |             128 |           | v
   +-------+                         +-------+                 +-----------+
```

SMEM tensors `sA` and `sB` include a pipeline staging dimension:

``` text
sA: (BM, BK, PIPE) = (128, 64, 4)       sB: (BN, BK, PIPE) = (256, 64, 4)
```

`get_slice(warp_group_thread_layout(warp_group_idx))` — each warpgroup receives its slice of the tiled MMA footprint. With `atom_layout_mnk = (2, 1, 1)` and inst shape `(64, 256, 16)`, the tiled MMA covers `(2x64, 1x256, 16) = (128, 256, 16)` which exactly matches the CTA tile in M and N. Each warpgroup owns one 64-row slice of M:

``` text
sA (one pipeline stage, BM=128, BK=64):

Warpgroup 0's slice               Warpgroup 1's slice
inst_K  inst_K  inst_K  inst_K
 =16     =16     =16     =16
|<--->|<--->|<--->|<--->|         |<--->|<--->|<--->|<--->|
+-----+-----+-----+-----+  ^     +-----+-----+-----+-----+  ^
|  0  |  1  |  2  |  3  |  |64   |  0  |  1  |  2  |  3  |  |64
+-----+-----+-----+-----+  v     +-----+-----+-----+-----+  v
|<-- MMA_K = BK/inst_K = 4 -->|  |<-- MMA_K = 4 ---------->|
MMA_M = 64/64 = 1                MMA_M = 64/64 = 1

gC (BM=128, BN=256):

+---------------------------+  ^
|  Warpgroup 0: 64 x 256    |  | 64
|                           |  |
+---------------------------+  v
|  Warpgroup 1: 64 x 256    |  ^
|                           |  | 64
+---------------------------+  v
<--------- N = 256 -------->
MMA_M = 64/64 = 1, MMA_N = 256/256 = 1
```

After partition (per warpgroup):

- `tCsA: (MMA, MMA_M, MMA_K, PIPE) = (MMA, 1, 4, 4)` — MMA_M = BM / (atom_M x inst_M) = 128 / (2x64) = 1, MMA_K = BK / inst_K = 64 / 16 = 4
- `tCsB: (MMA, MMA_N, MMA_K, PIPE) = (MMA, 1, 4, 4)` — MMA_N = BN / (atom_N x inst_N) = 256 / (1x256) = 1, MMA_K = 4
- `tCgC: (MMA, MMA_M, MMA_N) = (MMA, 1, 1)` — MMA_M = 1, MMA_N = 1

The first mode `MMA` contains the atom's **thread x value** layout — it encodes which registers within a warpgroup hold which matrix elements. The remaining modes are repeat counts that tile the atom across the full CTA tile.

> [!NOTE]
> Because the WGMMA instruction shape is large (64 x {64..256}), the tiled MMA footprint typically covers the entire CTA tile in M and N with just one or two warpgroups. This means MMA_M and MMA_N are often 1. The MMA_K dimension is where the repeat count is non-trivial (BK / inst_K iterations per pipeline stage).

**1-warpgroup example (contrast)**

For a smaller tile `(128, 128, 64)` with `atom_layout_mnk = (1, 1, 1)`, inst shape `(64, 128, 16)`, and `num_stages = 4`, the tiled MMA covers only `(64, 128, 16)`. Now a single warpgroup must iterate over two atom-blocks along M:

- `tCsA: (MMA, MMA_M, MMA_K, PIPE) = (MMA, 2, 4, 4)` — MMA_M = 128 / (1x64) = 2
- `tCsB: (MMA, MMA_N, MMA_K, PIPE) = (MMA, 1, 4, 4)` — MMA_N = 128 / (1x128) = 1
- `tCgC: (MMA, MMA_M, MMA_N) = (MMA, 2, 1)`

``` python
# Based on examples/cute/hopper/kernel/dense_gemm/dense_gemm.py
@cute.kernel
def kernel(tiled_mma: cute.TiledMma, ...):
    tidx, _, _ = cute.arch.thread_idx()

    # CTA-tiled global tensors
    gA_mkl = cute.local_tile(
        mA_mkl, tile_shape_mnk, tile_coord_mnkl, proj=(1, None, 1)
    )
    gB_nkl = cute.local_tile(
        mB_nkl, tile_shape_mnk, tile_coord_mnkl, proj=(None, 1, 1)
    )
    gC_mnl = cute.local_tile(
        mC_mnl, tile_shape_mnk, tile_coord_mnkl, proj=(1, 1, None)
    )

    # Warpgroup-oriented slicing (128 threads per warpgroup)
    warp_group_idx = cute.arch.make_warp_uniform(
        tidx // num_threads_per_warp_group     # 128
    )
    warp_group_thread_layout = cute.make_layout(
        mma_warp_groups,                        # e.g. 2
        stride=num_threads_per_warp_group,      # 128
    )
    thr_mma = tiled_mma.get_slice(
        warp_group_thread_layout(warp_group_idx)
    )

    # Partition C from global
    tCgC = thr_mma.partition_C(gC_mnl)  # (MMA, MMA_M, MMA_N)

    # Partition A/B from staged SMEM
    tCsA = thr_mma.partition_A(sA)      # (MMA, MMA_M, MMA_K, PIPE)
    tCsB = thr_mma.partition_B(sB)      # (MMA, MMA_N, MMA_K, PIPE)
```

### Pre and Post-Conditions for Partitioning

- The inputs of `partition_A`, `partition_B`, and `partition_C` should be at least rank-2 tensors.
- The output layout is constrained by the selected MMA atom:
  - For A, the output has layout `(MMA, MMA_M, MMA_K, ...)`.
  - For B, the output has layout `(MMA, MMA_N, MMA_K, ...)`.
  - For C, the output has layout `(MMA, MMA_M, MMA_N, ...)`.
- Partitioning reasons about layout, not memory space or element type. When `a_src=OperandSource.RMEM`, the same tiled MMA shape still determines the logical A footprint, but A is materialized as a register fragment rather than a shared-memory descriptor.

### Making Fragments

Fragments are the tensors that the WGMMA instruction operates on. For dense WGMMA:

- **Fragment A**: an SMEM descriptor when `a_src=OperandSource.SMEM`, or an RMEM register fragment when `a_src=OperandSource.RMEM`.
- **Fragment B**: an SMEM descriptor pointing into staged shared memory buffers.
- **Fragment C (accumulator)**: an RMEM tensor that serves as both the input C and output D of `cute.gemm()`.

WGMMA fragments for A and B are **SMEM descriptors** — the hardware reads directly from shared memory. There is no explicit SMEM → RMEM copy step for operands A and B. The accumulator, however, still lives in per-thread registers (RMEM).

#### Creating fragment descriptors and accumulator fragments

Fragment creation has two parts:

**1. A and B fragment descriptors**

`make_fragment_A` and `make_fragment_B` take the MMA-partitioned SMEM views (`tCsA` / `tCsB`) and produce descriptor tensors that the WGMMA instruction consumes. Each descriptor points to one tile within a pipeline stage in shared memory.

``` python
# MMA-partitioned SMEM views (see "Partitioning Tensors")
tCsA = thr_mma.partition_A(sA)   # (MMA, MMA_M, MMA_K, PIPE)
tCsB = thr_mma.partition_B(sB)   # (MMA, MMA_N, MMA_K, PIPE)

# SMEM descriptor fragments consumed by cute.gemm()
tCrA = tiled_mma.make_fragment_A(tCsA)   # (MMA, MMA_M, MMA_K, PIPE)
tCrB = tiled_mma.make_fragment_B(tCsB)   # (MMA, MMA_N, MMA_K, PIPE)
```

Continuing the 2-warpgroup example from [Partitioning Tensors](#partitioning-tensors) (F16 atom = (64, 256, 16), `tile_shape_mnk = (128, 256, 64)`, `atom_layout_mnk = (2, 1, 1)`, `num_stages = 4`):

``` text
tCsA: (MMA, MMA_M=1, MMA_K=4, PIPE=4)
tCsB: (MMA, MMA_N=1, MMA_K=4, PIPE=4)

make_fragment_A(tCsA) -> tCrA: (MMA, 1, 4, 4)
make_fragment_B(tCsB) -> tCrB: (MMA, 1, 4, 4)

Each element of tCrA/tCrB is an SMEM descriptor — one per
(MMA_K, PIPE) pair. The hardware reads SMEM directly via the
descriptor; no explicit SMEM -> RMEM load is needed.

tCrA per warpgroup (4 pipeline stages, 4 K-blocks each):

              |<-- MMA_K = BK/inst_K = 4 -->|
  stage 0:    +------+------+------+------+
              | k=0  | k=1  | k=2  | k=3  |  inst_M=64 (MMA_M=1)
              +------+------+------+------+
  stage 1:    +------+------+------+------+
              | k=0  | k=1  | k=2  | k=3  |  inst_M=64
              +------+------+------+------+
  stage 2:    +------+------+------+------+
              | k=0  | k=1  | k=2  | k=3  |  inst_M=64
              +------+------+------+------+
  stage 3:    +------+------+------+------+
              | k=0  | k=1  | k=2  | k=3  |  inst_M=64
              +------+------+------+------+

Similarly for tCrB with shape (MMA, MMA_N=1, MMA_K=4, PIPE=4).
```

> [!NOTE]
> WGMMA fragments for A and B are SMEM descriptors — the hardware reads SMEM directly, so there is no `ldmatrix` retiling step required before `cute.gemm()`.

**When A comes from registers (\`\`OperandSource.RMEM\`\`)**

In fused kernels, the output of one MMA can become the A operand of the next. The second `TiledMma` is created with `a_src=OperandSource.RMEM`, and `make_fragment_A` is **not** used. Instead:

1.  The accumulator's C layout `(MMA, MMA_M, MMA_N)` is converted to the A layout `(MMA, MMA_M, MMA_K)` expected by the second `TiledMma`.
2.  The accumulator values are type-converted and stored into an RMEM tensor with the A layout.
3.  The resulting RMEM tensor is passed directly to `cute.gemm()` as the A operand — no SMEM descriptor is involved.

See the Hopper FMHA example (`examples/cute/hopper/kernel/attention/fmha.py`) for the complete pattern.

**2. C fragment (accumulator)**

The accumulator lives in per-thread registers (RMEM). Its shape is derived from the partitioned C layout. The accumulator starts at zero before the K loop and is updated in-place by each `cute.gemm()` call.

``` python
# Partition C from global (see "Partitioning Tensors")
tCgC = thr_mma.partition_C(gC_mnl)   # (MMA, MMA_M, MMA_N)

# Allocate RMEM accumulator with the same shape
acc_shape = tCgC.shape
acc = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
acc.fill(0.0)
```

For the same running example:

``` text
tCgC: (MMA, MMA_M=1, MMA_N=1)

make_rmem_tensor(tCgC.shape, Float32) -> acc: (MMA, 1, 1)

The accumulator stays in RMEM for the entire main loop.
cute.gemm() reads A/B from SMEM descriptors and accumulates into acc.

+-----------------------------------+
|  acc: (MMA, 1, 1) in RMEM         |
|  64 x 256 elements per warpgroup  |
|  Float32                          |
+-----------------------------------+
```

### Creating SMEM layouts for A and B

The SMEM layouts define how A and B tiles are staged in shared memory, including swizzling for bank-conflict-free descriptor access. The helper functions in `cutlass.utils.hopper_helpers` handle the details.

**Host side** (`@cute.jit`):

``` python
import cutlass.utils.hopper_helpers as sm90_utils

# Create SMEM layouts (includes swizzle + staging)
a_smem_layout = sm90_utils.make_smem_layout_a(
    a_layout,          # LayoutEnum — row-major or col-major
    tile_shape_mnk,    # CTA tile (M, N, K)
    a_dtype,           # element type (e.g. Float16)
    num_stages,        # pipeline depth
)
b_smem_layout = sm90_utils.make_smem_layout_b(
    b_layout,
    tile_shape_mnk,
    b_dtype,
    num_stages,
)
epi_smem_layout = sm90_utils.make_smem_layout_epi(
    c_dtype,
    c_layout,
    epi_tile,
    epi_stage,
)
```

`make_smem_layout_a` and `make_smem_layout_b` are convenience helpers that build a complete, staged SMEM layout in four steps:

1.  **Extract the operand tile shape.** For A the `(M, K)` portion of `tile_shape_mnk` is kept via `cute.slice_`; for B the `(N, K)` portion.

2.  **Determine the major mode.** The major mode (K-major or MN-major) is read from the layout enum (`a_layout.is_k_major_a()`). The major-mode dimension size is used for swizzle selection.

3.  **Select and materialise the swizzle atom.** A heuristic (`get_smem_layout_atom`) picks the widest swizzle whose contiguous size (in bits) evenly divides the major-mode dimension:

    | Swizzle    | Contiguous bits |
    |------------|-----------------|
    | SW128      | 1024 (128 B)    |
    | SW64       | 512 (64 B)      |
    | SW32       | 256 (32 B)      |
    | Interleave | 128 (16 B)      |

    `make_smem_layout_atom` then combines the chosen swizzle with a compact outer layout into a `ComposedLayout(swizzle, outer)`.

4.  **Tile to the operand shape and append the staging dimension.** `cute.tile_to_shape` broadcasts the atom to the full `(M_or_N, K)` shape with `num_stages` appended. The `order` argument controls which dimension is contiguous: `(0, 1, 2)` for K-major (K innermost), `(1, 0, 2)` for MN-major (MN innermost).

For the running F16 example (`tile_shape_mnk = (128, 256, 64)`, `num_stages = 4`, K-major A, K-major B):

``` text
A operand (K-major, tile = (M=128, K=64)):
  major_mode_size = 64
  64 * 16 bits = 1024 bits → SW128
  atom = make_smem_layout_atom(K_SW128, Float16)
  tile_to_shape(atom, (128, 64, 4), order=(0,1,2))
  -> a_smem_layout: ComposedLayout with shape (128, 64, 4)

B operand (K-major, tile = (N=256, K=64)):
  major_mode_size = 64
  64 * 16 bits = 1024 bits → SW128
  atom = make_smem_layout_atom(K_SW128, Float16)
  tile_to_shape(atom, (256, 64, 4), order=(0,1,2))
  -> b_smem_layout: ComposedLayout with shape (256, 64, 4)
```

**Kernel side** (`@cute.kernel`):

The layout and swizzle are passed to shared-memory allocation. The result is a `ComposedLayout` whose `.outer` is the logical layout and `.inner` is the swizzle:

``` python
# Based on examples/cute/hopper/kernel/dense_gemm/dense_gemm.py
sA = storage.sA.get_tensor(
    a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
)
sB = storage.sB.get_tensor(
    b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
)
```

After allocation:

- `sA` has shape `(BM, BK, PIPE) = (128, 64, 4)`.
- `sB` has shape `(BN, BK, PIPE) = (256, 64, 4)`.

These are the staged SMEM tensors consumed by `partition_A` / `partition_B` and `make_fragment_A` / `make_fragment_B` (see [Making Fragments](#making-fragments)).

> [!NOTE]
> If you need finer control, you can build layout atoms directly with `cute.nvgpu.warpgroup.make_smem_layout_atom(...)` and compose the final SMEM layout manually via `cute.tile_to_shape`.

### Executing the GEMM (Main Loop)

The main loop iterates over K-tiles. The WGMMA-specific part of each iteration is the **fence / gemm / commit / wait** sequence:

``` python
acc.fill(0.0)
tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

for k_tile in cutlass.range(k_pipe_mmas, k_tile_cnt, 1, unroll=1):
    # ... wait for TMA load (pipeline details in dense_gemm.py) ...

    cute.nvgpu.warpgroup.fence()
    tile_crd = (None, None, None, consumer_read.index)
    cute.gemm(tiled_mma, acc, tCrA[tile_crd], tCrB[tile_crd], acc)
    cute.nvgpu.warpgroup.commit_group()
    cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)

    # ... release buffer & advance pipeline (see dense_gemm.py) ...

cute.nvgpu.warpgroup.wait_group(0)
```

Key points:

- `fence()` orders prior SMEM writes before WGMMA issue.
- `commit_group()` publishes queued WGMMA instructions as a group.
- `wait_group(n)` waits until at most `n` groups remain in flight. `wait_group(0)` after the loop drains all work before the epilogue.
- `Field.ACCUMULATE` — `True` accumulates (`D += A*B`), `False` overwrites (`D = A*B`). The dense GEMM sets `True` and zero-fills `acc` so the first iteration computes `0 + A*B`.

### Complete Workflow

Putting it all together, a typical Hopper WGMMA GEMM has this structure. The MMA-relevant steps are highlighted; see `dense_gemm.py` for the full kernel including TMA, pipeline, and epilogue details.

``` python
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import OperandMajorMode
import cutlass.cute.nvgpu.warpgroup as warpgroup
import cutlass.utils.hopper_helpers as sm90_utils

# --- Host side (@cute.jit) ---

# 1. MMA op + tiled MMA
op = warpgroup.MmaF16BF16Op(
    cutlass.Float16, cutlass.Float32, (64, 128, 16),
    warpgroup.OperandSource.SMEM, OperandMajorMode.K, OperandMajorMode.K,
)
tiled_mma = cute.make_tiled_mma(op)

# 2. SMEM layouts
a_smem_layout = sm90_utils.make_smem_layout_a(a_layout, tile_shape_mnk, a_dtype, num_stages)
b_smem_layout = sm90_utils.make_smem_layout_b(b_layout, tile_shape_mnk, b_dtype, num_stages)

# 3. TMA copy atoms + kernel launch (see dense_gemm.py)
```

``` python
# --- Kernel side (@cute.kernel) ---

# 4. Allocate SMEM
smem = cutlass.utils.SmemAllocator()
storage = smem.allocate(SharedStorage)
sA = storage.sA.get_tensor(
    a_smem_layout.outer, swizzle=a_smem_layout.inner)   # (BM, BK, PIPE)
sB = storage.sB.get_tensor(
    b_smem_layout.outer, swizzle=b_smem_layout.inner)   # (BN, BK, PIPE)

# 5. CTA-tiled global tensors
gA_mkl = cute.local_tile(mA_mkl, tile_shape_mnk, tile_coord, proj=(1, None, 1))
gB_nkl = cute.local_tile(mB_nkl, tile_shape_mnk, tile_coord, proj=(None, 1, 1))
gC_mnl = cute.local_tile(mC_mnl, tile_shape_mnk, tile_coord, proj=(1, 1, None))

# 6. Warpgroup slice, partition & make fragments
warp_group_idx = cute.arch.make_warp_uniform(tidx // num_threads_per_warp_group)
warp_group_thread_layout = cute.make_layout(mma_warp_groups, stride=num_threads_per_warp_group)
thr_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))

tCsA = thr_mma.partition_A(sA)             # (MMA, MMA_M, MMA_K, PIPE)
tCsB = thr_mma.partition_B(sB)             # (MMA, MMA_N, MMA_K, PIPE)
tCrA = tiled_mma.make_fragment_A(tCsA)     # SMEM descriptor
tCrB = tiled_mma.make_fragment_B(tCsB)     # SMEM descriptor
tCgC = thr_mma.partition_C(gC_mnl)         # (MMA, MMA_M, MMA_N)
acc  = cute.make_rmem_tensor(tCgC.shape, acc_dtype)

# 7. TMA pipeline setup + prefetch (see dense_gemm.py)

# 8. Main loop — fence / gemm / commit / wait
acc.fill(0.0)
tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

for k_tile in cutlass.range(k_pipe_mmas, k_tile_cnt, 1, unroll=1):
    # ... wait for TMA load ...
    cute.nvgpu.warpgroup.fence()
    tile_crd = (None, None, None, consumer_read.index)
    cute.gemm(tiled_mma, acc, tCrA[tile_crd], tCrB[tile_crd], acc)
    cute.nvgpu.warpgroup.commit_group()
    cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)
    # ... release buffer, advance pipeline ...

cute.nvgpu.warpgroup.wait_group(0)

# 9. Epilogue: RMEM → SMEM (stmatrix) → GMEM (TMA store)
# ... (see dense_gemm.py)
```

See also:

- Dense GEMM example: `examples/cute/hopper/kernel/dense_gemm/dense_gemm.py`
- Persistent GEMM example: `examples/cute/hopper/kernel/dense_gemm/dense_gemm_persistent.py`
- FMHA example (RMEM A path): `examples/cute/hopper/kernel/attention/fmha.py`
- Helper utilities: `cutlass.utils.hopper_helpers`

---

<!-- source: mma_docs/tcgen05_programming.rst -->

## tcgen05 MMA Programming Guide

Blackwell (SM100) introduces the **tcgen05** family of PTX instructions — the 5th-generation Tensor Core MMA (matrix multiply-accumulate) operations. They compute `D = A * B + C` with 2x–4x the throughput of Hopper's WGMMA instructions, depending on data type.

Key architectural characteristics:

- **Tensor Memory (TMEM):** A new on-chip memory dedicated to the accumulator (and, optionally, operand A). tcgen05 MMA reads and writes the accumulator in TMEM directly, freeing the register file for other work.
- **Single-thread launch:** Only one thread issues the MMA instruction.
- **CTA-pair cooperation:** Two adjacent CTAs can jointly execute a single MMA, doubling the tile size without extra synchronization logic.

This guide shows how to program these operations through the CuTe Python DSL, using SMEM for operands A and B, and TMEM for the accumulator.

### Global Memory (GMEM) to MMA data flow overview

Tcgen05 MMA instructions requires us to stage A input operands in Shared Memory (SMEM) or Tensor Memory (TMEM), and B input operands in SMEM. The accumulator is always stored in TMEM.

The diagram below traces the full data flow of a tcgen05 GEMM kernel, for the most common case where A and B matrices are stored in GMEM, and the output matrix --read from TMEM-- is written to GMEM.

There are 3 parallel tracks where each has 2 sub-tracks. Three parallel tracks are for operands A, B, and C/D, respectively. The two sub-tracks are for copying data between different memory spaces and for MMA execution.

- **Operand A** (and symmetrically **Operand B**):
  - First, we need to create SMEM tensors for A and B matrices: `sA` and `sB`. These tensors are physically allocated tensors that are the destination of copy and the source operands for the MMA instructions.
  - Next the **data copy flow** creates the tensor views for copying data from GMEM to SMEM. It starts with `mA` tensor that represents the matrix A in global memory. `mA` → `local_tile` → `gA` operation creates the local tile view of A that is the slice of A matrix needed to compute the given MMA's output tile partitioning. `gA` → `partition_A` → `tCgA` partitions the full MMA sized tile into smaller tiles which are needed to copy the correct portion of A/B matrix to SMEM by individual CTAs cooperating for the MMA (1CTA vs 2CTA pair MMA cases). Then `tma_partition` produces TMA views `tAsA`, `tAgA`, and the loop copies tiles from GMEM into SMEM via `copy(tma, tAgA[k], tAsA[stage])`.
  - In parallel, the **MMA flow** turns the SMEM tensors into iterable tensors of SMEM descriptors for MMA instructions. `sA` (the same shared-memory allocation written by TMA) → `make_fragment_A` → `tCrA` (they are passed to `cute.gemm()`). Note that the SMEM descriptors are views created from the SMEM tensor that is interpretable by the MMA instructions.
- **Accumulator C/D**:
  - **TMEM accumulator flow** (gemm input/output): `make_fragment_C(MMA_partition_shape_C)` → `tCtAcc`, which serves as the accumulator input/output of `cute.gemm()` (and MMA instruction).
  - **Output flow** (GMEM destination): The LDTM loads results into registers and a final store writes them to global memory. `mC` → `local_tile` → `gC` → `partition_C` → `tCgC`. This path creates the tensor views that will be stored to GMEM.

``` text
Operand A Dataflow Path               Operand B Dataflow Path                 Accumulator C/D Dataflow Path
───────────────────────               ───────────────────────                 ─────────────────────────────

mA: (M, K)           [GMEM]             mB: (N, K)            [GMEM]             ┌──── TMEM ──────────┐
│                                       │                                       │ partition_shape_C()│
│ local_tile(mA, mma_tiler, coord)      │ local_tile(mB, mma_tiler, coord)      │ make_fragment_C()  │
▼                                       ▼                                       │ bind to tmem_ptr   │
gA: (BM, BK, k)      [GMEM]             gB: (BN, BK, k)       [GMEM]             └───────┬────────────┘
│                                       │                                               │
│ thr_mma.partition_A(gA)               │ thr_mma.partition_B(gB)               tCtAcc:(MMA,MMA_M,MMA_N) [TMEM]
▼                                       ▼                                               │
tCgA:(MMA,MMA_M,      [GMEM]            tCgB:(MMA,MMA_N,      [GMEM]                     │
      MMA_K,k)                                MMA_K,k)                                  │
│                                       │                                               │        mC: (M, N)     [GMEM]
│  ┌──── SMEM ─────────┐                │  ┌──── SMEM ─────────┐                        │        │
│  │ sA = alloc(layout)│                │  │ sB = alloc(layout)│                        │        │ local_tile
│  └──┬────────┬───────┘                │  └──┬────────┬───────┘                        │        ▼
│     │        │                        │     │        │                                │        gC: (BM, BN)   [GMEM]
│     │   make_fragment_A(sA)           │     │   make_fragment_B(sB)                   │        │ partition_C
│     │        │                        │     │        │                                │        ▼
│     │        ▼                        │     │        ▼                                │        tCgC:(MMA,MMA_M,
│     │  tCrA:(MMA,MMA_M,               │     │  tCrB:(MMA,MMA_N,                       │              MMA_N)
│     │        MMA_K,STAGE)             │     │        MMA_K,STAGE)                     │        [GMEM] (epi dest)
│     │  [SMEM descriptors]             │     │  [SMEM descriptors]                     │        │
│     │        └─────────────┐          │     │        └─────────────┐                  │        │
╰─────┤                      │          ╰─────┤                      │                  │        │
      ▼                      │                ▼                      │                  │        │
tma_partition(tma,           │              tma_partition(tma,       │                  │        │
 sA, tCgA)                   │               sB, tCgB)               │                  │        │
 → tAsA, tAgA                │               → tBsB, tBgB            │                  │        │
      ▼                      │                    ▼                  │                  │        │
  ┌───┴────────────────────┐ │             ┌──────┴─────────────────┐│                  │        │
  │ TMA copy loop (A path):│ │             │ TMA copy loop (B path):││                  │        │
  │ copy(tma, tAgA[k],     │ │             │ copy(tma, tBgB[k],     ││                  │        │
  │      tAsA[stage])      │ │             │      tBsB[stage])      ││                  │        │
┌─▶│ (writes into sA;       │ │         ┌──▶│ (writes into sB;       ││                  │        │
│  │  tCrA reads same sA)   │ │         │   │  tCrB reads same sB)   ││                  │        │
│  │ repeat for next k/stage│ │         │   │ repeat for next k/stage││                  │        │
│  └────────────────────────┘ │         │   └────────────────────────┘│                  │        │
│        │                    │         │         │                   │                  │        │
└────────┘                    ▼         └─────────┘                   ▼                  ▼        │
                             └───────┬───────────────────────────────┴──────────────────┘        │
                                     │                                                           │
                                     ▼                                                           │
                            ┌──────────────────────────────────────────────┐                     │
                            │ GEMM Loop:                                   |                     │
                            | cute.gemm(tiled_mma,                         │                     │
                            │  tCtAcc,       D (output),                   │                     │
                       ┌──▶ │  tCrA[stage],  A (SMEM desc -> sA),          │                     │
                       │    │  tCrB[stage],  B (SMEM desc -> sB),          │                     │
                       │    │  tCtAcc)       C (accumulator input)         │                     │
                       │    └──────────────────────────────────────────────┘                     │
                       │       │     │                                                           │
                       └───────┘     |                                                           │
                                     ▼                                                           │
                               Epilogue:                                                         │
                               t2r = make_tmem_copy(LdOp, tCtAcc)                                │
                               tTR_tAcc = t2r.partition_S(tCtAcc)                                │
                               tTR_gC   = t2r.partition_D(tCgC) ◀────────────────────────────────┘
                               tTR_rAcc = make_rmem_tensor(...)
                                     │
                                     ▼
                               LDTM: copy(t2r, tTR_tAcc, tTR_rAcc)
                               [TMEM → RMEM]
                                     │
                                     ▼
                               Store: copy(atom, tTR_rAcc, tTR_gC)
                               [RMEM → GMEM]
```

**Naming convention:**

- `mma_tiler_mnk` = `(BM, BN, BK)` — per-CTA (or per-CTA-pair) MMA tile
- `mX` = a global tensor, such as `mA`, `mB`, `mC`
- `gX` = MMA-tiler tiled GMEM slice, e.g. `(BM, BK, k)` for A
- `tCgX` = CTA-partitioned GMEM tensor, e.g. `(MMA, MMA_M, MMA_K, k)` for A
- `sX` = SMEM allocation (`sA`, `sB`)
- `tCrX` = SMEM-descriptor MMA fragment, e.g. `(MMA, MMA_M, MMA_K, STAGE)` for A
- `tCtX` = TMEM tensor; `tCtAcc` = TMEM accumulator `(MMA, MMA_M, MMA_N[, ACC_STAGE])`
- `tAsA` / `tBsB` = TMA-partitioned SMEM views of A / B
- `tAgA` / `tBgB` = TMA-partitioned GMEM views of A / B
- `tTR_*` = T2R (TMEM→RMEM) partitioned tensors used in the epilogue

### Setting up the TiledMMA, MMA Ops

As shown in the data flow overview, CuTe DSL provides many utilities to tile/partition the global memory tensors, and create fragment views of SMEM and TMEM tensors for MMA instructions.

To utilize these functions, we need to setup the TiledMMA, MMA Ops first.

#### Creating a tcgen05 MMA Op

A tcgen05 MMA op describes the hardware instruction to use, it has parameters like data types, instruction shape, CTA group, operand A source (SMEM or TMEM), and operand major modes.

``` python
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05, OperandMajorMode

op = tcgen05.MmaF16BF16Op(
    cutlass.Float16,              # A/B element type
    cutlass.Float32,              # accumulator type
    (128, 256, 16),               # instruction shape (M, N, K)
    tcgen05.CtaGroup.ONE,         # CTA group
    tcgen05.OperandSource.SMEM,   # A operand from shared memory
    OperandMajorMode.K,           # A is K-major
    OperandMajorMode.K,           # B is K-major
)
```

The key parameters are:

- **Instruction shape** `(M, N, K)`: determines the size of one hardware MMA instruction. Larger M and N amortize instruction overhead.
- **OperandSource**: `SMEM` reads A from a shared memory descriptor; `TMEM` reads A directly from tensor memory.
- **OperandMajorMode**: `K` for K-major (default), `MN` for transposed layout. Transpose A requires `a_src=SMEM`; when `a_src=TMEM`, A is always K-major.

CuTe DSL provides implementation of many tcgen05 MMA ops:

| PTX name | Python class | Constructor parameters |
|----|----|----|
| `tcgen05.mma.cta_group::{cg}.kind::tf32` | `tcgen05.MmaTF32Op` | `instruction_shape, cta_group, a_src, a_major_mode, b_major_mode` |
| `tcgen05.mma.cta_group::{cg}.kind::f16` | `tcgen05.MmaF16BF16Op` | `ab_dtype, acc_dtype, instruction_shape, cta_group, a_src, a_major_mode, b_major_mode` |
| `tcgen05.mma.cta_group::{cg}.kind::i8` | `tcgen05.MmaI8Op` | `ab_dtype, instruction_shape, cta_group, a_src, a_major_mode, b_major_mode` |
| `tcgen05.mma.cta_group::{cg}.kind::f8f6f4` | `tcgen05.MmaF8F6F4Op` | `a_dtype, b_dtype, acc_dtype, instruction_shape, cta_group, a_src, a_major_mode, b_major_mode` |
| `tcgen05.mma.cta_group::{cg}.kind::mxf8f6f4.block_scale` | `tcgen05.MmaMXF8F6F4Op` | `a_dtype, b_dtype, instruction_shape, cta_group, a_src, a_major_mode, b_major_mode` |
| `tcgen05.mma.cta_group::{cg}.kind::mxf4.block_scale` | `tcgen05.MmaMXF4Op` | `instruction_shape, cta_group, a_src` |
| `tcgen05.mma.cta_group::{cg}.kind::mxf4nvf4.block_scale` | `tcgen05.MmaMXF4NVF4Op` | `sf_dtype, instruction_shape, cta_group, a_src` |

tcgen05 MMA ops

#### Creating a Tiled MMA

A `TiledMma` tiles the MMA atom across the thread block. You can pass the op directly or create an explicit atom first.

``` python
# Option 1: directly from op (common shorthand)
tiled_mma = cute.make_tiled_mma(op)

# Option 2: explicit atom creation
atom = cute.make_mma_atom(op)
tiled_mma = cute.make_tiled_mma(atom)
```

#### Spatial tiling with a repeat count (using `atom_layout_mnk`)

A repeat tuple `(M_rep, N_rep, K_rep)` replicates the atom across the M, N, and K dimensions, producing a larger tiled MMA that covers a bigger CTA tile.

``` python
tiled_mma = cute.make_tiled_mma(atom, (2, 2, 1))
```

The coordinates of atoms could be thought as a 4D coordinate: (v, m, n, k). v is the CTAs for a single MMA (for CtaGroup.ONE always 0, for CtaGroup.TWO always 0 or 1), m is the M dimension repeat count, n is the N dimension repeat count, and k is the K dimension repeat count.

``` text
MMA Atom CtaGroup.ONE                 make_tiled_mma(atom, (2, 2, 1))
+---------------+                     +----------------+----------------+
|               |                     |                |                | ^
|  128 x 256    |                     | Atom (0,0,0,0) | Atom (0,0,1,0) | |
|    x 16       |   --(2,2,1)-->      |  128 x 256     |  128 x 256     | | 2 x M_atom
|               |      repeat         |    x 16        |    x 16        | |  = 256
|               |                     |                |                | |
+---------------+                     +----------------+----------------+ |
                                      |                |                | |
                                      | Atom (0,1,0,0) | Atom (0,1,1,0) | |
                                      |  128 x 256     |  128 x 256     | |
                                      |    x 16        |    x 16        | |
                                      |                |                | v
                                      +----------------+----------------+
                                      <---- 2 x N_atom = 512 -------->
                                      K unchanged = 16
```

``` text
MMA Atom CtaGroup.TWO                 make_tiled_mma(atom, (2, 2, 1))
+---------------+                     +----------------+----------------+
| CTA v = 0     |                     | Atom (0,0,0,0) | Atom (0,0,1,0) | ^
|  128 x 256    |                     |  128 x 256     |  128 x 256     | |
|    x 16       |                     |    x 16        |    x 16        | | 2CTA Atom
+...............+                     +................+................+ |
| CTA v = 1     |   --(2,2,1)-->      | Atom (1,0,0,0) | Atom (1,0,1,0) | |
|  128 x 256    |      repeat         |  128 x 256     |  128 x 256     | |
|    x 16       |                     |    x 16        |    x 16        | v
+---------------+                     +----------------+----------------+
                                      | Atom (0,1,0,0) | Atom (0,1,1,0) | ^
                                      |  128 x 256     |  128 x 256     | |
                                      |    x 16        |    x 16        | | 2CTA Atom
                                      +................+................+ |
                                      | Atom (1,1,0,0) | Atom (1,1,1,0) | |
                                      |  128 x 256     |  128 x 256     | |
                                      |    x 16        |    x 16        | v
                                      +----------------+----------------+
                                      <---- 2 x N_atom = 512 -------->
                                      Per CTA: 2 x M_atom = 256
                                      Cluster M (v*m*128): 512
                                      K unchanged = 16
```

#### Custom tile permutation with `permutation_mnk`

`make_tiled_mma` accepts an optional `permutation_mnk` argument that controls how the atom tiles are laid out across the M, N, and K dimensions. `permutation_mnk` is a tuple of layouts or ints that represent the tile size and reordering of values. These permutation operations could be applied to optimize the data access patterns for MMAs.

For example, with `inst_m=256` and 2 atoms in M (total M tile = 512), a permutation can interleave the two atoms' M rows:

``` python
# inst_m=256, inst_n=256, inst_k=16
m_layout = cute.make_layout(
      shape=(128, 2, 2),      # (inst_m // 2, 2, 2)
      stride=(1, 256, 128),   # (1, inst_m, inst_m // 2)
)
tiled_mma = cute.make_tiled_mma(
      atom,
      atom_layout_mnk=(1, 1, 1),
      permutation_mnk=(m_layout, 256, 16),
)
```

The layout `(128,2,2):(1,256,128)` maps logical flat indices to physical M rows in colex order (mode 0 fastest), interleaving the two atoms' halves:

``` text
Without permutation                                     With permutation_mnk
(sequential, default)                                   m_layout = (128,2,2):(1,256,128)

+---------------+ ^                ^                   +---------------+ ^
| MMA 0 top     | | 128  CTA 0     |                   | MMA 0 top     | | 128  CTA 0
| rows 0-127    | |                |                   | rows 0-127    | |
+...............+ +                |  Tile 0           +---------------+ v
| MMA 0 bottom  | | 128  CTA 1     |                   | MMA 1 top     | | 128  CTA 0
| rows 128-255  | |                |                   | rows 128-255  | |
+---------------+ v                v                   +---------------+ v
| MMA 1 top     | ^                ^                   | MMA 0 bottom  | | 128  CTA 1
| rows 256-383  | | 128  CTA 0     |                   | rows 256-383  | |
+...............+ +                |  Tile 1           +---------------+ v
| MMA 1 bottom  | | 128  CTA 1     |                   | MMA 1 bottom  | | 128  CTA 1
| rows 384-511  | |                |                   | rows 384-511  | |
+---------------+ v                v                   +---------------+ v
<-- inst_N=256 ->                                      <-- inst_N=256 ->
inst_K = 16                                            inst_K = 16

Tile 0: rows 0-255   (contiguous)                      Tile 0: rows {0-127, 256-383}
Tile 1: rows 256-511 (contiguous)                      Tile 1: rows {128-255, 384-511}
CTA 0 owns rows {0-127, 256-383}                       CTA 0 owns rows {0-127, 256-383}
CTA 1 owns rows {128-255, 384-511}                     CTA 1 owns rows {128-255, 384-511}
```

When `permutation_mnk` is not provided (default), the tile ordering is sequential and no permutation is applied.

#### Creating Trivial Tiled MMA

Since tcgen05 MMAs have quite large instruction shapes, most common TiledMmas created are trivial tiled MMAs, with single M, N repetitions, i.e., `atom_layout_mnk`, and `permutation_mnk` are generally unused. CuTe DSL provides a convenience function `make_trivial_tiled_mma` to create such trivial MMAs with automatic MmaOp kind selection based on the data types.

``` python
import cutlass.utils.blackwell_helpers as sm100_utils

tiled_mma = sm100_utils.make_trivial_tiled_mma(
      a_dtype,
      b_dtype,
      a_major_mode,
      b_major_mode,
      acc_dtype,
      cta_group,
      mma_tiler_mnk,
)

# Equivalent to
tiled_mma = cute.make_tiled_mma(
      cute.make_mma_atom(
            cute.MmaXyzOp(
                  # ... parameters of MmaXyzOp
            ),
      ),
)
```

### Partitioning Tensors

Before computing MMAs, we want to partition the global memory tensors according to the tiled MMA layout. For tcgen05, this maps each CTA's work to the correct portion of the global memory tensors.

We have two steps to partition the global memory tensors:

- Local tile partitioning: partition the global memory tensors into local tiles, each of size `mma_tiler_mnk`. This is the portion of the global memory tensors that will be processed by a single CTA MMA or a 2CTA cooperative MMA.
- MMA partition: partition the local tile into CTA-sized, per-MMA-instruction tiles (note that each CTA needs to load its own portion to SMEM for 2CTA cooperative MMA). The per-operand shapes are `(MMA, MMA_M, MMA_K, ...)` for A, `(MMA, MMA_N, MMA_K, ...)` for B, and `(MMA, MMA_M, MMA_N, ...)` for C.

Note that for tcgen05, SMEM tensors are not partitioned. See [Making Fragments](#making-fragments) for more details.

#### Trivial TiledMma with CtaGroup.ONE MMAs (single CTA):

For the trivial tiled MMAs with CtaGroup.ONE tcgen05 MMA operations, partitioning the mma_tiler sized tile is an identity operation, i.e., single CTAs' tile is the same as the mma_tiler sized tile. The main difference between the result of `local_tile` and `partition_[A/B]` is that, the latter produces a view that can be iterated in per-MMA instruction fashion.

Example: `GEMM (M, N, K) = (512, 768, 384)`, `mma_tiler_mnk = (128, 256, 64)`, `CtaGroup.ONE`, F16 atom = 128x256x16 (inst_M=128, inst_N=256, inst_K=16).

Global memory tensors:

``` text
mA: (M, K) = (512, 384)       mB: (N, K) = (768, 384)       mC: (M, N) = (512, 768)

     K=384                          K=384                       N=768
   |<----------->|               |<----------->|           |<----------------->|
   +-------------+               +-------------+           +---+---+---+-------+
   |             | ^             |             | ^         |   |   |   |       | ^
   |     mA      | | M=512       |     mB      | | N=768   |   |   |   |       | | M=512
   |             | v             |             | v         |   |   |   |       | v
   +-------------+               +-------------+           +---+---+---+-------+
```

Tiling with `mma_tiler_mnk = (BM, BN, BK) = (128, 256, 64)` gives M/BM = 512/128 = 4 tiles, N/BN = 768/256 = 3 tiles, K/BK = 384/64 = 6 tiles:

``` text
mA tiled into (M/BM x K/BK)   mB tiled into (N/BN x K/BK)     mC tiled into (M/BM x N/BN)
= (4 x 6) blocks              = (3 x 6) blocks                = (4 x 3) blocks
                                                               * coordinates annotated on the matrix
                                                                 are the mma_coord_mn of the GEMM.

  BK=64  x6                     BK=64  x6                        BN=256  x3
|<--->|                        |<--->|                        |<----->|
+-----+-----+-- --+            +-----+-----+-- --+            +-------+-------+-------+
|     |     |..|  | ^          |     |     |..|  | ^          | (0,0) | (0,1) | (0,2) | ^
|     |     |  |  | | BM=128   |     |     |  |  | | BN=256   |       |       |       | | BM=128
+-----+-----+-- --+ v          +-----+-----+-- --+ v          +-------+-------+-------+ v
|     |     |..|  | ^          |     |     |..|  | ^          | (1,0) | (1,1) | (1,2) | ^
|     |     |  |  | | BM=128   |     |     |  |  | | BN=256   |       |       |       | | BM=128
+-----+-----+-- --+ v          +-----+-----+-- --+ v          +-------+-------+-------+ v
|     |     |..|  | ^          |     |     |..|  | ^          | (2,0) | (2,1) | (2,2) | ^
|     |     |  |  | | BM=128   |     |     |  |  | | BN=256   |       |       |       | | BM=128
+-----+-----+-- --+ v          +-----+-----+-- --+ v          +-------+-------+-------+ v
|     |     |..|  | ^                                         | (3,0) | (3,1) | (3,2) | ^
|     |     |  |  | | BM=128                                  |       |       |       | | BM=128
+-----+-----+-- --+ v                                         +-------+-------+-------+ v
```

Each CTA picks one (M-coord, N-coord) coordinate. For example, CTA at `mma_coord = (0, 1, :)`.

After `local_tile` — one CTA's tile has `k = K/BK = 384/64 = 6` tiles to process for A, B tensors, and a single tile for C tensor:

``` text
gA: (BM, BK, k) = (128, 64, 6)   gB: (BN, BK, k) = (256, 64, 6)   gC: (BM, BN) = (128, 256)
(k has 6 tiles total: indices 0..5)

     BK=64                              BK=64                                BN=256
   |<----->|                          |<----->|                           |<--------->|
   +-------+---------+-------+        +-------+---------+-------+         +-----------+
   |       |         |       |        |       |         |       |         |           | ^
BM= | gA k0 | k1...k4 | gA k5 |    BN= | gB k0 | k1...k4 | gB k5 |     BM= |    gC     | 128
128 |       |         |       |    256 |       |         |       |     128 |           | v
   +-------+---------+-------+        +-------+---------+-------+         +-----------+
```

`get_slice(0)` — single CTA owns the full tile. BM and BN match the atom, BK is split into MMA_K atom-sized steps:

``` text
gA (BK split into MMA_K atoms)                           gC
  inst_K  inst_K  inst_K  inst_K
  =16     =16     =16     =16
|<--->|<--->|<--->|<--->|
+-----+-----+-----+-----+--                             +-----------+
|  0  |  1  |  2  |  3  |..  BM=128  (MMA_M=1)          |           | BM=128
+-----+-----+-----+-----+                               +-----------+
|<-- MMA_K = BK/inst_K = 4 -->|

gB (BK split into MMA_K atoms)
  inst_K  inst_K  inst_K  inst_K
  =16     =16     =16     =16
|<--->|<--->|<--->|<--->|
+-----+-----+-----+-----+--
|  0  |  1  |  2  |  3  |..  BN=256  (MMA_N=1)
+-----+-----+-----+-----+
```

After partition (single CTA):

- `tCgA: (MMA, MMA_M, MMA_K, k) = (MMA, 1, 4, 6)` — MMA_M = BM/inst_M = 128/128 = 1, MMA_K = BK/inst_K = 64/16 = 4
- `tCgB: (MMA, MMA_N, MMA_K, k) = (MMA, 1, 4, 6)` — MMA_N = BN/inst_N = 256/256 = 1, MMA_K = BK/inst_K = 64/16 = 4
- `tCgC: (MMA, MMA_M, MMA_N) = (MMA, 1, 1)` — MMA_M = BM/inst_M = 1, MMA_N = BN/inst_N = 1

With CuTe DSL, all these calculations are handled for you provided `mma_tiler_mnk` and `mma_coord`.

``` python
@cute.kernel
def kernel(tiled_mma: cute.TiledMma, ...):
    gA = cute.local_tile(mA, mma_tiler_mnk, mma_coord, proj=(1, None, 1)) # (BM, BK, k) for A
    gB = cute.local_tile(mB, mma_tiler_mnk, mma_coord, proj=(None, 1, 1)) # (BN, BK, k) for B
    gC = cute.local_tile(mC, mma_tiler_mnk, mma_coord, proj=(1, 1, None)) # (BM, BN) for C

    # Single CTA MMA: cta index is always 0
    thr_mma = tiled_mma.get_slice(0)

    tCgA = thr_mma.partition_A(gA)   # (MMA, MMA_M, MMA_K, num_k_tiles) for A
    tCgB = thr_mma.partition_B(gB)   # (MMA, MMA_N, MMA_K, num_k_tiles) for B
    tCgC = thr_mma.partition_C(gC)   # (MMA, MMA_M, MMA_N) for C
```

#### Trivial TiledMma with CtaGroup.TWO MMAs (2-CTA cluster, each CTA owns half the M-tile):

With `CtaGroup.TWO`, two CTAs cooperate on a single tile. The V-coordinate (0 or 1) identifies which CTA within the pair. `get_slice(V)` gives each CTA its half of the M dimension, while B is fully shared.

Example: `GEMM (M, N, K) = (512, 768, 384)`, `mma_tiler_mnk = (256, 256, 64)`, `CtaGroup.TWO`, F16 MMA atom = 128x256x16 (inst_M=128, inst_N=256, inst_K=16).

Global matrices:

``` text
mA: (M, K) = (512, 384)       mB: (N, K) = (768, 384)        mC: (M, N) = (512, 768)

    K=384                          K=384                            N=768
|<----------->|               |<----------->|                |<----------------->|
+-------------+               +-------------+                +---+---+---+-------+
|             | ^             |             | ^              |   |   |   |       | ^
|     mA      | | M=512       |     mB      | | N=768        |   |   |   |       | | M=512
|             | v             |             | v              |   |   |   |       | v
+-------------+               +-------------+                +---+---+---+-------+
```

Tiling with `mma_tiler_mnk = (BM, BN, BK) = (256, 256, 64)` gives M/BM = 512/256 = 2 tiles in M-mode, N/BN = 768/256 = 3 tiles in N-mode, K/BK = 384/64 = 6 tiles in K-mode:

``` text
mA tiled into (M/BM x K/BK)       mB tiled into (N/BN x K/BK)     mC tiled into (M/BM x N/BN)
= (2 x 6) blocks                  = (3 x 6) blocks                 = (2 x 3) blocks
                                                                   * coordinates annotated on the matrix
                                                                     are the mma_coord_mn of the GEMM.

  BK=64  x6                       BK=64  x6                         BN=256  x3
|<--->|                         |<--->|                        |<----->|
+-----+-----+-- --+             +-----+-----+-- --+            +-------+-------+-------+
|     |     |..|  | ^           |     |     |..|  | ^          | (0,0) | (0,1) | (0,2) | ^
|     |     |  |  | | BM=256    |     |     |  |  | | BN=256   |       |       |       | | BM=256
+-----+-----+-- --+ v           +-----+-----+-- --+ v          +-------+-------+-------+ v
|     |     |..|  | ^           |     |     |..|  | ^          | (1,0) | (1,1) | (1,2) | ^
|     |     |  |  | | BM=256    |     |     |  |  | | BN=256   |       |       |       | | BM=256
+-----+-----+-- --+ v           +-----+-----+-- --+ v          +-------+-------+-------+ v
                                |     |     |..|  | ^
                                |     |     |  |  | | BN=256
                                +-----+-----+-- --+ v
```

Each CTA pair picks one (M-coord, N-coord) coordinate. For example, CTA pair at `mma_coord_mnk = (0, 0, :)`.

``` text
gA: (BM, BK, k) = (256, 64, 6)   gB: (BN, BK, k) = (256, 64, 6)   gC: (BM, BN) = (256, 256)

     BK=64                          BK=64                       BN=256
   |<----->|                     |<----->|                 |<--------->|
   +-------+--                   +-------+--               +-----------+
   |       |..                   |       |..               |           | ^
BM=  |  gA   | k=6             BN= |  gB   | k=6         BM= |    gC     | 256
256 |       |                 256 |       |                 |           | v
   +-------+                     +-------+                 +-----------+
```

`get_slice(V)` splits BM between CTAs; BK is split into `MMA_K` steps:

``` text
gA (BM split, BK split into MMA_K atoms)                gC (BM split)
  inst_K  inst_K  inst_K  inst_K
  =16     =16     =16     =16
|<--->|<--->|<--->|<--->|
+-----+-----+-----+-----+--                             +-----------+
|  0  |  1  |  2  |  3  |..  ^  CTA 0                   |   CTA 0   | ^
|     |     |     |     |    | BM/2=128 (V=0)           |   (V=0)   | | BM/2=128
+-----+-----+-----+-----+    v                          +-----------+ v
|  0  |  1  |  2  |  3  |..  ^  CTA 1                   |   CTA 1   | ^
|     |     |     |     |    | BM/2=128 (V=1)           |   (V=1)   | | BM/2=128
+-----+-----+-----+-----+    v                          +-----------+ v
|<-- MMA_K = BK/inst_K = 4 -->|

gB (BN split for SMEM loading, BK split into MMA_K atoms)
  inst_K  inst_K  inst_K  inst_K
  =16     =16     =16     =16
|<--->|<--->|<--->|<--->|
+-----+-----+-----+-----+--
|  0  |  1  |  2  |  3  |..  ^  CTA 0
|     |     |     |     |    | BN/2=128  (V=0)
+-----+-----+-----+-----+    v
|  0  |  1  |  2  |  3  |..  ^  CTA 1
|     |     |     |     |    | BN/2=128  (V=1)
+-----+-----+-----+-----+    v
```

Both CTAs consume the full gB for MMA, but for SMEM loading each CTA loads its N-half.

After partition (per CTA, e.g. CTA 0):

- `tCgA: (MMA, MMA_M, MMA_K, k) = (MMA, 1, 4, 6)` — MMA_M = (BM/2)/inst_M = 128/128 = 1, MMA_K = BK/inst_K = 64/16 = 4
- `tCgB: (MMA, MMA_N, MMA_K, k) = (MMA, 1, 4, 6)` — MMA_N = BN/inst_N = 256/256 = 1, MMA_K = BK/inst_K = 64/16 = 4
- `tCgC: (MMA, MMA_M, MMA_N) = (MMA, 1, 1)` — MMA_M = (BM/2)/inst_M = 1, MMA_N = BN/inst_N = 1

Of course with CuTe DSL none of these calculations are needed. The DSL handles all the tiling and partitioning for you provided `mma_tiler_mnk` and `mma_coord`.

``` python
@cute.kernel
def kernel(tiled_mma: cute.TiledMma, cta_layout_vmnk: cute.Layout, ...):
    bidx, bidy, _ = cute.arch.block_idx()

    # V-coordinate: which CTA within the 2-CTA group (0 or 1)
    mma_coord_vmnk = (
        bidx % cute.size(cta_layout_vmnk, mode=[0]),   # V (CTA rank)
        bidx // cute.size(cta_layout_vmnk, mode=[0]),   # M tile
        bidy,                                           # N tile
        None,                                           # K (all tiles)
    )
    mma_coord_mnk = mma_coord_vmnk[1:]

    gA = cute.local_tile(mA, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
    gB = cute.local_tile(mB, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
    gC = cute.local_tile(mC, mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))

    # 2-CTA: each CTA passes its V-coordinate to get its half of the work
    thr_mma = tiled_mma.get_slice(mma_coord_vmnk[0])

    tCgA = thr_mma.partition_A(gA)   # (MMA, MMA_M, MMA_K, num_k_tiles)
    tCgB = thr_mma.partition_B(gB)   # (MMA, MMA_N, MMA_K, num_k_tiles)
    tCgC = thr_mma.partition_C(gC)   # (MMA, MMA_M, MMA_N)
```

> [!NOTE]
> Annotation `tCgX` means that the tensor is partitioned w.r.t C matrix coordinates, i.e., the output tile of each CTA.

#### Pre and Post-Conditions for TiledMMA Partitioning

- The inputs of the partition should be at least rank-2 tensors.
- The output of the partition will have the layout that is compatible with the MMA atom's operand:
  - For A, the output will have the layout (MMA, MMA_M, MMA_K, ...).
  - For B, the output will have the layout (MMA, MMA_N, MMA_K, ...).
  - For C, the output will have the layout (MMA, MMA_M, MMA_N, ...).
- Note that the partition doesn't enforce any rules on the tensor's memory space or the tensor's data type. It only cares about the layout.

#### What happens when we use `atom_layout_mnk`?

The valid coordinates to `get_slice` are the valid coordinates to (v,m,n,k) coordinate space of the tiled MMA. `mma_tiler_mnk` should be updated such that `mma_tiler_mnk[0] >= mma_shape[0] * |m|`, `mma_tiler_mnk[1] >= mma_shape[1] * |n|`, and `mma_tiler_mnk[2] >= mma_shape[2] * |k|`.

The result of the `partition_A`, `partition_B`, and `partition_C` remain the same.

### Making Fragments

Fragments are the descriptor-level tensors that the MMA instruction operates on. For tcgen05:

- **Fragment A**: SMEM descriptor when `a_src=SMEM`, or a TMEM address when `a_src=TMEM`.
- **Fragment B**: SMEM descriptor pointing into staged shared memory buffers.
- **Fragment C (accumulator)**: lives in Tensor Memory (TMEM), allocated via `TmemAllocator`.

#### Creating fragment descriptors and descriptor tensors

Unlike older architectures where fragments live in per-thread registers, tcgen05 fragments are **descriptors** pointing into SMEM (for A and B) or **addresses** into TMEM (for the accumulator C). The fragment creation has three parts:

**1. A and B fragments**

*When A comes from SMEM* (`a_src=OperandSource.SMEM`):

`make_fragment_A` and `make_fragment_B` take the staged SMEM tensors (`sA`, `sB`) and produce descriptor tensors that the MMA instruction consumes. Each descriptor points to one stage's tile in shared memory.

``` python
# 1. Build the SMEM layouts (see "Creating SMEM layouts for A and B")
# a_smem_layout = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, ...)
# b_smem_layout = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, ...)

# 2. Allocate SMEM tensors from those layouts
# sA = smem.allocate_tensor(layout=a_smem_layout.outer, swizzle=a_smem_layout.inner, ...)
# sB = smem.allocate_tensor(layout=b_smem_layout.outer, swizzle=b_smem_layout.inner, ...)

# 3. Create fragment descriptors from the SMEM tensors
tCrA = tiled_mma.make_fragment_A(sA)  # (MMA, MMA_M, MMA_K, STAGE)
tCrB = tiled_mma.make_fragment_B(sB)  # (MMA, MMA_N, MMA_K, STAGE)
```

Continuing the CtaGroup.ONE example (m128n256k16 atom, `mma_tiler_mnk = (128, 256, 64)`, 3 pipeline stages):

``` text
sA is an SMEM tensor with shape (MMA, MMA_M, MMA_K, STAGES),
allocated with appropriate size (see "Creating SMEM layouts for A and B").

For 128x256x16 atom, mma_tiler_mnk = (BM, BN, BK) = (128, 256, 64), 3 stages:
 MMA_M = BM/inst_M = 128/128 = 1, MMA_K = BK/inst_K = 64/16 = 4, STAGES = 3

sA: (MMA, MMA_M=1, MMA_K=4, STAGES=3)

              |<--MMA_K = BK/inst_K = 4-->|
  stage 0:    +------+------+------+------+
              | k=0  | k=1  | k=2  | k=3  |  inst_M=128
              +------+------+------+------+
  stage 1:    +------+------+------+------+
              | k=0  | k=1  | k=2  | k=3  |  inst_M=128
              +------+------+------+------+
  stage 2:    +------+------+------+------+
              | k=0  | k=1  | k=2  | k=3  |  inst_M=128
              +------+------+------+------+

make_fragment_A(sA) produces SMEM descriptors with the same shape:
tCrA: (MMA, MMA_M, MMA_K, STAGES) = (MMA, 1, 4, 3)

Each element is an SMEM descriptor — one per (MMA_K, STAGE) pair.
Similarly for sB/tCrB with shape (MMA, MMA_N=1, MMA_K=4, STAGE=3).
```

Each element of `tCrA` / `tCrB` is an SMEM descriptor that the MMA hardware reads directly — not a register value. Note that, when we print the layout of `tCrA` (or similarly `tCrB`), we will see that `MMA` dimension of `(MMA, MMA_M, MMA_K, STAGES)` will appear to be `1`. This is because this mode is an indivisible SMEM descriptor representing the whole SMEM buffer that a single MMA instruction will consume.

*When A comes from TMEM* (`a_src=OperandSource.TMEM`):

In use cases like FMHA or mixed-input GEMM, operand A can be sourced from TMEM instead of SMEM. In this case, `make_fragment_A` is called to obtain the layout, but the fragment is bound to a TMEM pointer instead of an SMEM tensor:

``` python
# Build the SMEM layout for A (see `Creating SMEM layouts for A and B`_).
# The layout defines the tile shape the MMA expects, even though the data
# will live in TMEM.
# a_smem_layout = sm100_utils.make_smem_layout_a(...)

# Use make_fragment_A with the outer layout to get the expected shape
tCrA_layout = tiled_mma.make_fragment_A(a_smem_layout.outer).layout

# Compute the TMEM pointer offset (A is placed after the accumulator columns).
# TMEM columns are 32-bit wide, so scale to element offset for narrower types
# (e.g. Float16: scale = 32 // 16 = 2).
column_to_element_scale = 32 // acc_dtype.width
tmem_ptr_a = cute.recast_ptr(
    accumulators.iterator + num_acc_tmem_cols * column_to_element_scale,
    dtype=mma_dtype,
)

# Bind to TMEM storage
tCrA = cute.make_tensor(tmem_ptr_a, tCrA_layout)
```

The A fragment in TMEM is laid out after the accumulator's TMEM columns.

**2. C fragment (accumulator) — TMEM allocation**

The accumulator lives in Tensor Memory (TMEM), a dedicated on-chip memory separate from registers and SMEM. Creating the C fragment is a four-step process:

``` python
# Step 1: Query the partitioned accumulator shape
acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
# acc_shape: (MMA, MMA_M, MMA_N) = (MMA, 1, 1)

# Step 2: Append a staging dimension for ping-pong overlap
#   Use 1 for simple kernels (no overlap), or 2+ to overlap
#   MMA and epilogue on different TMEM buffers.
num_acc_stages = 2
acc_shape_staged = cute.append(acc_shape, num_acc_stages)

# Step 3: Create a fragment to establish the layout
tCtAcc = tiled_mma.make_fragment_C(acc_shape_staged)
# tCtAcc: (MMA, MMA_M, MMA_N, ACC_STAGE)

# Step 4: Bind to actual TMEM storage
tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)
```

``` text
partition_shape_C((BM, BN) = (128, 256))
 -> (MMA, MMA_M, MMA_N) = (MMA, 1, 1)
     MMA_M = BM/inst_M = 128/128 = 1
     MMA_N = BN/inst_N = 256/256 = 1

cute.append(acc_shape, 2)
 -> (MMA, 1, 1, 2)

make_fragment_C(acc_shape_staged)
 -> tCtAcc layout: ((128, 256), 1, 1, 2)

After binding to TMEM (2-stage ping-pong):
+---------------------------+---------------------------+
|  tCtAcc stage 0           |  tCtAcc stage 1           |
|  128 x 256 accumulators   |  128 x 256 accumulators   |
|  (Float32)                |  (Float32)                |
+---------------------------+---------------------------+
```

#### Creating SMEM layouts for A and B

The SMEM layouts define how A and B tiles are stored in shared memory, including swizzling for bank-conflict-free access. The helper functions handle the details: partitioned shape from the tiled MMA, swizzle atom selection, tiling to the MMA shape, and multi-stage buffering.

**Host side** (`@cute.jit`):

``` python
import cutlass.utils.blackwell_helpers as sm100_utils

# Create SMEM layouts (includes swizzle + staging)
a_smem_layout = sm100_utils.make_smem_layout_a(
    tiled_mma, mma_tiler_mnk, a.element_type, num_stages,
)
b_smem_layout = sm100_utils.make_smem_layout_b(
    tiled_mma, mma_tiler_mnk, b.element_type, num_stages,
)
```

`make_smem_layout_a` and `make_smem_layout_b` are convenience helpers that build a complete, staged SMEM layout in four steps:

1.  **Determine the major mode.** The major mode (K-major or MN-major) is read from the MMA op's `a_major_mode` / `b_major_mode` attribute (or can be overridden via the `is_k_major` keyword argument).

2.  **Compute the partitioned SMEM tile shape.** The tiled MMA is asked for the partitioned shape of the operand via `tiled_mma.partition_shape_A` (or `partition_shape_B`). `cute.dice` strips the irrelevant mode first — for A the `(M, K)` portion is kept, for B the `(N, K)` portion. The result is a hierarchical shape `((MMA, MMA_MN, MMA_K), repeat_MN, repeat_K)` that is flattened into a 2D `(MN, K)` size for swizzle selection.

3.  **Select and materialise the swizzle atom.** A heuristic (`get_smem_layout_atom_ab`) picks the widest swizzle whose contiguous size (in bits) evenly divides the major-mode dimension:

    | Swizzle    | Contiguous bits |
    |------------|-----------------|
    | SW128      | 1024 (128 B)    |
    | SW64       | 512 (64 B)      |
    | SW32       | 256 (32 B)      |
    | Interleave | 128 (16 B)      |

    `make_smem_layout_atom` then combines the chosen swizzle with a compact `(MN_elems, 8)` or `(8, K_elems)` outer layout (depending on the major mode) into a `ComposedLayout(swizzle, outer)`.

4.  **Tile to the MMA shape and append the staging dimension.** `tile_to_mma_shape` broadcasts the atom to the full partitioned shape (with `num_stages` appended). The `order` argument controls which dimension is contiguous: `(1, 2, 3)` for K-major (K innermost), `(2, 1, 3)` for MN-major (MN innermost).

The resulting layout is then fed into SMEM tensors are allocated using the layout info:

**Kernel side** (`@cute.kernel`):

``` python
smem = cutlass.utils.SmemAllocator()
sA = smem.allocate_tensor(
    element_type=io_dtype,
    layout=a_smem_layout.outer,
    byte_alignment=128,
    swizzle=a_smem_layout.inner,
)
sB = smem.allocate_tensor(
    element_type=io_dtype,
    layout=b_smem_layout.outer,
    byte_alignment=128,
    swizzle=b_smem_layout.inner,
)
```

> [!NOTE]
> **Creating SMEM layouts without utilities** If you want to create SMEM layouts without using the utilities, you can do the following:
>
> ``` python
> swizzle = cute.Swizzle(3, 4, 3)
> mma_tile = tiled_mma.partition_shape_A((mma_tiler_mnk[0], mma_tiler_mnk[2]))
> smem_tile = tcgen05.tile_to_mma_shape(swizzle, mma_tile, order=(1, 2, 3))
> ```

### Executing the GEMM (Main Loop)

The main loop iterates over K-tiles, loading A and B from global memory via TMA into staged SMEM buffers, then issuing `cute.gemm` for each tile. The TMA copy details are omitted for brevity.

``` python
for k_tile_idx in cutlass.range(num_k_tiles):
   # Wait for TMA load to complete for this K-tile
   ab_full = ab_consumer.wait_and_advance()

   # Set accumulate mode: first tile overwrites, subsequent tiles accumulate
   tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile_idx != 0)

   # Issue MMA: tCtAcc += tCrA * tCrB
   tile_crd = (None, None, None, ab_full.index)
   cute.gemm(tiled_mma, tCtAcc, tCrA[tile_crd], tCrB[tile_crd], tCtAcc)

   # Release the SMEM buffer for the next TMA load
   ab_full.release()
```

Key points:

- `tcgen05.Field.ACCUMULATE` controls whether the MMA accumulates into D (`True`) or overwrites D with `A * B` (`False`). Set to `False` for the first K-tile and `True` for all subsequent tiles.
- `cute.gemm` is asynchronous. Synchronization is handled by the pipeline barriers (`cutlass.pipeline.sm100.PipelineTmaUmma`).
- The `tile_crd` selects which pipeline stage's SMEM buffer to read from.

### Reading the accumulator from TMEM

``` python
tCtAcc = tiled_mma.make_fragment_C(mma_tiler_mnk[:2]) # (MMA, MMA_M, MMA_N) for C
# TMEM allocation (done once, before the main loop)
tmem = cutlass.utils.TmemAllocator(...)
tmem.allocate(num_cols=512)
tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

# Build copy atom for TMEM → RMEM load
copy_atom_t2r = cute.make_copy_atom(
    tcgen05.Ld32x32bOp(tcgen05.Repetition.x64),
    cutlass.Float32,
)
tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc[(None, None), 0, 0])
thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)

# (T2R, T2R_M, NumTiles)
tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc)
# (T2R, T2R_M, NumTiles)
tTR_gC = thr_copy_t2r.partition_D(tCgC)

# (T2R, T2R_M)
tTR_rAcc = cute.make_rmem_tensor(tTR_gC[None, None, 0].shape, acc_dtype)

cute.copy(tiled_copy_t2r, tTR_tAcc[None, None, i], tTR_rAcc)
```

### Complete Workflow

Putting it all together, a typical Blackwell tcgen05 GEMM has this structure:

**Host function** (`@cute.jit`):

``` python
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05, OperandMajorMode
import cutlass.utils.blackwell_helpers as sm100_utils

@cute.jit
def host_function(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    # 1. Create the MMA op and tiled MMA
    op = tcgen05.MmaF16BF16Op(
        cutlass.Float16, cutlass.Float32,
        (128, 256, 16),
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        OperandMajorMode.K,
        OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(op)

    # 2. Create SMEM layouts for A and B
    a_smem_layout = sm100_utils.make_smem_layout_a(
        tiled_mma, mma_tiler_mnk, a.element_type, num_stages,
    )
    b_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma, mma_tiler_mnk, b.element_type, num_stages,
    )

    # 3. Create TMA copy atoms for global -> shared memory loads
    copy_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    tma_a = cute.nvgpu.make_tiled_tma_atom_A(
        copy_op, a, a_smem_layout, mma_tiler_mnk, tiled_mma,
    )
    tma_b = cute.nvgpu.make_tiled_tma_atom_B(
        copy_op, b, b_smem_layout, mma_tiler_mnk, tiled_mma,
    )

    # 4. Launch the kernel
    grid = cute.ceil_div((*c.layout.shape, 1), mma_tiler_mnk[:2])
    kernel(tiled_mma, tma_a, tma_b, c).launch(
        grid=grid, block=(128, 1, 1),
    )
```

**Kernel function** (`@cute.kernel`):

``` python
@cute.kernel
def kernel(
    tiled_mma: cute.TiledMma,
    tma_a: cpasync.TmaInfo,
    tma_b: cpasync.TmaInfo,
    mC: cute.Tensor,
):
    # -- Setup --
    bidx, bidy, _ = cute.arch.block_idx()
    mma_coord_mnk = (bidx, bidy, None)

    # Global tensors for A and B live inside the TMA descriptor
    mA = tma_a.tma_tensor   # (M, K)
    mB = tma_b.tma_tensor   # (N, K)

    # Allocate SMEM for A, B (staged) and pipeline barriers
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(...)  # staged SMEM for A
    sB = smem.allocate_tensor(...)  # staged SMEM for B

    # Allocate TMEM for the accumulator
    tmem = cutlass.utils.TmemAllocator(...)
    tmem.allocate(num_cols=512)

    # -- Partition and make fragments --
    # (BM, BK, k)
    gA = cute.local_tile(mA, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
    # (BN, BK, k)
    gB = cute.local_tile(mB, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
    # (BM, BN)
    gC = cute.local_tile(mC, mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))

    thr_mma = tiled_mma.get_slice(0)
    tCgA = thr_mma.partition_A(gA)   # (MMA, MMA_M, MMA_K, num_k_tiles)
    tCgB = thr_mma.partition_B(gB)   # (MMA, MMA_N, MMA_K, num_k_tiles)
    tCgC = thr_mma.partition_C(gC)   # (MMA, MMA_M, MMA_N)

    # SMEM descriptor fragments
    tCrA = tiled_mma.make_fragment_A(sA)   # (MMA, MMA_M, MMA_K, STAGE)
    tCrB = tiled_mma.make_fragment_B(sB)   # (MMA, MMA_N, MMA_K, STAGE)

    # TMEM accumulator
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)  # (MMA, MMA_M, MMA_N)

    # Bind accumulator to TMEM
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

    # TMA partition for global → shared memory copies
    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        tma_a.atom, 0, cute.make_layout(1),
        cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3),
    )
    tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
        tma_b.atom, 0, cute.make_layout(1),
        cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3),
    )

    # -- Main loop: iterate over K-tiles --
    num_k_tiles = cute.size(gA, mode=[2])
    for k_tile_idx in cutlass.range(num_k_tiles):
        # TMA load A, B into staged SMEM (producer side)
        # copy(tma_a.atom, tAgA[k], tAsA[stage])
        # copy(tma_b.atom, tBgB[k], tBsB[stage])
        # ... (see pipeline documentation)

        # Wait for data
        ab_full = ab_consumer.wait_and_advance()

        # MMA
        tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile_idx != 0)
        tile_crd = (None, None, None, ab_full.index)
        cute.gemm(tiled_mma, tCtAcc, tCrA[tile_crd], tCrB[tile_crd], tCtAcc)

        ab_full.release()

    # -- Epilogue: copy accumulator from TMEM to global memory --
    copy_atom_t2r = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), cutlass.Float32,
    )
    tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc[(None, None), 0, 0])
    thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)

    tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc)
    tTR_gC   = thr_copy_t2r.partition_D(tCgC)
    tTR_rAcc = cute.make_rmem_tensor(tTR_gC[None, None, 0].shape, acc_dtype)

    # TMEM → RMEM, then RMEM → GMEM
    for i in cutlass.range(num_tiles):
        cute.copy(tiled_copy_t2r, tTR_tAcc[None, None, i], tTR_rAcc)
        cute.copy(store_atom, tTR_rAcc, tTR_gC[None, None, i])
```

### Beyond Simple Dense MMAs

The tcgen05 MMA DSL supports more complex MMA operations than just the simple dense MMA.

- Block-scaled MMA

Internal builds additionally provide:

- Sparse MMA

#### Sparse MMA

Sparse MMA exploits **X:Y = {1:2, 2:4, 4:8} structured sparsity** in operand A: out of every Y consecutive K-elements, exactly X are non-zero and Y-X are zero. The kernel stores the compressed A values separately from the **metadata** tensor `E`, which records which 2 of 4 positions are non-zero.

Compared to a dense MMA kernel, a sparse kernel differs in five areas:

**1. MMA op creation** — use `MmaF16BF16SparseOp` with an extra `sparse_metadata_format` parameter. The instruction K is **doubled** (32 vs 16 for dense F16/BF16) to account for the compressed operand. The example here builds its sparse `TiledMma` through `sm100_utils.make_sparse_trivial_tiled_mma(...)`:

``` python
from cutlass.cute.nvgpu.warp.mma import SparseMetadataFormat

# Dense F16 (for comparison): inst_K = 16
dense_op = tcgen05.MmaF16BF16Op(
    cutlass.Float16, cutlass.Float32, (128, 256, 16),
    tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
    cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
)

# Sparse F16: inst_K = 32 (2× dense, since A is 2:4 compressed)
sparse_op = tcgen05.MmaF16BF16SparseOp(
    cutlass.Float16, cutlass.Float32, (128, 256, 32),
    tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
    cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
    SparseMetadataFormat.TID,
)

# The sparse GEMM example uses the public helper
tiled_mma = sm100_utils.make_sparse_trivial_tiled_mma(
    a_raw_dtype, a_major_mode, b_major_mode, acc_dtype, cta_group,
    mma_tiler_mn=mma_tiler_mnk[:2],
    sparse_metadata_format=SparseMetadataFormat.TID,
)
```

**2. Compressed A and metadata E tensors** — operand A is stored with **half** the K-elements (the two non-zero values per group of 4), using a `sparse_elem<2, dtype>` type. The metadata tensor E is a compact bit-field (`sparse_elem<8, uint8>` for F16/BF16) that encodes the sparsity pattern.

``` python
# Sparse element types
a_sparse_dtype = sm100_utils.make_sparse_a_dtype(a_raw_dtype)   # sparse_elem<2, F16>
e_sparse_dtype = sm100_utils.make_sparse_e_dtype(a_raw_dtype)   # sparse_elem<8, uint8>

# GMEM layouts for compressed A and metadata E
sp_a_ptr = cute.recast_ptr(a.iterator, dtype=a_sparse_dtype)
sp_a_layout = sm100_utils.make_sparse_gmem_layout_a(
    mnkl,
    a_raw_dtype,
    is_k_major=(a_major_mode == cute.nvgpu.OperandMajorMode.K),
    sparsity=2,
)
sp_a_tensor = cute.make_tensor(sp_a_ptr, sp_a_layout)

sp_e_ptr = cute.recast_ptr(e.iterator, dtype=e_sparse_dtype)
sp_e_layout = sm100_utils.make_sparse_gmem_layout_e(mnkl, a_raw_dtype)
sp_e_tensor = cute.make_tensor(sp_e_ptr, sp_e_layout)
```

``` text
Dense A: (M, K)                    Sparse A (compressed): (M, (2, K/2))
+--+--+--+--+--+--+--+--+         +--+--+--+--+
| a| 0| b| 0| c| 0| d| 0|   →     | a| b| c| d|   (only non-zeros stored)
+--+--+--+--+--+--+--+--+         +--+--+--+--+

Metadata E encodes positions:      E: [00, 10, 00, 10]
(which 2 of 4 are non-zero)            ↑       ↑
                                    positions of a,b in each group
```

**3. Extra SMEM layouts, TMA loads, and allocations** — sparse kernels use dedicated layout helpers for A and E. An additional TMA descriptor loads the metadata into SMEM alongside A and B. In the example here, `E` uses its own logical tile `mma_tiler_e`:

``` python
# Host side: SMEM layouts
a_smem_layout = sm100_utils.make_sparse_smem_layout_a(
    tiled_mma, mma_tiler_mnk, a_raw_dtype, num_stages, sparsity=2,
)
e_smem_layout = sm100_utils.make_sparse_smem_layout_e(
    tiled_mma, mma_tiler_e, a_raw_dtype, num_stages,
)

# Host side: TMA atom for metadata E (note mma_tiler_e and internal_type=Uint64)
a_op = sm100_utils.cluster_shape_to_tma_atom_A(cluster_shape_mn, tiled_mma.thr_id)
tma_atom_e, tma_tensor_e = cute.nvgpu.make_tiled_tma_atom_A(
    a_op,
    sp_e_tensor,
    cute.slice_(e_smem_layout, (None, None, None, 0)),
    mma_tiler_e,
    tiled_mma,
    cluster_layout_vmnk.shape,
    internal_type=cutlass.Uint64,
)

# Kernel side: SMEM allocation for metadata
sE = smem.allocate_tensor(
    element_type=e_sparse_dtype,
    layout=e_smem_layout.outer,
    byte_alignment=128,
    swizzle=e_smem_layout.inner,
)
```

**4. Metadata TMEM allocation and SMEM→TMEM copy (S2T)** — the metadata must live in TMEM for the MMA instruction. It is placed **after** the accumulator columns, and an S2T copy moves it from SMEM to TMEM each K-tile. The example here also recasts both sides to raw `uint8` before building the S2T copy and wraps the SMEM source in an S2T descriptor tensor:

``` python
# TMEM layout for metadata (placed after accumulator)
e_tmem_layout = sm100_utils.make_sparse_tmem_layout_e(
    cute.slice_(e_smem_layout_staged, (None, None, None, 0)).shape,
    a_raw_dtype,
)
acc_tmem_col_offset = tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
if cutlass.const_expr(acc_dtype.width < 32):
    acc_tmem_col_offset = acc_tmem_col_offset * (32 // acc_dtype.width)
e_tmem_ptr = cute.recast_ptr(
    acc_tmem_ptr + acc_tmem_col_offset, dtype=e_sparse_dtype,
)
tCtE = cute.make_tensor(e_tmem_ptr, e_tmem_layout)

# S2T copy setup (SMEM → TMEM for metadata)
e_raw_dtype = cutlass.Uint8
copy_atom_s2t_e = cute.make_copy_atom(
    tcgen05.Cp128x128bOp(cta_group), e_raw_dtype,
)
tCtE_recast = cute.recast_tensor(tCtE, e_raw_dtype)
tiled_copy_s2t_E = tcgen05.make_s2t_copy(copy_atom_s2t_e, tCtE_recast)
thr_copy_s2t_E = tiled_copy_s2t_E.get_slice(0)

sE_recast = cute.recast_tensor(sE, e_raw_dtype)
thr_tCsE_s2t_ = thr_copy_s2t_E.partition_S(sE_recast)
thr_tCsE_s2t = tcgen05.get_s2t_smem_desc_tensor(
    tiled_copy_s2t_E, thr_tCsE_s2t_
)
thr_tCtE_s2t = thr_copy_s2t_E.partition_D(tCtE_recast)
```

**5. Modified main loop** — each K-tile iteration loads the metadata via S2T, then sets the `METADATA` field on the atom before calling `gemm`. The `gemm` call signature itself is unchanged; the metadata is implicit via the atom field. The full kernel also contains leader-CTA synchronization and optional metadata reuse when `utccp_reuse_cnt > 1`; the schematic below keeps only the dataflow-relevant steps:

``` python
tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

for k_tile in range(k_tile_cnt):
    # S2T: move metadata for the current stage from SMEM to TMEM.
    cute.copy(tiled_copy_s2t_E, e_smem_stage, e_tmem_stage)

    for kblk_idx in cutlass.range(cute.size(tCrA, mode=[2]), unroll_full=True):
        e_idx = metadata_index_for(k_tile, kblk_idx)
        tiled_mma.set(tcgen05.Field.METADATA, tCtE[None, None, e_idx].iterator)
        cute.gemm(tiled_mma, tCtAcc, tCrA_kblk, tCrB_kblk, tCtAcc)
        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
```

``` text
Dense main loop (per K-tile):
  set(ACCUMULATE, ...)
  gemm(tiled_mma, tCtAcc, tCrA[s], tCrB[s], tCtAcc)

Sparse main loop (schematic, per K-tile):
  copy(s2t_E, sE[stage], tCtE)               ← metadata SMEM → TMEM
  set(METADATA, tCtE[e_idx].iterator)        ← point atom at metadata
  set(ACCUMULATE, ...)
  gemm(tiled_mma, tCtAcc, tCrA[s], tCrB[s], tCtAcc)   ← same call
```

The epilogue (TMEM → RMEM → GMEM) is identical to a dense kernel.

#### Block-scaled MMA

Block-scaled MMA multiplies narrow-type matrices (the tcgen05 MXF8 and MXF4-family ops shown here) with **per-block scale factors** along GEMM-K. Each `sf_vec_size` consecutive K-elements shares one scale factor, so the hardware computes `D = (SFA · A) * (SFB · B) + C`. Unlike dense A/B, SFA/SFB must be staged in **TMEM** before `gemm`.

Supported ops: `MmaMXF8F6F4Op`, `MmaMXF4Op`, `MmaMXF4NVF4Op`.

Compared to a dense MMA kernel, a block-scaled kernel differs in five areas:

**1. MMA op creation** — block-scaled ops fix the accumulator to FP32 and add scale-factor typing. The examples usually build `TiledMma` through `sm100_utils.make_blockscaled_trivial_tiled_mma(...)`, which dispatches to `MmaMXF8F6F4Op`, `MmaMXF4Op`, or `MmaMXF4NVF4Op` from `(ab_dtype, sf_vec_size)`:

``` python
# Direct op examples
mxf8_op = tcgen05.MmaMXF8F6F4Op(
    cutlass.Float8E4M3FN, (128, 256, 32),
    tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
    cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
)

# MXF4/NVF4 example (MmaMXF4Op is the sf_vec_size=32 companion)
nvf4_op = tcgen05.MmaMXF4NVF4Op(
    cutlass.Float8E8M0FNU,     # sf_dtype: UE8M0 or UE4M3
    (128, 256, 64),
    tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
)

# Helper used by the block-scaled examples here
tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
    ab_dtype, a_major_mode, b_major_mode, sf_dtype, sf_vec_size,
    cta_group, mma_tiler_mn,
)
```

**2. Extra scale-factor tensors and SMEM layouts** — derive SFA/SFB tensors from the A/B shapes, then build staged SMEM layouts for them:

``` python
import cutlass.utils.blockscaled_layout as blockscaled_utils

# Scale-factor tensors
sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, sf_vec_size)
sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)
sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b_tensor.shape, sf_vec_size)
sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)

# Staged SMEM layouts
sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
    tiled_mma, mma_tiler_mnk, sf_vec_size, num_ab_stage,
)
sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
    tiled_mma, mma_tiler_mnk, sf_vec_size, num_ab_stage,
)
```

**3. Extra TMA loads and SMEM allocations** — there are four GMEM→SMEM loads instead of two. SFA follows the A-side TMA path; SFB uses `cluster_shape_to_tma_atom_SFB(...)` and may use its own tiler/layout in 2CTA kernels. The pipeline byte count also includes the SFA/SFB traffic:

``` python
# TMA atoms for SFA/SFB (note internal_type=Int16 for packing)
sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(cluster_shape_mn, tiled_mma.thr_id)
tma_sfa = cute.nvgpu.make_tiled_tma_atom_A(
    sfa_op, sfa_tensor, sfa_smem_layout_staged,
    mma_tiler_mnk, tiled_mma, cluster_layout_vmnk.shape,
    internal_type=cutlass.Int16,
)

sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(cluster_shape_mn, tiled_mma.thr_id)
tma_sfb = cute.nvgpu.make_tiled_tma_atom_B(
    sfb_op, sfb_tensor, sfb_smem_layout_staged,
    mma_tiler_sfb, tiled_mma_sfb, cluster_layout_sfb_vmnk.shape,
    internal_type=cutlass.Int16,
)

# Kernel side: allocate staged SMEM for scale factors
sSFA = smem.allocate_tensor(element_type=sf_dtype, layout=tma_sfa.smem_layout, ...)
sSFB = smem.allocate_tensor(element_type=sf_dtype, layout=tma_sfb.smem_layout, ...)
```

**4. Scale-factor TMEM allocation and SMEM→TMEM copy (S2T)** — before each `gemm`, SFA/SFB must be copied from staged SMEM into TMEM. The examples compact away zero-stride modes and wrap the SMEM source in an S2T descriptor tensor:

``` python
# TMEM allocation for scale factors
tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
    tiled_mma, mma_tiler_mnk, sf_vec_size,
    cute.slice_(tma_sfa.smem_layout, (None, None, None, 0)),
)
tCtSFA = tmem_pool.allocate_tensor(tCtSFA_layout, sf_dtype)
tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
    tiled_mma, mma_tiler_mnk, sf_vec_size,
    cute.slice_(tma_sfb.smem_layout, (None, None, None, 0)),
)
tCtSFB = tmem_pool.allocate_tensor(tCtSFB_layout, sf_dtype)

# S2T copy setup (SMEM → TMEM for scale factors)
copy_atom_s2t = cute.make_copy_atom(
    tcgen05.Cp4x32x128bOp(cta_group), sf_dtype,
)

# SFA shown; SFB follows the same pattern.
tCtSFA_compact = cute.filter_zeros(tCtSFA)
tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFA_compact)
thr_copy_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
tCsSFA_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
    tiled_copy_s2t_sfa,
    thr_copy_s2t_sfa.partition_S(cute.filter_zeros(sSFA)),
)
tCtSFA_compact_s2t = thr_copy_s2t_sfa.partition_D(tCtSFA_compact)
# Repeat with tCtSFB / sSFB to produce:
# tiled_copy_s2t_sfb, tCsSFB_compact_s2t, and tCtSFB_compact_s2t.
```

**5. Modified main loop** — per K-tile, load A/B/SFA/SFB into SMEM, copy SFA/SFB to TMEM, then call `gemm` with `[value, scale]` operands. The persistent kernel separates TMA and MMA warps; the tutorial-style loop below keeps only the operand flow:

``` python
for k_tile in cutlass.range(num_k_tiles):
    # TMA load A, B, SFA, SFB into SMEM
    cute.copy(tma_a.atom,   tAgA[None, ab_empty.count],   tAsA[None, ab_empty.index],   ...)
    cute.copy(tma_b.atom,   tBgB[None, ab_empty.count],   tBsB[None, ab_empty.index],   ...)
    cute.copy(tma_sfa.atom, tAgSFA[None, ab_empty.count], tAsSFA[None, ab_empty.index], ...)
    cute.copy(tma_sfb.atom, tBgSFB[None, ab_empty.count], tBsSFB[None, ab_empty.index], ...)

    ab_full = ab_consumer.wait_and_advance()

    # S2T: copy scale factors from SMEM to TMEM
    s2t_stage_coord = (None, None, None, None, ab_full.index)
    cute.copy(
        tiled_copy_s2t_sfa,
        tCsSFA_compact_s2t[s2t_stage_coord],
        tCtSFA_compact_s2t,
    )
    cute.copy(
        tiled_copy_s2t_sfb,
        tCsSFB_compact_s2t[s2t_stage_coord],
        tCtSFB_compact_s2t,
    )

    # MMA with scale factors passed as [value, scale] pairs
    tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
    tile_crd = (None, None, None, ab_full.index)
    cute.gemm(
        tiled_mma,
        tCtAcc,
        [tCrA[tile_crd], tCtSFA],    # A value (SMEM) + A scale (TMEM)
        [tCrB[tile_crd], tCtSFB],    # B value (SMEM) + B scale (TMEM)
        tCtAcc,
    )

    ab_full.release()
```

``` text
Dense tcgen05 mainloop (schematic):
  gemm(tiled_mma, tCtAcc, tCrA[s], tCrB[s], tCtAcc)

Block-scaled tcgen05 mainloop (schematic):
  copy(s2t_sfa, sSFA[stage], tCtSFA)           ← scale A to TMEM
  copy(s2t_sfb, sSFB[stage], tCtSFB)           ← scale B to TMEM
  gemm(tiled_mma, tCtAcc, [tCrA[s], tCtSFA], [tCrB[s], tCtSFB], tCtAcc)
```

The epilogue (TMEM → RMEM → GMEM) is identical to a dense kernel.

See also:

- Tutorial: step-by-step dense F16 GEMM — `examples/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_0.py` (and `fp16_gemm_1.py` through `fp16_gemm_6.py` for progressive optimizations)
- Tutorial: block-scaled NVFP4 GEMM — `examples/cute/blackwell/tutorial/tutorial_gemm/nvfp4_gemm_0.py`
- Dense GEMM (production): `examples/cute/blackwell/kernel/dense_gemm/dense_gemm.py`
- Persistent dense GEMM: `examples/cute/blackwell/kernel/dense_gemm/dense_gemm_persistent.py`
- Block-scaled GEMM: `examples/cute/blackwell/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent.py`
- Sparse GEMM: `examples/cute/blackwell/kernel/sparse_gemm/sparse_gemm_persistent.py`
- Helper utilities: `cutlass.utils.blackwell_helpers`
- Block-scaled layout utilities: `cutlass.utils.blockscaled_layout`

---

<!-- source: limitations.rst -->

## Limitations

### Overview

CuTe DSL is an embedded domain-specific language within Python. It utilizes a subset of Python's syntax to provide a streamlined programming experience. It is important to understand that CuTe DSL does NOT implement the complete Python language semantics in its JIT compilation process.

This section documents the current limitations of the CuTe DSL. While some of these limitations may be addressed in future releases, developers should be aware of them when building applications with the DSL.

### Notable unsupported features

- convolutions
- preferred clusters
- Windows support

### Programming Model

**CuTe Layout Algebra Only support 32bit**
Today, we only support 32bit shapes/strides in CuTe layouts. 64bit or arbitrary width support is planned for future releases.

**Python Native Data Types**
CuTe DSL supports Python data structures when used for "meta-programming," but these structures cannot be treated as dynamic values modifiable at runtime. For instance, lists and dictionaries can be used to configure kernel parameters during compilation or serve as containers for dynamic values, but their structure and organization cannot be altered during kernel execution.

- **Static Values:**
  - Evaluated during JIT compilation phase
  - Immutable after compilation completes
  - Most Python native types (lists, tuples, dictionaries) are processed as static values
  - Primarily utilized for "meta-programming" and configuration purposes
  - Example: Lists can contain dynamic values but their structure cannot be modified during kernel execution

- **Dynamic Values:**
  - Evaluated during runtime execution
  - Modifiable during execution of JIT-compiled functions
  - Only a specific subset of Python types are supported as dynamic values
  - Primitive types are automatically converted when passed as function arguments:
    - `int` → `Int32` (may be updated to `Int64` in future releases)
    - `bool` → `Bool`
    - `float` → `Float32` (may be updated to `Float64` in future releases)

The JIT compiler processes Python native types analogously to C++ template parameters. The compiled code cannot manipulate dynamic values of composite types such as lists, tuples, or dictionaries.

For example, following code doesn't work as traditional Python program inside JIT function.

``` python
@cute.jit
def foo(a: Float32, b: Float32, i: Int32, res: cute.Tensor):
    xs = [a, b]
    # indexing list with dynamic index is not supported in CuTe DSL:
    res[0] = xs[i]

    if i == 0:
        # This will alway append Float32(3.0) to the list regardless
        # of the runtime value of `i`
        xs.append(Float32(3.0))

    for i in range(10):
        # This only append one element to the list at compile-time
        # as loop doesn't unroll at compile-time
        xs.append(Float32(1.0))
```

**Python Function**
The DSL currently has **limited support for return values** from Python functions. At the moment, only `constexpr` values can be returned, while returning **dynamic values** is **not yet supported**. This capability is planned for a future release.

Example:

``` python
@cute.jit
def baz(a: cutlass.Constexpr):
    return a + 1

@cute.jit
def foo(a: cutlass.Int32):
    return a + 1

@cute.jit
def bar(a: cutlass.Int32):
    val = foo(a)  # works

val = baz(10)   # works
val = bar(10)   # works
foo(10)         # currently unsupported in CuTe DSL
```

**Expression or Statement with Dependent Type**
CuTe DSL implements static typing and does not support dependent types. The type of each expression must be determinable during compile time, in contrast to standard Python which implements dynamic typing.

Example illustrating functionality in Python that is not supported in the DSL:

``` python
# Valid in standard Python, but unsupported in CuTe DSL
max(int(1), float(2.0))  # => 2.0 : float
max(int(3), float(2.0))  # => 3   : int
```

In CuTe DSL, types are promoted. For example:

``` python
@cute.jit
def foo(a: Int32, b: Float32, res: cute.Tensor):
    res[0] = max(a, b)  # Type is automatically promoted to Float32
```

Following code using inlined if-else expression with dependent types is not supported in CuTe DSL:

``` python
@cute.jit
def foo(cond: Boolean, a: Int32, b: Float32, res: cute.Tensor):
    res[0] = a if cond else b
```

**Control Flow**
The DSL transforms Python control flow statements (`if`, `for`, `while`) during Abstract Syntax Tree (AST) processing into structured control flow in MLIR which has the same constraints as dependent types. For instance, changing type of a variable in loop body is not allowed.

- Variables must be defined prior to the control flow statement
- Type consistency must be maintained throughout the control flow statement
- Don't support early exit or return from if-else statements

Example illustrating functionality in Python that is not supported in the DSL:

``` python
@cute.jit
def foo():
    a = Int32(1)
    for i in range(10):
        a = Float32(2)  # Changing type inside loop-body is not allowed in the DSL
```

**Built-in Operators**
The DSL transforms built-in operators like `and`, `or`, `max`, `min`, etc. into MLIR operations. They also follow the same constraints of dependent types. For instance, `a and b` requires `a` and `b` to be of the same type.

**Special Variables**
The DSL treats `_` as a special variable that it's value is meant to be ignored. It is not allowed to read `_` in the DSL.

Example illustrating functionality in Python that is not supported in the DSL:

``` python
@cute.jit
def foo():
    _ = 1
    print(_)  # This is not allowed in the DSL
```

**Object Oriented Programming**
The DSL is implemented on top of Python and supports Python's object-oriented programming (OOP) features for meta-programming at compile-time.

However, similar to other composed data types, the DSL provides limited support for OOP when objects contain dynamic values. It is strongly recommended to avoid passing dynamic values between member methods through class state in your code.

The following example illustrates functionality in Python that is not supported in the DSL without implementing the `DynamicExpression` protocol:

``` python
class Foo:
    def __init__(self, a: Int32):
        self.a = a

    def set_a(self, i: Int32):
        self.a = i

    def get_a(self):
        return self.a

@cute.jit
def foo(a: Int32, res: cute.Tensor):
    foo = Foo(a)
    for i in range(10):
        foo.set_a(i)

    # This fails to compile because `a` is assigned a local value defined within the for-loop body
    # and is not visible outside of the loop body
    res[0] = foo.get_a()
```

The example above fails to compile because `Foo.a` is assigned a local value defined within the for-loop body, which is not visible outside the loop body.

The CuTe DSL implements an internal mechanism that provides limited support for OOP patterns via protocol. As the DSL continues to evolve to support additional features, this mechanism is subject to change and is not recommended for direct use in users' code for better portability.

**CuTe Layout algebra in native Python**
Entirety of CuTe Layout algebra operations and APIs require JIT compilation. These functionalities are exclusively available within JIT-compiled functions and cannot be accessed in standard Python execution environments.

Additionally, there exists a restricted set of data types that can be passed as arguments to JIT-compiled functions, which further constrains their usage in native Python contexts. Only following CuTe algebra types are supported as JIT function arguments: `Tensor`, `Pointer`, `Shape`, `Stride`, `Coord` and `IntTuple`. For `Stride`, we don't support `ScacledBasis` from native Python Context. Unfortunately, in the first release, we don't support passing `Layout` under native Python Context.

**Block-level Utilities (block_copy)**
The block-level utility `block_copy` provides a high-level abstraction for common copy patterns, but has the following limitations:

**block_copy limitations:**

- **Limited copy op support**: Currently only `TmaCopyOp`-based tiled copies (TMA loads/stores) and S2T copies (SMEM to TMEM, e.g., `tcgen05.Cp*Op`) are supported. Other `TiledCopy` ops will raise `NotImplementedError`. Support for additional copy ops may be added in future releases.

**Global variables**
CuTe DSL does not support global variables. It is not allowed to use `global` in the DSL. The following example illustrates functionality in Python that is not supported in the DSL:

``` python
@cute.jit
def foo():
    global x
    x = 1

foo()
```

The example above fails to compile because `global x` is not supported in the DSL.

**Nonlocal variables**
The use of the `nonlocal` keyword is restricted in CuTe DSL. CuTe DSL does not support capturing variables from an outer (enclosing) scope that is outside of the JIT-compiled function. If you try to use `nonlocal` to refer to a variable defined in Python code that is not tracked by current JIT context, a runtime error will be raised.

``` python
def outer():
    x = 1

    @cute.jit
    def inner():
        nonlocal x  # Not supported
        x = 2

    inner()
```

The above code will fail with a runtime error because `x` is defined in a scope not managed by the CuTe DSL's JIT compilation. Nonlocal variables must be managed within the same JIT context; otherwise, a runtime error will be raised.

#### Suggestions

For reliable and predictable results:

- Avoid dependent types in your code
- Implement explicit type conversion for dynamic values
- Clearly distinguish between static (compile-time) and dynamic (runtime) values
- Use type annotations as much as possible to help JIT compiler to identify type to avoid ambiguity

``` python
# Example demonstrating explicit typing
alpha = 1.0  # Explicitly defined as float using `1.0` instead of `1`
             #  or `float(1)`
beta = 2.0   # Explicitly defined as float
result = max(alpha, beta)  # Will correctly perform float comparison
```

**Debugging Capabilities**
Debugging tools and facilities for the Python DSL are currently more limited in comparison to the C++ API. For instance, we don't support single-stepping through the JIT-compiled code. And lack of exception handling in JIT-compiled code makes it hard to debug in some cases.

**Integration with Frameworks**
Integration with certain deep learning frameworks is in early development stages and may have limitations. For instance, converting frameworking tensor to cute.Tensor is known to have overhead with 2us~3us per tensor as we convert from general DLPack protocol which offers comptibility with all frameworks.

**Hashing DSL APIs and Objects**
DSL APIs and Objects are sensitive to MLIR context, region or other contextual information which has no meaning cross different context. Any stateful design rely on `__hash__` likely misbehave with unexpected results. An example is `functools.lru_cache`, which combined with `@cute.jit`, it may cache MLIR object from one context and use in another one.

### Future Improvements

The CuTe DSL development team is actively addressing these limitations. Upcoming releases will aim to:

- Implement support for return values from JIT compiled functions
- Improve support for built-in operators to handle more cases without dependent types
- Enhance debugging capabilities and tools
- Improve error messages with precise diagnostic information
- Extend support for additional numeric data types
- Improve performance of converting framework tensor to `cute.Tensor` with native support for different frameworks
- Offer more user friendly benchmarking methodology

### Design Limitations Likely to Remain

The primary objective of CuTe DSL is to provide a domain-specific language for expressing complex CUDA kernels with optimal GPU performance, not to execute arbitrary Python code on GPU hardware.

The following limitations will likely remain by design:

- **Complex Data Structures as Dynamic Values**: Lists, tuples, and dictionaries will continue to function as static containers. While they can store dynamic values, their structure (adding/removing elements) cannot be modified during execution of JIT-compiled functions.
- **Dependent Types**: Supporting dependent types would introduce substantial complexity and adversely affect the performance characteristics of generated code.
- **CuTe Layout Algebra**: We don't have plan to extend the support of CuTe Layout Algebra under native Python Context. We are planning to extend support for data types and allow JIT function to interoperate with native Python code.

---

<!-- source: faqs.rst -->

## FAQs

### General

**Are the DSLs replacing C++ templates?**

> TL;DR: No - but also yes. The CUTLASS 4.0 release (CuTe DSL), along with all future extensions to our Python-native programming models, does not come at the expense of CUTLASS C++. CUTLASS 2.x and 3.x C++ APIs are both going to continue receiving fixes and updates for the architectures we support them for. However, CUTLASS 4.x CuTe DSL is fully isomorphic in its programming model and performance with CuTe C++ for Blackwell, and it is our hope that the community embraces this for much easier while still equally performant custom kernel development. This is why we are releasing CuTe DSL with support for all architectures starting with the NVIDIA Ampere Architecture.

**What is the difference between CuTe DSL, CUTLASS Python, and CUTLASS DSLs?**

> CUTLASS Python was the Python interface for instantiating C++ kernels via a Python frontend. This is now deprecated with the release of CUTLASS 4.0. CUTLASS DSLs are a family of Python DSLs for native device programming in Python. Currently, this is limited to our initial release of CuTe DSL, but future versions will include higher-level abstractions that gradually trade off control for convenience.

**What should I learn, CUTLASS C++ or the Python DSLs?**

> We believe the Python DSLs will significantly improve the learning curve and recommend starting with them for all newcomers, as they eliminate the inherent complexity of learning C++ metaprogramming for GPU kernel programming. Since CuTe C++ and CuTe DSL share fully isomorphic programming models and patterns, any knowledge gained can eventually be applied to C++.

**Where will the code live? PIP wheel or GitHub repo? Do I have to build it myself?**

> This is a major change compared to CUTLASS C++ and Python DSLs. Going forward, the GitHub code only exists as a way for users to file issues and pull requests against. While it can be used with the pip wheel, we do not recommend most users do so unless they are hacking on the DSL itself. For all other users, we recommend they simply `pip install nvidia-cutlass-dsl` and use the pip wheel as the single source of truth for the dialect compiler and DSL implementation. CUTLASS GitHub repository will contain a `requirements.txt` file pinning the version of the wheel consistent with the state of the OSS repository (please see Quick Start). This means getting started with CUTLASS is easier than ever: no more CMake command lines to learn and no more builds to kick off. Simply install the pip wheel and start running the examples.

### Migration

**Should I port my code from C++ templates to Python?**

> Almost certainly not, unless you need extremely fast JIT times for your kernel and C++ compile times are a blocker for you. The 2.x and 3.x APIs will continue to be supported, and Nvidia's Hopper and Blackwell architectures 3.x will continue to improve in terms of features and performance.

**Are portability promises different with Python?**

> For the initial release while the DSL is still in beta, we do not promise any portability as we may make changes to the DSL itself. While we do not expect any changes to the CuTe operations, the DSL utilities, decorators, helper classes like pipelines and schedulers may change as we refine them with community feedback. We encourage users to file issues and discussions on GitHub during this beta period with their feedback!
>
> In the long term, we plan to continue to treat the OSS community with care. Just like the prior history of CUTLASS, we plan not to break users unless necessary, but we reserve the right to make limited breaking changes in case we believe it is a net benefit to the community and project. These will be announced ahead of time and/or clearly highlighted in the CHANGELOG of each release.

### Technical

**What NVIDIA architectures will it support?**

> CuTe DSL will support all NVIDIA GPU architectures starting with NVIDIA Ampere Architecture (SM80).

**Will it be compatible with DL frameworks (e.g., PyTorch, JAX)?**

> Yes, we will provide utilities to convert from DLPack-supported tensor formats to `cute.Tensor`. This should allow a user to never have to leave Python when writing model code in their framework of choice. Our JAX interoperability story is not as strong as PyTorch's today, however, we are actively working on improving it and welcome contributions in this space.

**Does it compile to PTX or SASS?**

> CuTe DSL compiles the program down to PTX. After that, we currently use the PTX compiler that ships with the CUDA toolkit to compile the PTX down to SASS. We plan to remove this limitation in the future and allow the use of the PTX JIT that is included in the CUDA driver in case a user does not have a CUDA toolkit installed.

**Do I need to use NVCC or NVRTC?**

> No, the `nvidia-cutlass-dsl` wheel packages is everything needed to generate GPU kernels. It shares the driver requirements of the 12.9 toolkit which can be found [here](https://developer.nvidia.com/cuda-toolkit-archive).

**How would one debug the code?**

> Since CuTe DSL is not native python and an embedded DSL instead, tools like `pdb` cannot be used. However, if you have experience with GPU kernel programming, the debugging techniques will be nearly identical. Typically, compile time and runtime printing of types and values are the most expedient. Please see [documentation on printing](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/print.ipynb) to learn how to print types and values at both compile time and runtime. You can also use `cuda-gdb` to set breakpoints in the program and step through the execution or use tools such as `compute-sanitizer` to detect and triage bugs in your program. As the DSL matures, our source location tracking from Python user programs will also improve to provide more helpful source-level mapping when setting breakpoints and using other tools such as nsight.

**How would one implement warp specialization in CuTe DSL?**

> Exactly the same way you would in C++ but in a Python-native syntax instead. Consult our Control Flow and ["Blackwell kernel example"](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py) for a detailed how-to guide.

**Can I call functions from other functions or use OOP?**

> Yes. We frequently call functions from one another and set up class hierarchies to organize and modularize our code for pipelines and schedulers. Consult the Introduction documentation or our examples for more details.

### License

**What is the license for CuTe DSL and the associated GitHub samples?**

> CuTe DSL components available [on Github](https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL) and via the nvidia-cutlass-dsl Python pip wheel are released under the ["NVIDIA Software End User License Agreement (EULA)"](https://github.com/NVIDIA/cutlass/tree/main/EULA.txt). Because the pip package includes a compiler that shares several components with the CUDA Toolkit, it is subject to usage terms and restrictions similar to those of the CUDA SDK. Please refer to the EULA for specific terms of use.
>
> CuTe DSL samples and Jupyter notbooks, released [on GitHub](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL) are provided under the BSD 3-Clause License and may be used and redistributed under those terms. This distinction ensures that developers have flexibility when using or modifying the code samples, independent of the compiler and runtime components governed by the EULA.
>
> If you have any questions or need clarification, feel free to contact us.

---

<!-- source: cute_dsl_api/changelog.rst -->

## Changelog for CuTe DSL API changes

### [4.4.0](https://github.com/NVIDIA/cutlass/releases/tree/main) (2026-03-24)

- Added native support for `typing.NamedTuple` as a JIT function argument.
  - A NamedTuple whose fields are DSL scalar types (`Int32`, `Float32`, …) can be passed directly to `@cute.jit` / `cute.compile` without any protocol implementation.
  - Fields are flattened field-by-field through the existing pytree system and reconstructed via the NamedTuple constructor on entry to the kernel body. Field attribute access (`tup.a`, `tup.b`, …) works as in native Python.
  - NamedTuple fields are **immutable** (tuple subclass). To replace a field, construct a new NamedTuple inside the kernel. Use `@native_struct` when mutable fields are required.
  - See Struct-like JIT Arguments for a guide to NamedTuple, `@native_struct`, and other struct-like JIT argument types.

### [4.3.0](https://github.com/NVIDIA/cutlass/releases/tree/main) (2025-10-20)

- Debuggability improvements:
  - Supported source location tracking for DSL APIs
  - Supported dumping PTX and CUBIN
- Removed deprecated `cutlass.<arch>_utils.SMEM_CAPACITY["<arch_str>"]` and `cutlass.utils.ampere_helpers`
- Supported calling nested functions without capturing variables inside dynamic control flow
- Replaced usage of `cute.arch.barrier` in examples with corresponding APIs in `pipeline`
  - Use `pipeline.sync` for simple cases like synchronizing the whole CTA
  - Use `pipeline.NamedBarrier` to customize barriers with different participating threads and barrier id
- Added new APIs `repeat` and `repeat_as_tuple`
- Added new APIs `make_rmem_tensor` to create tensor in register memory (replace `make_fragment` with better naming)
- Added new APIs `make_rmem_tensor_like` which create rmem tensor from a tensor using the same shape with compact col-major strides
- Added `TmemAllocator` for allocating tensor memory
- Updated `SmemAllocator.allocate` to support allocation of a single scalar value
- Fixed `TensorSSA.reduce` to support static value as initial value
- Updated docstring for following APIs to be more concise and easier to understand:
  - `make_layout_tv`
  - `is_static`
  - `PipelineAsync`
  - `SmemAllocator`
- Fixed documentation for `pipeline`, `utils` and `cute.math` (`cute.math` is part of top level documentation)

### [4.2.0](https://github.com/NVIDIA/cutlass/releases/tag/v4.2.0) (2025-09-10)

- Added back `cute.make_tiled_copy` per the request from community
- Added support for explicit and implicit broadcast in `TensorSSA`
  - `cutlass.cute.TensorSSA`: support `broadcast_to` and implicit broadcasting for binary operations.
- Supported printing `TensorSSA` value in `cutlass.cute.print_tensor`
- Updated `cute.gemm` to support all dispatch patterns and improved checks for illegal inputs
- Introduced automatic kernel smem usage calculation for launch config.
- Introduced per op fast-math control for math ops(e.g. `exp`, `exp2`, `log2`, `log`)
- Introduced `CopyReduceBulkTensorTileS2GOp` in [tcgen05/copy.py](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/copy.py) to support TMA Reduce.

### [4.1.0](https://github.com/NVIDIA/cutlass/releases/tag/v4.1.0) (2025-07-16)

- for loop
  - Python built-in `range` now always generates codes and executes at runtime
  - `cutlass.range` is advanced `range` with kernel code level unrolling and pipelining control
  - Deprecated `cutlass.range_dynamic`, please replace with `range` or `cutlass.range`
  - **Experimental** Added `pipelining` control for compiler generated software pipeline code
- while/if
  - `while`/`if` now by default generates codes and executes at runtime unless `cutlass.const_expr` is specified for the predicate
  - Deprecated `cutlass.dynamic_expr`, please remove it
- Rename mbarrier functions to reduce ambiguity
- Modify SyncObject API (`MbarrierArray`, `NamedBarrier`, `TmaStoreFence`) to match `std::barrier`
- Change pipeline `create` function to take only keyword arguments, and make `barrier_storage` optional.
- Introduce `cutlass.cute.arch.get_dyn_smem_size` api to get runtime dynamic shared memory size.
- Various API Support for SM100 BlockScaled Gemm
  - Introduce BlockScaled MmaOps in [tcgen05/mma.py](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/mma.py), and provide a `make_blockscaled_trivial_tiled_mma` function in [blackwell_helpers.py](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/utils/blackwell_helpers.py) to help construct a BlockScaled TiledMma.
  - Introduce S2T CopyOps in [tcgen05/copy.py](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/copy.py).
  - Introduce BlockScaled layout utilities in [blockscaled_layout.py](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/utils/blockscaled_layout.py) for creating the required scale factor layouts in global memory, shared memory and tensor memory.
- `cutlass.cute.compile` now supports compilation options. Refer to [JIT compilation options](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_jit_compilation_options.html) for more details.
- `cutlass.cute.testing.assert_` now works for device JIT function. Specify `--enable-assertions` as compilation option to enable.
- `cutlass.cute.make_tiled_copy` is now deprecated. Please use `cutlass.cute.make_tiled_copy_tv` instead.
- Shared memory capacity query
  - Introduce `cutlass.utils.get_smem_capacity_in_bytes` for querying the shared memory capacity.
  - `<arch>_utils.SMEM_CAPACITY["<arch_str>"]` is now deprecated.

### [4.0.0](https://github.com/NVIDIA/cutlass/releases/tag/v4.0.0) (2025-06-03)

- Fixed API mismatch in class `cute.runtime.Pointer`: change `element_type` to `dtype` to match `typing.Pointer`
