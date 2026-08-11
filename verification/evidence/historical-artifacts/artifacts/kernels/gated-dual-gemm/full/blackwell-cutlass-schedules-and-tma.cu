// Extracted from sources/blogs/tflops-gap-fp4-moe.md by scripts/extract_blog_code.py
// Heading: # TFLOPS Gap: Why FP4 MoE Kernel Engineering Matters on Blackwell > ## Three Key Optimization Techniques > ### 2. Blackwell-Specific CUTLASS Schedules and TMA
// Original fence languages: <unlabeled> and cuda
// See artifacts/blogs/tflops-gap-fp4-moe/code/PROVENANCE.yaml for origin + license metadata.

// --- Blackwell-optimized CUTLASS schedule (from the unlabeled config fence) ---
// SGLang selects this kernel schedule + thread-block shape for Blackwell NvFP4 MoE:
//
//   KernelSchedule  = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100
//   ThreadBlockShape = Shape<_128, _128, _128>
//
// Key features of this schedule:
//   * Warp Specialization for FP4: dedicated warp roles for loading FP4 data,
//     dequantizing to FP16/BF16, and accumulating in FP32.
//   * TMA Integration: asynchronous bulk tensor loads bypassing L1 cache,
//     feeding directly into shared memory with strict 128-byte alignment.
//   * 1 SM Grouping: multiple experts processed per SM rather than
//     one-expert-per-SM, better for variable expert sizes.
//   * Native NvFP4 Support: hardware FP4 instructions instead of software
//     emulation.

// --- TMA alignment padding (from the cuda-labeled fence) ---
// TMA alignment is enforced by padding blockscale offsets to 128-byte boundaries:
blockscale_offsets[expert_id + 1] = (expert_offsets[expert_id + 1] + 127) / 128 * 128;
