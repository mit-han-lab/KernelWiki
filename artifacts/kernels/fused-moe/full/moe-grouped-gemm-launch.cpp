// KernelWiki illustrative pseudocode; not upstream DeepGEMM or contest code.
// It documents only the varying-M segmentation used by a grouped GEMM.
// It is intentionally incomplete and is not a compilable or timed kernel.

// Grouped-GEMM packs a variable list of per-expert GEMMs into one kernel
// launch via a prefix-sum offset array; layouts are contiguous (M-axis),
// masked (variable-K), or K-grouped depending on router output.
struct GroupedGemmArgs {
    int num_groups;
    int* m_prefix;                    // [num_groups+1]
    const __nv_fp8_e4m3* A;
    const __nv_fp8_e4m3* B;
    const float* scales_a;
    const float* scales_b;
    __half* C;
    int N, K;
};

__global__ void grouped_gemm_launch(GroupedGemmArgs args) {
    int group = blockIdx.y;
    int m_start = args.m_prefix[group];
    int m_end   = args.m_prefix[group + 1];
    // Dispatch a standard tile-level GEMM for [m_start, m_end) × N × K.
}
