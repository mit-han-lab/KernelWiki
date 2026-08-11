// provenance: derived from pr-vllm-23696, kernel-gated-dual-gemm, technique-epilogue-fusion; not upstream code
// origin: wiki/kernels/gated-dual-gemm.md Phase 3 variant

// Gated dual GEMM fused epilogue: gate_out = silu(gate_gemm_out) * up_gemm_out
// Two GEMMs share the same A operand; their outputs are combined in the
// epilogue warpgroup without touching global memory between them.

#include <cuda_fp16.h>
#include <cuda_fp8.h>

template <int TILE_M, int TILE_N>
__device__ void gated_dual_gemm_epilogue(
    float gate_acc[TILE_M][TILE_N],   // accumulator from gate GEMM
    float up_acc[TILE_M][TILE_N],     // accumulator from up GEMM
    __half* out)
{
    for (int m = 0; m < TILE_M; ++m) {
        for (int n = 0; n < TILE_N; ++n) {
            float g = gate_acc[m][n];
            // SiLU(x) = x * sigmoid(x)
            float silu_g = g * (1.f / (1.f + __expf(-g)));
            out[m * TILE_N + n] = __float2half(silu_g * up_acc[m][n]);
        }
    }
}
