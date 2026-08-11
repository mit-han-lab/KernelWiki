"""
NVFP4 Block-Scaled GEMV - Attempt 9

╔══════════════════════════════════════════════════════════════════════════════╗
║  ❌ REGRESSION - Wider Vectorized Loads (uint2)                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Performance Results (vs Attempt 7 baseline):                                 ║
║    M=7168, K=16384, L=1  →  30.9 µs  (+16% regression)                        ║
║    M=4096, K=7168,  L=8  →  53.3 µs  (+18% regression)                        ║
║    M=7168, K=2048,  L=4  →  20.5 µs  (+25% regression)                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  What was tried:                                                              ║
║    - Changed from two uchar4 loads to single uint2 load (64-bit)              ║
║    - Goal: Reduce instruction count and maximize bandwidth utilization        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Why it failed:                                                               ║
║    1. Byte extraction overhead: Bitwise operations (& 0xFF, >> 8, etc.)       ║
║       to unpack uint2 into individual bytes                                   ║
║    2. make_uchar4() calls add instruction overhead                            ║
║    3. The compiler already optimizes two consecutive uchar4 loads efficiently ║
║    4. Direct uchar4 loads via *reinterpret_cast<const uchar4*> are highly     ║
║       optimized by the compiler                                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Lesson learned:                                                              ║
║    Wider loads only help when data can be used directly without unpacking.    ║
║    For FP4 processing requiring byte-level access, native uchar4 is optimal.  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from torch.utils.cpp_extension import load_inline
from task import input_t, output_t


cpp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c);
"""

cuda_source = """
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 128
#define ROWS_PER_BLOCK 4

__device__ __forceinline__ __half2 decode_fp4x2(uint8_t byte) {
    __half2_raw raw = __nv_cvt_fp4x2_to_halfraw2(
        static_cast<__nv_fp4x2_storage_t>(byte),
        __NV_E2M1
    );
    return *reinterpret_cast<__half2*>(&raw);
}

__device__ __forceinline__ float decode_fp8(int8_t byte) {
    __nv_fp8_storage_t storage = static_cast<__nv_fp8_storage_t>(byte);
    __half_raw raw = __nv_cvt_fp8_to_halfraw(storage, __NV_E4M3);
    return __half2float(__ushort_as_half(raw.x));
}

// Process 8 bytes (16 FP4 values) from a uint2 load
__device__ __forceinline__ float dot_scaled_8bytes(
    uint2 a8,
    uint2 b8,
    __half2 scale_h2
) {
    // Extract bytes from uint2
    uchar4 a4_0 = make_uchar4(
        a8.x & 0xFF,
        (a8.x >> 8) & 0xFF,
        (a8.x >> 16) & 0xFF,
        (a8.x >> 24) & 0xFF
    );
    uchar4 a4_1 = make_uchar4(
        a8.y & 0xFF,
        (a8.y >> 8) & 0xFF,
        (a8.y >> 16) & 0xFF,
        (a8.y >> 24) & 0xFF
    );
    uchar4 b4_0 = make_uchar4(
        b8.x & 0xFF,
        (b8.x >> 8) & 0xFF,
        (b8.x >> 16) & 0xFF,
        (b8.x >> 24) & 0xFF
    );
    uchar4 b4_1 = make_uchar4(
        b8.y & 0xFF,
        (b8.y >> 8) & 0xFF,
        (b8.y >> 16) & 0xFF,
        (b8.y >> 24) & 0xFF
    );

    // Process first 4 bytes (8 FP4 values)
    __half2 acc0 = __hmul2(decode_fp4x2(a4_0.x), __hmul2(decode_fp4x2(b4_0.x), scale_h2));
    __half2 acc1 = __hmul2(decode_fp4x2(a4_0.y), __hmul2(decode_fp4x2(b4_0.y), scale_h2));
    acc0 = __hfma2(decode_fp4x2(a4_0.z), __hmul2(decode_fp4x2(b4_0.z), scale_h2), acc0);
    acc1 = __hfma2(decode_fp4x2(a4_0.w), __hmul2(decode_fp4x2(b4_0.w), scale_h2), acc1);
    
    // Process second 4 bytes (8 FP4 values)
    acc0 = __hfma2(decode_fp4x2(a4_1.x), __hmul2(decode_fp4x2(b4_1.x), scale_h2), acc0);
    acc1 = __hfma2(decode_fp4x2(a4_1.y), __hmul2(decode_fp4x2(b4_1.y), scale_h2), acc1);
    acc0 = __hfma2(decode_fp4x2(a4_1.z), __hmul2(decode_fp4x2(b4_1.z), scale_h2), acc0);
    acc1 = __hfma2(decode_fp4x2(a4_1.w), __hmul2(decode_fp4x2(b4_1.w), scale_h2), acc1);

    __half2 result = __hadd2(acc0, acc1);
    float2 f = __half22float2(result);
    return f.x + f.y;
}

__device__ __forceinline__ float compute_row_sum(
    const uint2* __restrict__ row_a,    // Now typed as uint2* for wider loads
    const uint2* __restrict__ batch_b,
    const uint8_t* __restrict__ row_sfa,
    const uint8_t* __restrict__ batch_sfb,
    int K_sf,
    int tid
) {
    float acc = 0.0f;
    const int THREADS_PER_ROW = BLOCK_SIZE / ROWS_PER_BLOCK;

#pragma unroll 4
    for (int sf = tid; sf < K_sf; sf += THREADS_PER_ROW) {
        float scale = decode_fp8(static_cast<int8_t>(__ldg(&row_sfa[sf]))) *
                      decode_fp8(static_cast<int8_t>(__ldg(&batch_sfb[sf])));
        __half scale_h = __float2half(scale);
        __half2 scale_h2 = __halves2half2(scale_h, scale_h);

        // Single 64-bit load per matrix (instead of two 32-bit loads)
        uint2 a8 = __ldg(&row_a[sf]);
        uint2 b8 = __ldg(&batch_b[sf]);

        acc += dot_scaled_8bytes(a8, b8, scale_h2);
    }

    return acc;
}

__global__ void gemv_nvfp4_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const int8_t* __restrict__ sfa,
    const int8_t* __restrict__ sfb,
    half* __restrict__ c,
    int M, int K, int L,
    int N_rows
) {
    int m_start = blockIdx.x * ROWS_PER_BLOCK;
    int l = blockIdx.y;
    int tid = threadIdx.x;
    int row_in_block = tid / (BLOCK_SIZE / ROWS_PER_BLOCK); 
    int tid_in_row = tid % (BLOCK_SIZE / ROWS_PER_BLOCK);

    if (m_start >= M || l >= L) return;

    const int K_sf = K / 16;
    const int K_half = K / 2;
    const size_t batch_stride_a = static_cast<size_t>(M) * K_half;
    const size_t batch_stride_b = static_cast<size_t>(N_rows) * K_half;
    const size_t batch_stride_sfa = static_cast<size_t>(M) * K_sf;
    const size_t batch_stride_sfb = static_cast<size_t>(N_rows) * K_sf;

    const uint8_t* base_a = reinterpret_cast<const uint8_t*>(a);
    const uint8_t* base_b = reinterpret_cast<const uint8_t*>(b);
    const uint8_t* base_sfa = reinterpret_cast<const uint8_t*>(sfa);
    const uint8_t* base_sfb = reinterpret_cast<const uint8_t*>(sfb);

    const uint8_t* batch_a = base_a + l * batch_stride_a;
    const uint8_t* batch_b = base_b + l * batch_stride_b;
    const uint8_t* batch_sfa = base_sfa + l * batch_stride_sfa;
    const uint8_t* batch_sfb = base_sfb + l * batch_stride_sfb;

    // Calculate row index for this thread
    int m = m_start + row_in_block;
    if (m >= M) return;

    // Cast to uint2* for wider loads (8 bytes = 16 FP4 values = 1 scale factor block)
    const uint2* row_a = reinterpret_cast<const uint2*>(batch_a + static_cast<size_t>(m) * K_half);
    const uint2* batch_b_u2 = reinterpret_cast<const uint2*>(batch_b);
    const uint8_t* row_sfa = batch_sfa + static_cast<size_t>(m) * K_sf;

    float acc = compute_row_sum(
        row_a,
        batch_b_u2,
        row_sfa,
        batch_sfb,
        K_sf,
        tid_in_row
    );

    // Each row group (32 threads) reduces independently
    const int lane = tid_in_row;
    float row_sum = acc;

    // Warp reduction within each row group
    row_sum += __shfl_down_sync(0xffffffff, row_sum, 16);
    row_sum += __shfl_down_sync(0xffffffff, row_sum, 8);
    row_sum += __shfl_down_sync(0xffffffff, row_sum, 4);
    row_sum += __shfl_down_sync(0xffffffff, row_sum, 2);
    row_sum += __shfl_down_sync(0xffffffff, row_sum, 1);

    // Thread 0 of each row group writes the result
    if (lane == 0) {
        size_t c_idx = static_cast<size_t>(m) + static_cast<size_t>(l) * M;
        c[c_idx] = __float2half(row_sum);
    }
}

torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c) {
    int M = a.size(0);
    int K = a.size(1) * 2;
    int L = a.size(2);
    int N_rows = b.size(0);

    // Launch grid with M/ROWS_PER_BLOCK blocks
    int grid_m = (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    dim3 grid(grid_m, L);
    dim3 block(BLOCK_SIZE);

    auto* a_ptr = reinterpret_cast<const int8_t*>(a.data_ptr());
    auto* b_ptr = reinterpret_cast<const int8_t*>(b.data_ptr());
    auto* sfa_ptr = reinterpret_cast<const int8_t*>(sfa.data_ptr());
    auto* sfb_ptr = reinterpret_cast<const int8_t*>(sfb.data_ptr());
    auto* c_ptr = reinterpret_cast<half*>(c.data_ptr());

    gemv_nvfp4_kernel<<<grid, block>>>(
        a_ptr,
        b_ptr,
        sfa_ptr,
        sfb_ptr,
        c_ptr,
        M, K, L,
        N_rows
    );

    return c;
}

"""

module = load_inline(
    name='batched_scaled_gemv_v9',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['batched_scaled_gemv_cuda'],
    extra_cuda_cflags=[
        '-O3',
        '--use_fast_math',
        '-std=c++17',
        '-gencode=arch=compute_100a,code=sm_100a',
        '-maxrregcount=80'
    ],
    with_cuda=True,
    verbose=False
)

def custom_kernel(data: input_t) -> output_t:
    a, b, sfa_ref, sfb_ref, _, _, c = data
    return module.batched_scaled_gemv_cuda(
        a,
        b,
        sfa_ref,
        sfb_ref,
        c
    )

