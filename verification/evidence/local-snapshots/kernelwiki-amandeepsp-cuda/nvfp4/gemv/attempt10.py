"""
NVFP4 Block-Scaled GEMV - Attempt 10

╔══════════════════════════════════════════════════════════════════════════════╗
║  ❌ MAJOR REGRESSION - Multiple Accumulators (ILP)                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Performance Results (vs Attempt 7 baseline):                                 ║
║    M=7168, K=16384, L=1  →  41.4 µs  (+55% regression)                        ║
║    M=4096, K=7168,  L=8  →  59.7 µs  (+32% regression)                        ║
║    M=7168, K=2048,  L=4  →  22.5 µs  (+37% regression)                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  What was tried:                                                              ║
║    - 4 independent accumulator chains (acc0, acc1, acc2, acc3)                ║
║    - Goal: Hide FMA latency (~4 cycles) through instruction-level parallelism ║
║    - Stride of THREADS_PER_ROW * 4 instead of THREADS_PER_ROW                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Why it failed BADLY:                                                         ║
║    1. KERNEL IS MEMORY-BOUND, NOT COMPUTE-BOUND                               ║
║       → FMA latency hiding is irrelevant when waiting on memory               ║
║    2. Branch overhead: if (sf < K_sf) checks for each accumulator             ║
║       add warp divergence                                                     ║
║    3. Register pressure: 4 accumulators + intermediates likely cause          ║
║       register spilling to local memory                                       ║
║    4. Reduced coalescing: Stride of 128 instead of 32 reduces memory          ║
║       access efficiency                                                       ║
║    5. Function call overhead: Despite __forceinline__, compute_tile           ║
║       abstraction may hurt performance                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Key insight:                                                                 ║
║    The original #pragma unroll 4 in attempt 7 already provides sufficient     ║
║    ILP at the instruction level. The compiler unrolls the loop, and MEMORY    ║
║    LATENCY (not compute latency) is the actual bottleneck.                    ║
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
#define NUM_ACCUMULATORS 4

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

__device__ __forceinline__ float compute_tile(
    const uint8_t* row_a,
    const uint8_t* batch_b,
    const uint8_t* row_sfa,
    const uint8_t* batch_sfb,
    int sf
) {
    float scale = decode_fp8(static_cast<int8_t>(__ldg(&row_sfa[sf]))) *
                  decode_fp8(static_cast<int8_t>(__ldg(&batch_sfb[sf])));
    __half scale_h = __float2half(scale);
    __half2 scale_h2 = __halves2half2(scale_h, scale_h);

    int byte_base = sf << 3;  // sf * 8 using bit shift

    uchar4 a4_0 = *reinterpret_cast<const uchar4*>(&row_a[byte_base]);
    uchar4 b4_0 = *reinterpret_cast<const uchar4*>(&batch_b[byte_base]);
    uchar4 a4_1 = *reinterpret_cast<const uchar4*>(&row_a[byte_base + 4]);
    uchar4 b4_1 = *reinterpret_cast<const uchar4*>(&batch_b[byte_base + 4]);

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
    const uint8_t* row_a,
    const uint8_t* batch_b,
    const uint8_t* row_sfa,
    const uint8_t* batch_sfb,
    int K_sf,
    int tid
) {
    const int THREADS_PER_ROW = BLOCK_SIZE / ROWS_PER_BLOCK;
    const int STRIDE = THREADS_PER_ROW * NUM_ACCUMULATORS;

    // Multiple accumulators to hide FMA latency (~4 cycles)
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

#pragma unroll 2
    for (int sf_base = tid; sf_base < K_sf; sf_base += STRIDE) {
        int sf0 = sf_base;
        int sf1 = sf_base + THREADS_PER_ROW;
        int sf2 = sf_base + THREADS_PER_ROW * 2;
        int sf3 = sf_base + THREADS_PER_ROW * 3;

        // Independent accumulations - compiler can issue these in parallel
        if (sf0 < K_sf) acc0 += compute_tile(row_a, batch_b, row_sfa, batch_sfb, sf0);
        if (sf1 < K_sf) acc1 += compute_tile(row_a, batch_b, row_sfa, batch_sfb, sf1);
        if (sf2 < K_sf) acc2 += compute_tile(row_a, batch_b, row_sfa, batch_sfb, sf2);
        if (sf3 < K_sf) acc3 += compute_tile(row_a, batch_b, row_sfa, batch_sfb, sf3);
    }

    return acc0 + acc1 + acc2 + acc3;
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

    const uint8_t* row_a = batch_a + static_cast<size_t>(m) * K_half;
    const uint8_t* row_sfa = batch_sfa + static_cast<size_t>(m) * K_sf;

    float acc = compute_row_sum(
        row_a,
        batch_b,
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
    name='batched_scaled_gemv_v10',
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

