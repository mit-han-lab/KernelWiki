"""
NVFP4 Block-Scaled GEMV - Attempt 12

╔══════════════════════════════════════════════════════════════════════════════╗
║  🔬 EXPERIMENTAL - Software Pipelining                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Performance: NOT YET BENCHMARKED                                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  What this attempts:                                                          ║
║    - Software pipelining: prefetch next tile while computing current          ║
║    - Explicit prologue to load first tile into registers                      ║
║    - Pipeline shift: next becomes current at end of each iteration            ║
║    - Goal: Hide memory latency by overlapping loads with computation          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Expected outcome (based on previous attempts):                               ║
║    - Likely SLOWER due to:                                                    ║
║      1. Increased register pressure (holding both current and next data)      ║
║      2. Branch overhead from has_next checks                                  ║
║      3. Kernel is memory-bound - L1/L2 cache already provides prefetching     ║
║      4. __ldg already hints to hardware prefetcher                            ║
║    - The hardware memory subsystem on B200 is already highly optimized        ║
║      for this access pattern                                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Recommendation:                                                              ║
║    Profile with Nsight Compute to verify memory stalls before attempting      ║
║    manual software pipelining. Hardware prefetch is usually sufficient.       ║
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
#define THREADS_PER_ROW (BLOCK_SIZE / ROWS_PER_BLOCK)

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

__device__ __forceinline__ float compute_dot_product(
    uchar4 a4_0, uchar4 a4_1,
    uchar4 b4_0, uchar4 b4_1,
    __half2 scale_h2
) {
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

__device__ __forceinline__ float compute_row_sum_pipelined(
    const uint8_t* __restrict__ row_a,
    const uint8_t* __restrict__ batch_b,
    const uint8_t* __restrict__ row_sfa,
    const uint8_t* __restrict__ batch_sfb,
    int K_sf,
    int tid
) {
    float acc = 0.0f;
    
    // Early exit if no work for this thread
    if (tid >= K_sf) return acc;

    // === PROLOGUE: Load first tile ===
    int sf_curr = tid;
    int byte_base_curr = sf_curr << 3;
    
    // Load current scale factors
    uint8_t sfa_curr = __ldg(&row_sfa[sf_curr]);
    uint8_t sfb_curr = __ldg(&batch_sfb[sf_curr]);
    
    // Load current data
    uchar4 a4_0_curr = *reinterpret_cast<const uchar4*>(&row_a[byte_base_curr]);
    uchar4 b4_0_curr = *reinterpret_cast<const uchar4*>(&batch_b[byte_base_curr]);
    uchar4 a4_1_curr = *reinterpret_cast<const uchar4*>(&row_a[byte_base_curr + 4]);
    uchar4 b4_1_curr = *reinterpret_cast<const uchar4*>(&batch_b[byte_base_curr + 4]);

    // === MAIN LOOP with software pipelining ===
#pragma unroll 4
    for (int sf = tid; sf < K_sf; sf += THREADS_PER_ROW) {
        int sf_next = sf + THREADS_PER_ROW;
        bool has_next = (sf_next < K_sf);
        
        // Prefetch next tile while we compute current
        uint8_t sfa_next, sfb_next;
        uchar4 a4_0_next, b4_0_next, a4_1_next, b4_1_next;
        
        if (has_next) {
            int byte_base_next = sf_next << 3;
            sfa_next = __ldg(&row_sfa[sf_next]);
            sfb_next = __ldg(&batch_sfb[sf_next]);
            a4_0_next = *reinterpret_cast<const uchar4*>(&row_a[byte_base_next]);
            b4_0_next = *reinterpret_cast<const uchar4*>(&batch_b[byte_base_next]);
            a4_1_next = *reinterpret_cast<const uchar4*>(&row_a[byte_base_next + 4]);
            b4_1_next = *reinterpret_cast<const uchar4*>(&batch_b[byte_base_next + 4]);
        }

        // Compute on current tile (data already in registers)
        float scale = decode_fp8(static_cast<int8_t>(sfa_curr)) *
                      decode_fp8(static_cast<int8_t>(sfb_curr));
        __half scale_h = __float2half(scale);
        __half2 scale_h2 = __halves2half2(scale_h, scale_h);

        acc += compute_dot_product(a4_0_curr, a4_1_curr, b4_0_curr, b4_1_curr, scale_h2);

        // Pipeline shift: next becomes current
        if (has_next) {
            sfa_curr = sfa_next;
            sfb_curr = sfb_next;
            a4_0_curr = a4_0_next;
            b4_0_curr = b4_0_next;
            a4_1_curr = a4_1_next;
            b4_1_curr = b4_1_next;
        }
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
    int row_in_block = tid / THREADS_PER_ROW;
    int tid_in_row = tid % THREADS_PER_ROW;

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

    float acc = compute_row_sum_pipelined(
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
    name='batched_scaled_gemv_v12',
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
