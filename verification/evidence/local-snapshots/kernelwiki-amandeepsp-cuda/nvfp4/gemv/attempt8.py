"""
NVFP4 Block-Scaled GEMV - Attempt 8

╔══════════════════════════════════════════════════════════════════════════════╗
║  🥈 SECOND BEST - Split-K with Atomics                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Key characteristics:                                                         ║
║    - Same thread mapping as attempt 7                                         ║
║    - Split-K with K_TILE_SIZE=256                                             ║
║    - Uses atomicAdd for accumulation (float32 output)                         ║
║    - Additional grid dimension for K tiling                                   ║
║    - Requires FP32→FP16 conversion at end                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Why it's slower than Attempt 7:                                              ║
║    1. Atomic overhead: atomicAdd introduces contention and serialization      ║
║    2. Extra memory traffic: Split-K requires additional write phase           ║
║       and float32 intermediate storage                                        ║
║    3. Grid overhead: More blocks means more scheduling overhead for           ║
║       a memory-bound kernel                                                   ║
║    4. Data type conversion: Final FP32→FP16 conversion adds latency          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Lesson learned:                                                              ║
║    Split-K parallelism doesn't help when the kernel is memory-bound.          ║
║    The atomic contention and extra memory traffic outweigh any benefit        ║
║    from increased parallelism.                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from torch.utils.cpp_extension import load_inline
from task import input_t, output_t
import torch

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
#define K_TILE_SIZE 256

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

__device__ __forceinline__ __half2 dot_scaled_4bytes(
    uchar4 a4,
    uchar4 b4,
    __half2 scale_h2
) {
    __half2 acc0 = __hmul2(decode_fp4x2(a4.x), __hmul2(decode_fp4x2(b4.x), scale_h2));
    __half2 acc1 = __hmul2(decode_fp4x2(a4.y), __hmul2(decode_fp4x2(b4.y), scale_h2));
    acc0 = __hfma2(decode_fp4x2(a4.z), __hmul2(decode_fp4x2(b4.z), scale_h2), acc0);
    acc1 = __hfma2(decode_fp4x2(a4.w), __hmul2(decode_fp4x2(b4.w), scale_h2), acc1);

    return __hadd2(acc0, acc1);
}

__device__ __forceinline__ float compute_row_sum(
    const uint8_t* row_a,
    const uint8_t* batch_b,
    const uint8_t* row_sfa,
    const uint8_t* batch_sfb,
    int k_start_sf,
    int k_end_sf,
    int tid
) {
    float acc = 0.0f;
    const int THREADS_PER_ROW = BLOCK_SIZE / ROWS_PER_BLOCK;

#pragma unroll 4
    for (int sf = k_start_sf + tid; sf < k_end_sf; sf += THREADS_PER_ROW) {
        float scale = decode_fp8(static_cast<int8_t>(__ldg(&row_sfa[sf]))) *
                      decode_fp8(static_cast<int8_t>(__ldg(&batch_sfb[sf])));
        __half scale_h = __float2half(scale);
        __half2 scale_h2 = __halves2half2(scale_h, scale_h);

        int byte_base = sf << 3;  // sf * 8 using bit shift

        uchar4 a4_0 = *reinterpret_cast<const uchar4*>(&row_a[byte_base]);
        uchar4 b4_0 = *reinterpret_cast<const uchar4*>(&batch_b[byte_base]);
        uchar4 a4_1 = *reinterpret_cast<const uchar4*>(&row_a[byte_base + 4]);
        uchar4 b4_1 = *reinterpret_cast<const uchar4*>(&batch_b[byte_base + 4]);

        __half2 acc_h2_0 = dot_scaled_4bytes(a4_0, b4_0, scale_h2);
        __half2 acc_h2_1 = dot_scaled_4bytes(a4_1, b4_1, scale_h2);

        // Combine and accumulate in one step
        float2 f0 = __half22float2(acc_h2_0);
        float2 f1 = __half22float2(acc_h2_1);
        acc += f0.x + f0.y + f1.x + f1.y;
    }

    return acc;
}

__global__ void gemv_nvfp4_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const int8_t* __restrict__ sfa,
    const int8_t* __restrict__ sfb,
    float* __restrict__ c,
    int M, int K, int L,
    int N_rows
) {
    int m_start = blockIdx.x * ROWS_PER_BLOCK;
    int k_block_idx = blockIdx.y;
    int l = blockIdx.z;
    int tid = threadIdx.x;
    int row_in_block = tid / (BLOCK_SIZE / ROWS_PER_BLOCK); 
    int tid_in_row = tid % (BLOCK_SIZE / ROWS_PER_BLOCK);

    if (m_start >= M || l >= L) return;

    // Total K scale factors
    const int K_sf = K / 16;
    
    // Calculate K range for this block
    // Each K_TILE_SIZE is in terms of original K elements.
    // In terms of scale factors (blocks of 16), we divide by 16.
    const int sf_per_tile = K_TILE_SIZE / 16;
    
    int k_start_sf = k_block_idx * sf_per_tile;
    int k_end_sf = k_start_sf + sf_per_tile;
    if (k_end_sf > K_sf) k_end_sf = K_sf;
    
    if (k_start_sf >= K_sf) return;

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
        k_start_sf,
        k_end_sf,
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
        atomicAdd(&c[c_idx], row_sum);
    }
}

torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c) {
    int M = a.size(0);
    int K = a.size(1) * 2;
    int L = a.size(2);
    int N_rows = b.size(0);

    // Launch grid with M/ROWS_PER_BLOCK blocks
    int grid_m = (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    // Split K into blocks of K_TILE_SIZE
    int grid_k = (K + K_TILE_SIZE - 1) / K_TILE_SIZE;
    
    dim3 grid(grid_m, grid_k, L);
    dim3 block(BLOCK_SIZE);

    auto* a_ptr = reinterpret_cast<const int8_t*>(a.data_ptr());
    auto* b_ptr = reinterpret_cast<const int8_t*>(b.data_ptr());
    auto* sfa_ptr = reinterpret_cast<const int8_t*>(sfa.data_ptr());
    auto* sfb_ptr = reinterpret_cast<const int8_t*>(sfb.data_ptr());
    auto* c_ptr = reinterpret_cast<float*>(c.data_ptr());

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
    name='batched_scaled_gemv_v8',
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
    # Use float32 for accumulation to avoid precision issues with atomicAdd
    c_fp32 = torch.zeros(c.shape, dtype=torch.float32, device=c.device)
    module.batched_scaled_gemv_cuda(
        a,
        b,
        sfa_ref,
        sfb_ref,
        c_fp32
    )
    # Convert back to fp16
    return c_fp32.to(torch.float16)

