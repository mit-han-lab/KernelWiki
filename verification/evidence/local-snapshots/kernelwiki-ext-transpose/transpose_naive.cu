#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cuda/barrier>
#include <random>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

#include "utils.h"

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, char const *func, char const *file, int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(char const *file, int line) {
  cudaError_t const err{cudaGetLastError()};
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

template <int BLOCK_SIZE>
__global__ void kernel(const __grid_constant__ CUtensorMap tensor_map,
                       const __grid_constant__ CUtensorMap tensor_map_tr) {
  __shared__ alignas(1024) float smem_buffer[BLOCK_SIZE * BLOCK_SIZE];
  __shared__ alignas(1024) float smem_buffer_tr[BLOCK_SIZE * BLOCK_SIZE];
  int x = blockIdx.x * BLOCK_SIZE;
  int y = blockIdx.y * BLOCK_SIZE;

  int col = threadIdx.x % BLOCK_SIZE;
  int row = threadIdx.x / BLOCK_SIZE;

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;

  if (threadIdx.x == 0) {
    init(&bar, blockDim.x);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  barrier::arrival_token token;
  if (threadIdx.x == 0) {
    cde::cp_async_bulk_tensor_2d_global_to_shared(&smem_buffer, &tensor_map, x,
                                                  y, bar);
    token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
  } else {
    token = bar.arrive();
  }
  bar.wait(std::move(token));

  smem_buffer_tr[col * BLOCK_SIZE + row] = smem_buffer[row * BLOCK_SIZE + col];

  cde::fence_proxy_async_shared_cta();
  __syncthreads();
  if (threadIdx.x == 0) {
    cde::cp_async_bulk_tensor_2d_shared_to_global(&tensor_map_tr, y, x,
                                                  &smem_buffer_tr);
    cde::cp_async_bulk_commit_group();
    cde::cp_async_bulk_wait_group_read<0>();
  }

  if (threadIdx.x == 0) {
    (&bar)->~barrier();
  }
}

int main() {
  const int GMEM_WIDTH = 32768;
  const int GMEM_HEIGHT = 32768;
  const int BLOCK_SIZE = 32;
  const int SMEM_WIDTH = BLOCK_SIZE;
  const int SMEM_HEIGHT = BLOCK_SIZE;
  const size_t SIZE = GMEM_HEIGHT * GMEM_WIDTH * sizeof(float);

  float *h_in = new float[GMEM_HEIGHT * GMEM_WIDTH];
  float *h_out = new float[GMEM_HEIGHT * GMEM_WIDTH];

  std::default_random_engine generator(42);
  std::normal_distribution<float> distribution(0.0, 1.0);

  for (int i = 0; i < GMEM_HEIGHT * GMEM_WIDTH; ++i) {
    h_in[i] = distribution(generator);
  }

  float *d;
  float *d_tr;
  CHECK_CUDA_ERROR(cudaMalloc(&d, SIZE));
  CHECK_CUDA_ERROR(cudaMemcpy(d, h_in, SIZE, cudaMemcpyHostToDevice));
  void *tensor_ptr = (void *)d;
  CHECK_CUDA_ERROR(cudaMalloc(&d_tr, SIZE));
  void *tensor_ptr_tr = (void *)d_tr;

  CUtensorMap tensor_map{};
  CUtensorMap tensor_map_tr{};
  constexpr uint32_t rank = 2;
  uint64_t size[rank] = {GMEM_WIDTH, GMEM_HEIGHT};
  uint64_t size_tr[rank] = {GMEM_HEIGHT, GMEM_WIDTH};
  uint64_t stride[rank - 1] = {GMEM_WIDTH * sizeof(float)};
  uint64_t stride_tr[rank - 1] = {GMEM_HEIGHT * sizeof(float)};
  uint32_t box_size[rank] = {SMEM_WIDTH, SMEM_HEIGHT};
  uint32_t box_size_tr[rank] = {SMEM_HEIGHT, SMEM_WIDTH};
  uint32_t elem_stride[rank] = {1, 1};

  CUresult res = cuTensorMapEncodeTiled(
      &tensor_map,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
      rank,
      tensor_ptr,
      size,
      stride,
      box_size,
      elem_stride,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  assert(res == CUDA_SUCCESS);

  CUresult res_tr = cuTensorMapEncodeTiled(
      &tensor_map_tr,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
      rank,
      tensor_ptr_tr,
      size_tr,
      stride,
      box_size_tr,
      elem_stride,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  assert(res_tr == CUDA_SUCCESS);

  dim3 blockDim(SMEM_WIDTH * SMEM_HEIGHT, 1, 1);
  dim3 gridDim(GMEM_WIDTH / SMEM_WIDTH, GMEM_HEIGHT / SMEM_HEIGHT, 1);

  kernel<BLOCK_SIZE><<<gridDim, blockDim>>>(tensor_map, tensor_map_tr);

  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA_ERROR(cudaMemcpy(h_out, d_tr, SIZE, cudaMemcpyDeviceToHost));

  const float epsilon = 1e-5f;
  for (int x = 0; x < GMEM_HEIGHT; x++) {
    for (int y = 0; y < GMEM_WIDTH; y++) {
      float expected = h_in[x * GMEM_WIDTH + y];
      float actual = h_out[y * GMEM_HEIGHT + x];
      if (std::fabs(expected - actual) > epsilon) {
        std::cout << "Error at position (" << x << "," << y << "): expected "
                  << expected << " but got " << actual << std::endl;
        return -1;
      }
    }
  }

  std::cout << "Passed" << std::endl;

  int numRounds = 10000;
  size_t numCrossMemoryBound = 2 * SIZE;
  cudaEvent_t start, stop;
  float time;

  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));

  CHECK_CUDA_ERROR(cudaEventRecord(start));
  for (int i = 0; i < numRounds; i++) {
    kernel<BLOCK_SIZE><<<gridDim, blockDim>>>(tensor_map, tensor_map_tr);
  }
  CHECK_CUDA_ERROR(cudaEventRecord(stop));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
  CHECK_LAST_CUDA_ERROR();

  float latency = time / numRounds;
  float bandwidth = (numCrossMemoryBound / latency) / 1e6;

  std::cout << "Latency = " << latency << " ms" << std::endl;
  std::cout << "Bandwidth = " << bandwidth << " GB/s" << std::endl;
  std::cout << "% of max = " << bandwidth / 3300 * 100 << " %" << std::endl;

  CHECK_CUDA_ERROR(cudaFree(d));
  free(h_in);
  free(h_out);
}