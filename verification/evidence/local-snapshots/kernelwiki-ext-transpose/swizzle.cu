#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda/barrier>

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
__device__ int calculate_col_swizzle(int row, int col) {
  int i16 = (row * BLOCK_SIZE + col) * 4 >> 4;
  int y16 = i16 >> 3;
  int x16 = i16 & 7;
  int x16_swz = y16 ^ x16;
  return ((x16_swz * 4) & (BLOCK_SIZE - 1)) + (col & 3);
}

template <int BLOCK_SIZE>
__global__ void kernel(const __grid_constant__ CUtensorMap tensor_map) {
  __shared__ alignas(1024) int smem_buffer[BLOCK_SIZE * BLOCK_SIZE];

  int x = blockIdx.x * BLOCK_SIZE;
  int y = blockIdx.y * BLOCK_SIZE;

  int col = threadIdx.x % BLOCK_SIZE;
  int row = threadIdx.x / BLOCK_SIZE;

  int col_swizzle = calculate_col_swizzle<BLOCK_SIZE>(row, col);

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

  smem_buffer[row * BLOCK_SIZE + col_swizzle] = (row * BLOCK_SIZE + col) % 32;

  cde::fence_proxy_async_shared_cta();
  __syncthreads();

  if (threadIdx.x == 0) {
    cde::cp_async_bulk_tensor_2d_shared_to_global(&tensor_map, x, y,
                                                  &smem_buffer);
    cde::cp_async_bulk_commit_group();
    cde::cp_async_bulk_wait_group_read<0>();
  }

  if (threadIdx.x == 0) {
    (&bar)->~barrier();
  }
}

int main() {
  const int GMEM_WIDTH = 32;
  const int GMEM_HEIGHT = 32;
  const int BLOCK_SIZE = 32;
  const int SMEM_WIDTH = BLOCK_SIZE;
  const int SMEM_HEIGHT = BLOCK_SIZE;
  const size_t SIZE = GMEM_HEIGHT * GMEM_WIDTH * sizeof(int);
  std::cout << BLOCK_SIZE * BLOCK_SIZE * sizeof(int) << std::endl;

  int *h_in = new int[GMEM_HEIGHT * GMEM_WIDTH];
  int *h_out = new int[GMEM_HEIGHT * GMEM_WIDTH];

  srand(42);
  for (int i = 0; i < GMEM_HEIGHT * GMEM_WIDTH; ++i) {
    h_in[i] = rand() % 9;
  }


  int *d;
  CHECK_CUDA_ERROR(cudaMalloc(&d, SIZE));
  CHECK_CUDA_ERROR(cudaMemcpy(d, h_in, SIZE, cudaMemcpyHostToDevice));
  void *tensor_ptr = (void *)d;

  CUtensorMap tensor_map{};
  constexpr uint32_t rank = 2;
  uint64_t size[rank] = {GMEM_WIDTH, GMEM_HEIGHT};
  uint64_t stride[rank - 1] = {GMEM_WIDTH * sizeof(int)};
  uint32_t box_size[rank] = {SMEM_WIDTH, SMEM_HEIGHT};
  uint32_t elem_stride[rank] = {1, 1};

  CUresult res = cuTensorMapEncodeTiled(
      &tensor_map,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
      rank,
      tensor_ptr,
      size,
      stride,
      box_size,
      elem_stride,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  assert(res == CUDA_SUCCESS);

  dim3 blockDim(SMEM_WIDTH * SMEM_HEIGHT, 1, 1);
  dim3 gridDim(GMEM_WIDTH / SMEM_WIDTH, GMEM_HEIGHT / SMEM_HEIGHT, 1);

  kernel<BLOCK_SIZE><<<gridDim, blockDim>>>(tensor_map);

  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA_ERROR(cudaMemcpy(h_out, d, SIZE, cudaMemcpyDeviceToHost));

  std::cout << "Visualize Bank assignment:" << std::endl;
  utils::printMatrixHeatmap32(h_out, GMEM_HEIGHT, GMEM_WIDTH);
  std::cout << std::endl;

  CHECK_CUDA_ERROR(cudaFree(d));
  free(h_in);
  free(h_out);
}