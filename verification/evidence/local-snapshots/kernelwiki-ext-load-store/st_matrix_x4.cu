#include <cstdint>
#include <iostream>

__device__ __forceinline__ void stmatrix_sync_aligned_m8n8_x4_b16(
    uint32_t &d0, uint32_t &d1, uint32_t &d2, uint32_t &d3,
    const uint32_t &address) {
  asm volatile(
      "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(
          address),
      "r"(d0), "r"(d1), "r"(d2), "r"(d3));
}

__global__ void stmatrix(uint16_t *value) {
  constexpr int N = 64;
  __shared__ uint16_t smem[4 * N];
  auto tid = threadIdx.x;

  const uint32_t offset_rows = sizeof(uint16_t) * (tid % 8) * 8;
  const uint32_t offset_matrix = sizeof(uint16_t) * ((tid / 8) % 4) * 64;
  const uint32_t offset = offset_rows + offset_matrix;
  const uint32_t address = __cvta_generic_to_shared(smem) + offset;

  uint32_t frag1 = 0x00000000;
  frag1 |= (tid * 2 + 0);
  frag1 |= (tid * 2 + 1) << 16;
  uint32_t frag2 = 0x00000000;
  frag2 |= (tid * 2 + 0 + 64);
  frag2 |= (tid * 2 + 1 + 64) << 16;
  uint32_t frag3 = 0x00000000;
  frag3 |= (tid * 2 + 0 + 128);
  frag3 |= (tid * 2 + 1 + 128) << 16;
  uint32_t frag4 = 0x00000000;
  frag4 |= (tid * 2 + 0 + 192);
  frag4 |= (tid * 2 + 1 + 192) << 16;
  __syncthreads();

  stmatrix_sync_aligned_m8n8_x4_b16(frag1, frag2, frag3, frag4, address);

  __syncthreads();

  uint16_t number1 = static_cast<uint16_t>(frag1 & 0xFFFF);
  uint16_t number2 = static_cast<uint16_t>((frag1 >> 16) & 0xFFFF);
  printf("%d -> %d  %d   %d   \n", tid, (int)(smem[2 * tid]), (int)number1,
         (int)number2);
  uint16_t number3 = static_cast<uint16_t>(frag2 & 0xFFFF);
  uint16_t number4 = static_cast<uint16_t>((frag2 >> 16) & 0xFFFF);
  printf("%d -> %d  %d   %d   \n", tid, (int)(smem[2 * tid + 64]), (int)number3,
         (int)number4);
  uint16_t number5 = static_cast<uint16_t>(frag3 & 0xFFFF);
  uint16_t number6 = static_cast<uint16_t>((frag3 >> 16) & 0xFFFF);
  printf("%d -> %d  %d   %d   \n", tid, (int)(smem[2 * tid + 128]),
         (int)number5, (int)number6);
  uint16_t number7 = static_cast<uint16_t>(frag4 & 0xFFFF);
  uint16_t number8 = static_cast<uint16_t>((frag4 >> 16) & 0xFFFF);
  printf("%d -> %d  %d   %d   \n", tid, (int)(smem[2 * tid + 192]),
         (int)number7, (int)number8);
}

int main() {
  uint16_t *d_value;
  cudaMalloc(&d_value, sizeof(uint16_t));
  stmatrix<<<1, 32>>>(d_value);
  cudaDeviceSynchronize();
  cudaFree(d_value);
  return 0;
}