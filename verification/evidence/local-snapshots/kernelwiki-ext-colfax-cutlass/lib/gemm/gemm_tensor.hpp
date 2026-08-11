#pragma once

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>

#include "cute/arch/cluster_sm90.hpp"

using namespace cute;

namespace cfk {
template <typename TA, typename LayoutA, typename TB, typename LayoutB,
          typename TC, typename LayoutC, typename TiledMma>
__device__ void gemm(TiledMma &tiled_mma, const Tensor<TA, LayoutA> &tCrA,
                     const Tensor<TB, LayoutB> &tCrB,
                     Tensor<TC, LayoutC> &tCrC) {
  warpgroup_fence_operand(tCrC);
  warpgroup_arrive();

  cute::gemm(tiled_mma, tCrA, tCrB, tCrC);

  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_operand(tCrC);
}

template <typename TA, typename LayoutA, typename TB, typename LayoutB,
          typename TC, typename LayoutC, typename TiledMma>
__device__ void gemm_ldbar(TiledMma &tiled_mma, const Tensor<TA, LayoutA> &tCrA,
                           const Tensor<TB, LayoutB> &tCrB,
                           Tensor<TC, LayoutC> &tCrC,
                           cute::uint64_t &tma_load_mbar, int kPhaseBit) {
  /// Wait on the shared memory barrier until the phase bit flips from
  /// kPhaseBit value
  cute::wait_barrier(tma_load_mbar, kPhaseBit);
  warpgroup_fence_operand(tCrC);
  warpgroup_arrive();

  cute::gemm(tiled_mma, tCrA, tCrB, tCrC);

  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_operand(tCrC);
  __syncthreads(); 
}
} // namespace cfk
