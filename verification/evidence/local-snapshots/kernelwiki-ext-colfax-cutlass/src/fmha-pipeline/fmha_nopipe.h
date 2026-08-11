/***************************************************************************************************
 * Copyright (c) 2017 - 2023 COLFAX
 * Distributed under MIT License
 **************************************************************************************************/

/*! \file
    \brief Driver for the non-pipelined FMHA kernel.

*/

/////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>

#include <cutlass/cutlass.h>

#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

#include "gemm/copy_tensor.hpp"
#include "gemm/gemm_tensor.hpp"

#include "fmha_epilogue.h"
#include "online_softmax.h"
#include "reg2reg.h"
#include "shared_storage.h"

// PrecType = Precision of Computation used by GEMM (half_t by default).
// AccumType = Type of Accumulator used by GEMM (float by default).
// Other types are self-explanatory.
// Main FMHA Device Kernel.
// GemmNType = Precision of Computation used by GEMM-N (half_t by default).
// SoftType = Type of Accumulator used by GEMM-I and in softmax (float by
// default). Other types are self-explanatory.
template <class Gemm1Type, class AccumType, class SoftType, class Gemm2Type,
          class OutputType, class TiledMma0, class TiledMma1, class TiledCopyQ,
          class TileShapeQ, class GmemLayoutQ, class SmemLayoutQ,
          class TiledCopyK, class TileShapeK, class GmemLayoutK,
          class SmemLayoutK, class TileShapeS, class GmemLayoutS,
          class SmemLayoutS, class TiledCopyV, class TileShapeV,
          class GmemLayoutV, class SmemLayoutV, class SmemLayoutVt,
          class TiledCopyO, class TileShapeO, class GmemLayoutO,
          class SmemLayoutO, class GmemLayoutMI, class ClusterShape>
__global__ static void //__launch_bounds__(256, 1)
fmhaForwardNoPipeline(
    Gemm1Type const *Q, CUTE_GRID_CONSTANT TiledCopyQ const tmaLoadQ,
    TileShapeQ tileShapeQ, GmemLayoutQ gmemLayoutQ, SmemLayoutQ smemLayoutQ,
    Gemm1Type const *K, CUTE_GRID_CONSTANT TiledCopyK const tmaLoadK,
    TileShapeK tileShapeK, GmemLayoutK gmemLayoutK, SmemLayoutK smemLayoutK,
    Gemm1Type *S, TileShapeS tileShapeS, GmemLayoutS gmemLayoutS,
    SmemLayoutS smemLayoutS, int nTilesOfK, Gemm2Type *V,
    CUTE_GRID_CONSTANT TiledCopyV const tmaLoadV, TileShapeV tileShapeV,
    GmemLayoutV gmemLayoutV, SmemLayoutV smemLayoutV, SmemLayoutVt smemLayoutVt,
    OutputType *O, CUTE_GRID_CONSTANT TiledCopyO const tmaStoreO,
    TileShapeO tileShapeO, GmemLayoutO gmemLayoutO, SmemLayoutO smemLayoutO,
    SoftType *mi_ptr, SoftType *sPrimePtr, GmemLayoutMI gmemLayoutMi,
    float scale) {
  using namespace cute;

  // Use Shared Storage structure to allocate aligned SMEM addresses.
  extern __shared__ char shared_memory[];
  using SharedStorage =
      SharedStorage<Gemm1Type, Gemm2Type, OutputType, SmemLayoutQ, SmemLayoutK,
                    SmemLayoutS, SmemLayoutV, SmemLayoutO, ClusterShape>;
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(shared_memory);

  // Shared memory barriers use 64bits in SMEM for synchronization
  uint64_t *tma_load_mbar = shared_storage.tma_load_mbar;

  // Get the block coordinates for this CTA.
  auto blockIdxX = uint64_t(blockIdx.x);
  auto blockIdxH = uint64_t(blockIdx.y);
  auto blockIdxB = uint64_t(blockIdx.z);

  // Construct SMEM tensors.
  Tensor sQ =
      make_tensor(make_smem_ptr(shared_storage.smem_q.data()), smemLayoutQ);
  Tensor sO =
      make_tensor(make_smem_ptr(shared_storage.smem_o.data()), smemLayoutO);
  Tensor sK = make_tensor(make_smem_ptr(shared_storage.kv.smem_k.data()),
                          smemLayoutK)(_, _, 0);
#ifdef SINSMEM
  Tensor sS = make_tensor(make_smem_ptr(shared_storage.kv.smem_s.data()),
                          smemLayoutS)(_, _, 0);
#else
  // Just a dummy sS (with smem_v). It's required only for shape later.
  Tensor sS = make_tensor(make_smem_ptr(shared_storage.kv.smem_k.data()),
                          smemLayoutS)(_, _, 0);
#endif
  Tensor sV = make_tensor(make_smem_ptr(shared_storage.kv.smem_v.data()),
                          smemLayoutV)(_, _, 0);

  // Tensor for V Transpose; used in GEMM-II.
  Tensor sVt = make_tensor(make_smem_ptr(shared_storage.kv.smem_v.data()),
                           smemLayoutVt)(_, _, 0);

  // Get the full un-partitioned tensors.
  // TMA tensors are special tensors.
  Tensor mQ = tmaLoadQ.get_tma_tensor(shape(gmemLayoutQ));
  Tensor mK = tmaLoadK.get_tma_tensor(shape(gmemLayoutK));
  Tensor mV = tmaLoadV.get_tma_tensor(shape(gmemLayoutV));

  TiledMma0 tiledMma0;
  auto threadMma0 = tiledMma0.get_thread_slice(threadIdx.x);
  TiledMma1 tiledMma1;
  auto threadMma1 = tiledMma1.get_thread_slice(threadIdx.x);

  uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
  constexpr uint32_t cluster_shape_x = get<0>(ClusterShape{});
  uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x,
                                  block_rank_in_cluster / cluster_shape_x};
  uint16_t mcast_mask_a = 0;
  auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
  for (int n = 0; n < size(block_layout); ++n) {
    mcast_mask_a |= (uint16_t(1) << block_layout(n, 0, Int<0>{}));
  }

  //
  // Prepare the TMA_LOADS and TMA_STORE.
  // Always use the 0th slice for Q and O.
  // Slice for K and V is dependent on CTA id in cluster.
  // This will be 0 as well for trivial ClusterShape.
  //
  auto cta_tmaO = tmaStoreO.get_slice(0);
  auto cta_tmaQ = tmaLoadQ.get_slice(0);
  auto cta_tmak = tmaLoadK.get_slice(cluster_local_block_id.x);
  auto cta_tmaV = tmaLoadV.get_slice(cluster_local_block_id.x);

  // Get the block of Q for this CTA using the block coordinates.
  auto blkCoordQ = make_coord(blockIdxX, 0, blockIdxH, blockIdxB);
  Tensor gQ = local_tile(mQ, tileShapeQ, blkCoordQ);

  // Partition the copying of source tiles for Q among threads.
  Tensor tQgQX = cta_tmaQ.partition_S(gQ);
  // Group the REST_X modes and the TMA_X modes to easily iterate through the
  // tiles
  Tensor tQgQ = group_modes<1, rank(tQgQX)>(tQgQX); // (TMA,REST)
  auto kTiles = size<1>(tQgQ);
  assert(kTiles == 1);
  assert(kTiles == size<2>(gQ));

  //
  // Partition the copying of dest tiles for Q, K and V among threads.
  //
  Tensor tQsQX = cta_tmaQ.partition_D(sQ);
  Tensor tQsQ = group_modes<1, rank(tQsQX)>(tQsQX); // (TMA,REST)
  Tensor tKsKX = cta_tmak.partition_D(sK);
  Tensor tKsK = group_modes<1, rank(tKsKX)>(tKsKX);
  Tensor tVsVX = cta_tmaV.partition_D(sV);
  Tensor tVsV = group_modes<1, rank(tVsVX)>(tVsVX);
  static_assert(size<1>(tQsQ) == 1);
  static_assert(size<1>(tKsK) == 1);

  // Allocate "fragments/descriptors"
  // for first matmul.
  Tensor tSrQ = threadMma0.partition_fragment_A(sQ);
  Tensor tSrK = threadMma0.partition_fragment_B(sK);
  Tensor tSrS = partition_fragment_C(tiledMma0, tileShapeS);
  clear(tSrS);

  // Allocate "fragments/descriptors"
  // for second matmul.
  // Note: S becomes P.
  Tensor tOrV = threadMma1.partition_fragment_B(sVt);
  Tensor tOrO = partition_fragment_C(tiledMma1, tileShapeO);
  clear(tOrO);

// Use this flag to store result of GEMM-I to SMEM. GEMM-II
// will also read from SMEM. By default, this flag is disabled.
#ifdef SINSMEM
  Tensor tSsS = threadMma0.partition_C(sS);
  cute::fill(tSsS, Gemm2Type(0.0));
  Tensor tOrP = threadMma1.partition_fragment_A(sS);
#else
  Tensor tOrS = threadMma1.partition_fragment_A(sS);
  auto tOrPLayout = ReshapeTStoTP()(tSrS, tOrS);
  auto tOrP = make_tensor(tSrS.data(), tOrPLayout);
#endif


  auto reg2reg = ReorgCFp8toAFp8();
  // Allocate space for per-thread rowMax and rowSum in rmem.
  Tensor rowMax = make_tensor<AccumType>(Shape<Int<2 * size<1>(tSrS)>>{});
  Tensor rowSum = make_fragment_like(rowMax);
  cute::fill(rowMax, -cutlass::platform::numeric_limits<AccumType>::infinity());
  cute::fill(rowSum, AccumType(0.0));

  cute::cluster_arrive_relaxed();
  cute::cluster_wait();

  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();

  // Barrier 0-2 are transaction-bytes based. numthreads = 1 for them.
  // Barrier 0 is used for K copy, 1 for V copy, and 2 for Q copy.
  cfk::barrierInit(tma_load_mbar[0], 1);
  cfk::barrierInit(tma_load_mbar[1], 1);
  cfk::barrierInit(tma_load_mbar[2], 1);

  // Copy Q tile from GMEM to SMEM.
  cfk::copy(tQgQ(_, 0), tQsQ(_, 0), tmaLoadQ, tma_load_mbar[2]);
  cute::wait_barrier(tma_load_mbar[2], 0); // This is REQUIRED.

  // Prepare Tensors for first K copy on GMEM side.
  auto blkCoordK = make_coord(0, 0, blockIdxH, blockIdxB);
  Tensor gK = local_tile(mK, tileShapeK, blkCoordK);
  Tensor tKgKX = cta_tmak.partition_S(gK);
  Tensor tKgK = group_modes<1, rank(tKgKX)>(tKgKX); // (TMA,REST)
  assert(size<1>(tKgK) == size<2>(gK));
  assert(size<1>(tKgK) == kTiles);
  static_assert(size<1>(tKsK) == 1);

  // Copy first tile of K from GMEM to SMEM.
  cfk::copy(tKgK(_, 0), tKsK(_, 0), tmaLoadK, tma_load_mbar[0], mcast_mask_a);

// don't support QINRMEM for now
// #ifdef QINRMEM
//   Tensor tSsQ = threadMma0.partition_A(sQ);
//   cfk::copy(tSsQ, tSrQ);
// #endif

  // Initialize phase for barriers 0 and 1.
  int phase = 0;

#pragma unroll
  for (uint64_t blockIdxY = 0; blockIdxY < nTilesOfK; ++blockIdxY) {

    // Prepare Tensors for V copy on GMEM side.
#ifdef GEMM2FP8
    auto blkCoordV = make_coord(0, blockIdxY, blockIdxH, blockIdxB);
#else
    auto blkCoordV = make_coord(blockIdxY, 0, blockIdxH, blockIdxB);
#endif

    Tensor gV = local_tile(mV, tileShapeV, blkCoordV);
    Tensor tVgVX = cta_tmaV.partition_S(gV);
    Tensor tVgV = group_modes<1, rank(tVgVX)>(tVgVX);
    assert(size<1>(tVgV) == size<2>(gV));
    assert(size<1>(tVgV) == 1);

    // Copy current tile of V from GMEM to SMEM.
    cfk::syncCluster<ClusterShape>();
    cfk::copy(tVgV(_, 0), tVsV(_, 0), tmaLoadV, tma_load_mbar[1], mcast_mask_a);
    clear(tSrS);

    // Issue GEMM-I.
    cfk::gemm_ldbar(tiledMma0, tSrQ, tSrK, tSrS, tma_load_mbar[0], phase);

// Required for verification ONLY.
#ifdef COPYOUTMM0
    Tensor mS = make_tensor(make_gmem_ptr(S), gmemLayoutS);
    auto blkCoordS = make_coord(blockIdxX, blockIdxY, blockIdxH, blockIdxB);
    Tensor gS = local_tile(mS, tileShapeS, blkCoordS);
    Tensor tSgS = threadMma0.partition_C(gS);
    copy(tSrS, tSgS);
#endif

    // Copy next tile of K from GMEM to SMEM.
    if (blockIdxY != (nTilesOfK - 1)) {
      auto blkCoordK = make_coord(blockIdxY + 1, 0, blockIdxH, blockIdxB);
      auto gK = local_tile(mK, tileShapeK, blkCoordK);

      Tensor tKgKX = cta_tmak.partition_S(gK);
      Tensor tKgK = group_modes<1, rank(tKgKX)>(tKgKX);
      cfk::syncCluster<ClusterShape>();
      cfk::copy(tKgK(_, 0), tKsK(_, 0), tmaLoadK, tma_load_mbar[0],
                mcast_mask_a);
    }

    if (blockIdxY == 0) { // Compute Online Softmax and NO Output Rescaling.
      onlineSoftmaxAndRescale<true, AccumType>(rowMax, rowSum, tSrS, tOrO,
                                               scale);
    } else { // Compute Online Softmax and Output Rescaling.
      onlineSoftmaxAndRescale<false, AccumType>(rowMax, rowSum, tSrS, tOrO,
                                                scale);
    }
    warpgroup_fence_operand(tSrS);

#ifdef SINSMEM
    // ISSUE GEMM-II with Operand A from SMEM.
    // Copy OperandA from RMEM to SMEM before issuing.
#if 0
    if (cute::thread0()) {
         print ("before : \n");
         print_tensor(tSrS);
    }
         __syncthreads();
#endif

    cfk::copy(tSrS, tSsS);
#if 1
    if (cute::thread0()) {
         print ("after : \n");
         print_tensor(tSsS);
    }
         __syncthreads();
#endif

    cfk::gemm_ldbar(tiledMma1, tOrP, tOrV, tOrO, tma_load_mbar[1], phase);
#else
    // ISSUE GEMM-II with Operand A from RMEM.
    // Convert Operand A From AccumType [=float] to PrecType [=half_t] before
    // issuing.
    auto tSrSPrec = convert_type<Gemm2Type, AccumType>(tSrS);

#ifdef GEMM2FP8
    reg2reg(tSrSPrec);
#endif

    auto tOrP = make_tensor(tSrSPrec.data(), tOrPLayout);
    warpgroup_fence_operand(tSrS);
    cfk::gemm_ldbar(tiledMma1, tOrP, tOrV,
                    tOrO, tma_load_mbar[1], phase);
#endif
    // Flip phase for barrier.
    phase = (phase + 1) % 2;
  }

  fmhaForwardWriteOutTMA(tOrO, rowMax, rowSum, O, tileShapeO, gmemLayoutO,
                         tiledMma1, sO, tmaStoreO, warp_idx == 0,
                         SoftType(0.0));

// Write out rowMax and rowSum to GMEM.
// Required for verification ONLY.
#ifdef COPYOUTMI
  fmhaForwardWriteOutSoftMax(rowMax, rowSum, mi_ptr, sPrimePtr, gmemLayoutMi,
                             tiledMma0, tileShapeO);
#endif

  cute::cluster_arrive_relaxed();
  cute::cluster_wait();
  __syncthreads();
}
