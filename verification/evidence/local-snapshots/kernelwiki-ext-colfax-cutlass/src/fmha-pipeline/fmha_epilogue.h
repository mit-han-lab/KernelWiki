#pragma once

#include "online_softmax.h"
#include "shared_storage.h"

// Epilogue that copies RowSum and RowMax (For debugging only).
template <class RowMax, class RowSum, class SoftType, class TiledMma0,
          class GmemLayoutMI,
          class TileShapeO>
__device__ static void //__launch_bounds__(128, 2)
fmhaForwardWriteOutSoftMax(const RowMax &rowMax, const RowSum &rowSum,
                           SoftType *mi_ptr, SoftType *sPrimePtr,
                           GmemLayoutMI gmemLayoutMi,
                           const TiledMma0 &tiledMma0, TileShapeO tileShapeO) {

  // Get the block coordinates for this CTA.
  auto blockIdxX = uint64_t(blockIdx.x);
  auto blockIdxH = uint64_t(blockIdx.y);
  auto blockIdxB = uint64_t(blockIdx.z);

  Tensor miGlobal = make_tensor(make_gmem_ptr(mi_ptr), gmemLayoutMi);
  Tensor miGlobalOut =
      local_tile(miGlobal, make_shape(get<0>(tileShapeO), 1, 1),
                 make_coord(blockIdxX, blockIdxH, blockIdxB));
  Tensor sPrimeGlobal = make_tensor(make_gmem_ptr(sPrimePtr), gmemLayoutMi);
  Tensor sPrimeGlobalOut =
      local_tile(sPrimeGlobal, make_shape(get<0>(tileShapeO), 1, 1),
                 make_coord(blockIdxX, blockIdxH, blockIdxB));
  if (threadIdx.x % 4 == 0) {
    auto mmaThreadLayoutC = TiledMma0{}.get_layoutC_TV();
    auto mmaShapeMNK = cute::tile_shape(TiledMma0{});
    auto mmaShapeMN = make_shape(shape<0>(mmaShapeMNK), shape<1>(mmaShapeMNK));
    auto flatCoord =
        cute::idx2crd(mmaThreadLayoutC(threadIdx.x, 0), mmaShapeMN);
    auto rowIdGlobal = get<0>(flatCoord); // starting rowId
    auto rowId = 0;
    for (int i = rowIdGlobal; i < kQueriesPerBlock;
         i += NumMmaWarpGroups * 64) {
      miGlobalOut(i) = rowMax(rowId);
      sPrimeGlobalOut(i) = rowSum(rowId);
      miGlobalOut(i + 8) = rowMax(rowId + 1);
      sPrimeGlobalOut(i + 8) = rowSum(rowId + 1);
      rowId += 2;
    }
  }
}

// Epilogue that copies RMEM -> GMEM directly
// Reports as uncoalesced stores by the profiler
template <class TensorO, class RowMax, class RowSum, class SoftType,
          class OutputType, class TiledMma1, class TileShapeO,
          class GmemLayoutO>
__device__ static void //__launch_bounds__(128, 2)
fmhaForwardWriteOut(TensorO &tOrO, const RowMax &rowMax, const RowSum &rowSum,
                    OutputType *O, TileShapeO tileShapeO,
                    GmemLayoutO gmemLayoutO, const TiledMma1 &tiledMma1,
                    const SoftType &) {

  // Get the block coordinates for this CTA.
  auto blockIdxX = uint64_t(blockIdx.x);
  auto blockIdxH = uint64_t(blockIdx.y);
  auto blockIdxB = uint64_t(blockIdx.z);

  // Apply softmax normalization before writing out to GMEM.
  applySoftmaxNormalizer<SoftType>(rowSum, tOrO);

  Tensor mO = make_tensor(make_gmem_ptr(O), gmemLayoutO);
  auto blkCoordO = make_coord(blockIdxX, 0, blockIdxH, blockIdxB);
  Tensor gO = local_tile(mO, tileShapeO, blkCoordO);
  auto threadMma1 = tiledMma1.get_thread_slice(threadIdx.x);
  Tensor tOgO = threadMma1.partition_C(gO);

  // Write out to GMEM.
  copy(tOrO, tOgO);
}

// Epilogue with SMEM pass-through
template <class TensorO, class RowMax, class RowSum, class SoftType,
          class OutputType, class TiledMma1, class TileShapeO,
          class GmemLayoutO, class TensorSO,
          class ThreadLayoutO>
__device__ static void //__launch_bounds__(128, 2)
fmhaForwardWriteOutCoalesced(TensorO &tOrO, const RowMax &rowMax,
                             const RowSum &rowSum, OutputType *O,
                             TileShapeO tileShapeO, GmemLayoutO gmemLayoutO,
                             const TiledMma1 &tiledMma1, const TensorSO &sO,
                             ThreadLayoutO tO, const SoftType &) {

  // Get the block coordinates for this CTA.
  auto blockIdxX = uint64_t(blockIdx.x);
  auto blockIdxH = uint64_t(blockIdx.y);
  auto blockIdxB = uint64_t(blockIdx.z);

  // Apply softmax normalization before writing out to GMEM.
  applySoftmaxNormalizer<SoftType>(rowSum, tOrO);

  auto threadMma1 = tiledMma1.get_thread_slice(threadIdx.x);
  Tensor tOsOAcc = threadMma1.partition_C(sO);
  auto synchronize = [&]() {
    cutlass::arch::NamedBarrier::sync(size(TiledMma1{}), 0);
  };
  synchronize();
  cfk::copy_nosync(tOrO, tOsOAcc);
  synchronize();

  Tensor mO = make_tensor(make_gmem_ptr(O), gmemLayoutO);
  auto blkCoordO = make_coord(blockIdxX, 0, blockIdxH, blockIdxB);
  Tensor gO = local_tile(mO, tileShapeO, blkCoordO);

  auto tOgO =
      local_partition(gO, tO, threadIdx.x, Step<_1, _1>{}); // (THR_M,THR_N)
  auto tOsO =
      local_partition(sO, tO, threadIdx.x, Step<_1, _1>{}); // (THR_M,THR_N)

  // Write out to GMEM.
  copy(tOsO, tOgO);
}

// Epilogue with TMA Store
template <class TensorO, class RowMax, class RowSum, class SoftType,
          class OutputType, class TiledMma1, class TileShapeO,
          class GmemLayoutO, class TensorSO,
          class TiledCopyO>
__device__ static void //__launch_bounds__(128, 2)
fmhaForwardWriteOutTMA(TensorO &tOrO, const RowMax &rowMax,
                       const RowSum &rowSum, OutputType *O,
                       TileShapeO tileShapeO, GmemLayoutO gmemLayoutO,
                       const TiledMma1 &tiledMma1, const TensorSO &sO,
                       TiledCopyO const &tmaStoreO, bool leaderWarp,
                       const SoftType &) {

  // Get the block coordinates for this CTA.
  auto blockIdxX = uint64_t(blockIdx.x);
  auto blockIdxH = uint64_t(blockIdx.y);
  auto blockIdxB = uint64_t(blockIdx.z);

  // Apply softmax normalization before writing out to GMEM.
  applySoftmaxNormalizer<SoftType>(rowSum, tOrO);

  auto threadMma1 = tiledMma1.get_thread_slice(threadIdx.x);
  Tensor tOsOAcc = threadMma1.partition_C(sO);

//  Note: The following will not work for Warp specialized case.
//  So use synchronize() and cfk::copy_nosync.
//  cfk::copy(tOrO, tOsOAcc);

  auto synchronize = [&]() {
    cutlass::arch::NamedBarrier::sync(size(TiledMma1{}), 0);
  };

  synchronize();
  cfk::copy_nosync(tOrO, tOsOAcc);
  synchronize();

  Tensor mO = tmaStoreO.get_tma_tensor(shape(gmemLayoutO));
  auto blkCoordO = make_coord(blockIdxX, 0, blockIdxH, blockIdxB);
  Tensor gO = local_tile(mO, tileShapeO, blkCoordO);

  auto cta_tmaO = tmaStoreO.get_slice(0);

  Tensor tOgOX = cta_tmaO.partition_D(gO);
  Tensor tOgO = group_modes<1, rank(tOgOX)>(tOgOX); // (TMA,REST)
  assert(size<1>(tOgO) == 1);

  Tensor tOsOX = cta_tmaO.partition_S(sO);
  Tensor tOsO = group_modes<1, rank(tOsOX)>(tOsOX); // (TMA,REST)
  static_assert(size<1>(tOsO) == 1);

  int lane_predicate = cute::elect_one_sync();
  // Issue the TMA store.
  if (leaderWarp and lane_predicate) {
    cute::copy(tmaStoreO, tOsO, tOgO);
  }

  // Wait for TMA store to complete.
  tma_store_wait<0>();
}
