/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Driver for the pipelined and warp-specialized FMHA kernel.

    Based on the CUTLASS unit test for the PipelineTmaAsync class
    as it would be used in a warp-specialized loop.
*/

#pragma once

#define KERNEL_DBG_TRACE false

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/tensor.hpp>

#include <cutlass/cluster_launch.hpp>
#include <cutlass/util/reference/host/gemm.h>

#include "cutlass/core_io.h"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/print_error.hpp"

#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "fmha_consumer.h"
#include "fmha_epilogue.h"
#include "fmha_producer.h"
#include "shared_storage.h"

using namespace cute;
using namespace cutlass;

template <class Gemm1Type, class AccumType, class SoftType, class Gemm2Type,
          class OutputType, class TiledMma0, class TiledMma1, class TiledCopyQ,
          class TileShapeQ, class GmemLayoutQ, class SmemLayoutQ,
          class TiledCopyK, class TileShapeK, class GmemLayoutK,
          class SmemLayoutK, class TileShapeS, class GmemLayoutS,
          class SmemLayoutS, class TiledCopyV, class TileShapeV,
          class GmemLayoutV, class SmemLayoutV, class SmemLayoutVt,
          class TiledCopyO, class TileShapeO, class GmemLayoutO,
          class SmemLayoutO, class GmemLayoutMI, class ClusterShape>
__global__ static void __launch_bounds__(384, 1)
fmhaForwardPipelinedWspl(
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
  extern __shared__ char shared_memory[];
  using MainloopPipeline = typename cutlass::PipelineTmaAsync<stageCount>;
  // Change to this to use with CUTLASS 3.3 Pipeline API
  // using MainloopPipeline =
  //     typename cutlass::PipelineTmaAsync<stageCount, ClusterShape>;
  using PipelineState = typename cutlass::PipelineState<stageCount>;
  using BarrierType = typename MainloopPipeline::ProducerBarrierType;

  using SharedStorage =
      SharedStorage<Gemm1Type, Gemm2Type, OutputType, SmemLayoutQ, SmemLayoutK,
                    SmemLayoutS, SmemLayoutV, SmemLayoutO, ClusterShape>;
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(shared_memory);

  int warp_group_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
  int warp_idx_in_warpgroup =
      __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
  int warp_group_thread_idx = threadIdx.x % 128;
  dim3 block_id_in_cluster = cute::block_id_in_cluster();

  auto cluster_shape = ClusterShape{};

  // Unlike the unit test we always set this variable to 1
  // independent of cluster size.
  uint32_t const NumProducers = 1;

  // Get only TMA tensor mQ outside of producer loops.
  Tensor mQ = tmaLoadQ.get_tma_tensor(shape(gmemLayoutQ));

  // Compute TMA transaction bytes
  constexpr int per_cta_bytes =
      size(tileShapeK) * sizeof_bits_v<Gemm1Type> / 8 +
      size(tileShapeV) * sizeof_bits_v<Gemm2Type> / 8;
  uint32_t const TmaTransactionBytes = per_cta_bytes * NumProducers;

  // Construct SMEM tensors.
  Tensor sQ =
      make_tensor(make_smem_ptr(shared_storage.smem_q.data()), smemLayoutQ);
  Tensor sO =
      make_tensor(make_smem_ptr(shared_storage.smem_o.data()), smemLayoutO);
  Tensor sK =
      make_tensor(make_smem_ptr(shared_storage.kv.smem_k.data()), smemLayoutK);
#ifdef SINSMEM
  Tensor sS =
      make_tensor(make_smem_ptr(shared_storage.kv.smem_s.data()), smemLayoutS);
#else
  // Just a dummy sS (with smem_v). It's required only for shape later.
  Tensor sS =
      make_tensor(make_smem_ptr(shared_storage.kv.smem_k.data()), smemLayoutS);
#endif
  Tensor sV =
      make_tensor(make_smem_ptr(shared_storage.kv.smem_v.data()), smemLayoutV);

  // Tensor for V Transpose; used in GEMM-II.
  Tensor sVt =
      make_tensor(make_smem_ptr(shared_storage.kv.smem_v.data()), smemLayoutVt);

  // Get the block coordinates for this CTA.
  auto blockIdxX = uint64_t(blockIdx.x);
  auto blockIdxH = uint64_t(blockIdx.y);
  auto blockIdxB = uint64_t(blockIdx.z);

  // No pipelining for copying the block of Q.

  // Get the block of Q for this CTA using the block coordinates
  auto blkCoordQ = make_coord(blockIdxX, 0, blockIdxH, blockIdxB);
  Tensor gQ = local_tile(mQ, tileShapeQ, blkCoordQ);

  // Partition the copying of source tiles for Q among threads.
  auto cta_tmaQ = tmaLoadQ.get_slice(0);
  Tensor tQgQX = cta_tmaQ.partition_S(gQ);

  // Group the REST_X modes and the TMA_X modes to easily iterate through the
  // tiles
  Tensor tQgQ = group_modes<1, rank(tQgQX)>(tQgQX); // (TMA,REST)
  auto kTiles = size<1>(tQgQ);
  assert(kTiles == 1);
  assert(kTiles == size<2>(gQ));

  // Partition the copying of dest tile for Q among threads.
  Tensor tQsQX = cta_tmaQ.partition_D(sQ);
  Tensor tQsQ = group_modes<1, rank(tQsQX)>(tQsQX);

  // Copy Q tile from GMEM to SMEM.
  uint64_t *tma_load_mbar = shared_storage.tma_load_mbar;
  cfk::barrierInit(tma_load_mbar[0], 1); // This is REQUIRED.
  cfk::copy(tQgQ(_, 0), tQsQ(_, 0), tmaLoadQ, tma_load_mbar[0]);
  cute::wait_barrier(tma_load_mbar[0], 0); // This is REQUIRED.


  // In the WS kernel, we still initialize matmul objects
  // outside of the consumer body. This is done to avoid a
  // synchronization problem with the QINRMEM flag enabled.
  TiledMma0 tiledMma0;
  TiledMma1 tiledMma1;
  auto threadMma0 = tiledMma0.get_thread_slice(threadIdx.x);
  auto threadMma1 = tiledMma1.get_thread_slice(threadIdx.x);

  // Allocate "fragments/descriptors"
  // for first matmul.
  Tensor tSrQ = threadMma0.partition_fragment_A(sQ);
  Tensor tSrK = threadMma0.partition_fragment_B(sK);
  Tensor tSrS = partition_fragment_C(tiledMma0, tileShapeS);
  clear(tSrS);

#ifdef QINRMEM
  Tensor tSsQ = threadMma0.partition_A(sQ);
  cfk::copy_nosync(tSsQ, tSrQ);
  warpgroup_fence_operand(tSrQ);
#endif
  // Allocate "fragments/descriptors"
  // for second matmul.
  // Note: S becomes P.
  Tensor tOrV = threadMma1.partition_fragment_B(sVt);
  Tensor tOrS = threadMma1.partition_fragment_A(sS(_, _, 0));
  auto tOrPLayout = ReshapeTStoTP()(tSrS, tOrS);
  auto reg2reg = ReorgCFp8toAFp8();

  // FMHA OUTPUT (GEMM-II) accumulator.
  Tensor tOrO = partition_fragment_C(tiledMma1, tileShapeO);
  clear(tOrO);
  // Allocate space for per-thread rowMax and rowSum in rmem.
  Tensor rowMax = make_tensor<SoftType>(Shape<Int<2 * size<1>(tSrS)>>{});
  Tensor rowSum = make_fragment_like(rowMax);
  cute::fill(rowMax, -cutlass::platform::numeric_limits<SoftType>::infinity());
  cute::fill(rowSum, SoftType(0.0));

  // ------------ Pipelining begins -------------------------------

  // mbarrier.init
  typename MainloopPipeline::Params params;
  params.transaction_bytes = TmaTransactionBytes;
  if (warp_group_idx == 0) {
    params.role = MainloopPipeline::ThreadCategory::Producer;
  } else {
    params.role = MainloopPipeline::ThreadCategory::Consumer;
  }
  params.is_leader = warp_group_thread_idx == 0;
  params.num_consumers = NumMmaThreads;

  MainloopPipeline pipeline(shared_storage.storage, params, cluster_shape);
  // Change to this to use with CUTLASS 3.3 Pipeline API
  // MainloopPipeline pipeline(shared_storage.storage, params);

  int blockIdxY = 0;

  __syncthreads();

  // Ensure All CTAs in Cluster have completed init before issuing commits
  cute::cluster_arrive_relaxed();
  cute::cluster_wait();

  // Producer warpgroup
  if (warp_group_idx == 0) {
    // method in cutlass/arch/reg_reconfig.h
    // calls setmaxnreg.dec.sync.aligned.u32
    cutlass::arch::warpgroup_reg_dealloc<80>();

    int lane_predicate = cute::elect_one_sync();
    if (warp_idx_in_warpgroup == 0 && lane_predicate) {

      int tma_k_prologue = min(stageCount, nTilesOfK);

      // For the DMA (prologue) - we start with an opposite phase - since we
      // skip all waits i.e., we know that the buffer is indeed empty
      PipelineState smem_pipe_write =
          make_producer_start_state<MainloopPipeline>();
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tma_k_prologue; ++i) {
        pipeline.producer_acquire(smem_pipe_write);
        BarrierType *tmaBar = pipeline.producer_get_barrier(smem_pipe_write);
        fmhaForwardProducer(sK(_, _, i), tmaLoadK, tileShapeK, gmemLayoutK,
                            sV(_, _, i), tmaLoadV, tileShapeV, gmemLayoutV,
                            blockIdxY++, tmaBar, ClusterShape());
        ++smem_pipe_write;
      }
      int tma_k_iter = nTilesOfK - tma_k_prologue;

      CUTE_NO_UNROLL
      for (; tma_k_iter > 0; --tma_k_iter) {
        pipeline.producer_acquire(smem_pipe_write);
        BarrierType *tmaBar = pipeline.producer_get_barrier(smem_pipe_write);
        auto stage = smem_pipe_write.index();
        fmhaForwardProducer(sK(_, _, stage), tmaLoadK, tileShapeK, gmemLayoutK,
                            sV(_, _, stage), tmaLoadV, tileShapeV, gmemLayoutV,
                            blockIdxY++, tmaBar, ClusterShape());
        ++smem_pipe_write;
      }

      // Tail Loop
      // Handles the case where we never enter the mainloop
      PipelineState tail =
          tma_k_prologue == stageCount ? smem_pipe_write : PipelineState{};
      for (int i = 0; i < tma_k_prologue; ++i) {
        pipeline.producer_acquire(tail);
        ++tail;
      }
    }
  }
  // Consumer warpgroup(s)
  else if (warp_group_idx == 1 || warp_group_idx == 2) {
    // method in cutlass/arch/reg_reconfig.h
    // calls setmaxnreg.inc.sync.aligned.u32
    cutlass::arch::warpgroup_reg_alloc<192>();

    PipelineState smem_pipe_read;
    PipelineState smem_pipe_release;

    // Init Shared Memory read stages & PhaseBit
    static constexpr uint32_t K_PIPE_MMAS = 1;
    static_assert(K_PIPE_MMAS < stageCount, "ERROR : Too many MMAs in flight");

    // Total number of gemm iterations
    auto gemm_k_iterations = nTilesOfK;

    int mma_k_prologue = min(K_PIPE_MMAS, gemm_k_iterations);

    CUTLASS_PRAGMA_UNROLL
    for (int iter = 0; iter < mma_k_prologue; ++iter) {
      pipeline.consumer_wait(smem_pipe_read);
      warpgroup_arrive();

      int stage = smem_pipe_read.index();
      fmhaForwardConsumer(Q, K, V, S, tSrQ, tSrK(_, _, _, stage), tSrS,
                          tOrV(_, _, _, stage), tOrO, tOrPLayout, reg2reg, rowMax,
                          rowSum, tileShapeS, gmemLayoutS, scale, blockIdxY++,
                          tiledMma0, tiledMma1, AccumType(0), SoftType(0));
      ++smem_pipe_read;
    }
    gemm_k_iterations -= mma_k_prologue;

    CUTLASS_PRAGMA_NO_UNROLL
    for (; gemm_k_iterations > 0; --gemm_k_iterations) {
      /// Wait on the smem_pipe_read stage / phase
      pipeline.consumer_wait(smem_pipe_read);
      warpgroup_arrive();

      int stage = smem_pipe_read.index();
      fmhaForwardConsumer(Q, K, V, S, tSrQ, tSrK(_, _, _, stage), tSrS,
                          tOrV(_, _, _, stage), tOrO, tOrPLayout, reg2reg, rowMax,
                          rowSum, tileShapeS, gmemLayoutS, scale, blockIdxY++,
                          tiledMma0, tiledMma1, AccumType(0), SoftType(0));

      warpgroup_wait<2 * K_PIPE_MMAS>();
      //    warpgroup_fence_operand(tSrS);
      //    warpgroup_fence_operand(tOrO);

      pipeline.consumer_release(smem_pipe_release);

      // Advance stages
      ++smem_pipe_read;
      ++smem_pipe_release;
    }

    warpgroup_wait<0>();
    // Tail Loop
    for (int i = 0; i < K_PIPE_MMAS; ++i) {
      pipeline.consumer_release(smem_pipe_release);
      ++smem_pipe_release;
    }

    // TMA Store epilogue
    bool leaderWarp = warp_group_idx == 1 && warp_idx_in_warpgroup == 0;
    fmhaForwardWriteOutTMA(tOrO, rowMax, rowSum, O, tileShapeO, gmemLayoutO,
                           tiledMma1, sO, tmaStoreO, leaderWarp, SoftType(0.0));

// Write out rowMax and rowSum to GMEM.
// Required for verification ONLY.
#ifdef COPYOUTMI
    fmhaForwardWriteOutSoftMax(rowMax, rowSum, mi_ptr, sPrimePtr, gmemLayoutMi,
                               tiledMma0, tileShapeO);
#endif
  }
}
