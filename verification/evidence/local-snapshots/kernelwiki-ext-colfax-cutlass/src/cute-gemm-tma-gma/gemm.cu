/***************************************************************************************************
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
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
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>

#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/print_error.hpp"
#if defined(CUTLASS_ENABLE_CUBLAS) && CUTLASS_ENABLE_CUBLAS != 0
#include "cutlass/util/cublas_wrappers.hpp"
#endif
#include "cutlass/util/helper_cuda.hpp"
#include "gemm/copy_tensor.hpp"
#include "gemm/gemm_tensor.hpp"
#include "utils/cuda_launch.hpp"

template <class ElementTypeA, class ElementTypeB, class SmemLayoutA,
          class SmemLayoutB>
struct SharedStorage {
  cute::array_aligned<ElementTypeA, cute::cosize_v<SmemLayoutA>> smem_a;
  cute::array_aligned<ElementTypeB, cute::cosize_v<SmemLayoutB>> smem_b;
  cute::uint64_t tma_load_mbar[1];
};

template <class TiledMma, class ClusterShape, class TA, class TiledCopyA,
          class TileShapeA, class GmemLayoutA, class SmemLayoutA, class TB,
          class TiledCopyB, class TileShapeB, class GmemLayoutB,
          class SmemLayoutB, class TC, class TileShapeC, class GmemLayoutC>
__global__ static void gemm_device(
    TA const *A, TA *A_out, CUTE_GRID_CONSTANT TiledCopyA const tma_load_a,
    TileShapeA tile_shape_a, GmemLayoutA gmem_layout_a,
    SmemLayoutA smem_layout_a, TB const *B, TB *B_out,
    CUTE_GRID_CONSTANT TiledCopyB const tma_load_b, TileShapeB tile_shape_b,
    GmemLayoutB gmem_layout_b, SmemLayoutB smem_layout_b, TC *C,
    TileShapeC tile_shape_c, GmemLayoutC gmem_layout_c) {
  using namespace cute;
  using X = Underscore;
  CUTE_STATIC_ASSERT_V(product_each(shape(tile_shape_a)) ==
                       product_each(shape(smem_layout_a)));

  // Use Shared Storage structure to allocate and distribute aligned SMEM
  // addresses
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(shared_memory);
  // Shared memory barriers use 64bits in SMEM for synchronization
  uint64_t *tma_load_mbar = shared_storage.tma_load_mbar;

  uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
  constexpr uint32_t cluster_shape_x = get<0>(ClusterShape{});
  uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x,
                                  block_rank_in_cluster / cluster_shape_x};

  // Construct SMEM tensor
  Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem_a.data()),
                          smem_layout_a); // (CTA_TILE_M,CTA_TILE_N,...)

  // TMA requires special handling of strides to deal with coord codomain
  // mapping Represent the full tensors -- get these from TMA
  Tensor mA = tma_load_a.get_tma_tensor(shape(gmem_layout_a));
  Tensor mA_out = make_tensor(make_gmem_ptr(A_out), gmem_layout_a);
  auto blk_coord_a =
      make_coord(uint64_t(blockIdx.x), _, uint64_t(blockIdx.z)); // (m,k,l)

  constexpr int R = rank_v<TileShapeA>;
  Tensor gA =
      local_tile(mA, tile_shape_a,
                 blk_coord_a); // (CTA_TILE_M,CTA_TILE_N,...REST_M,REST_N,...)
  Tensor gA_out =
      local_tile(mA_out, tile_shape_a,
                 blk_coord_a); // (CTA_TILE_M,CTA_TILE_N,...REST_M,REST_N,...)

  //
  // Prepare the TMA_LOAD for A
  //

  auto cta_tma_a = tma_load_a.get_slice(cluster_local_block_id.y); // CTA slice

  Tensor tAgA_x = cta_tma_a.partition_S(gA); // (TMA,TMA_M,TMA_N,REST_M,REST_N)
  Tensor tAsA_x = cta_tma_a.partition_D(sA); // (TMA,TMA_M,TMA_N)

#if 0 
  if (thread0()) {
    print(tma_load_a);
    print("TILE  :  "); print(tile_shape_a); print("\n");
    print("  mA  :  "); print(  mA.data());   print(" o "); print(  mA.layout());   print("\n");
    print("  gA  :  "); print(  gA.data());   print(" o "); print(  gA.layout());   print("\n");
    print("tAgA_x:  "); print(tAgA_x.data()); print(" o "); print(tAgA_x.layout()); print("\n");
    print("  sA  :  "); print(  sA.data());   print(" o "); print(  sA.layout());   print("\n");
    print("tAsA_x:  "); print(tAsA_x.data()); print(" o "); print(tAsA_x.layout()); print("\n");
  }
#endif

  //
  // Perform the TMA_LOAD
  //

  // INPUT: Group the REST_X modes and the TMA_X modes to easily iterate through
  // the tiles
  Tensor tAgA = group_modes<1, rank(tAgA_x)>(tAgA_x); // (TMA,REST)
  Tensor tAsA = group_modes<1, rank(tAsA_x)>(tAsA_x); // (TMA,REST)
  static_assert(size<1>(tAsA) == 1);

  // OUTPUT: Group the CTA_TILE_X modes and REST_X modes for output
  Tensor gA_out_collapsed = group_modes<0, R>(
      group_modes<R, rank(gA_out)>(gA_out)); // (CTA_TILE, REST)

#if 0 
  if (thread0()) {
    print("tAgA  :  "); print(tAgA.data()); print(" o "); print(tAgA.layout()); print("\n");
    print("tAsA  :  "); print(tAsA.data()); print(" o "); print(tAsA.layout()); print("\n");
    print("gA_out_collapsed  :  "); print(gA_out_collapsed.data()); print(" o "); print(gA_out_collapsed.layout()); print("\n");
  }
#endif
  // Construct SMEM tensor for B
  Tensor sB = make_tensor(make_smem_ptr(shared_storage.smem_b.data()),
                          smem_layout_b); // (CTA_TILE_M,CTA_TILE_N,...)

  // TMA requires special handling of strides to deal with coord codomain
  // mapping Represent the full tensors -- get these from TMA
  Tensor mB = tma_load_b.get_tma_tensor(shape(gmem_layout_b));
  Tensor mB_out = make_tensor(make_gmem_ptr(B_out), gmem_layout_b);
  auto blk_coord_b =
      make_coord(uint64_t(blockIdx.y), _, uint64_t(blockIdx.z)); // (n, k, l)

  constexpr int RB = rank_v<TileShapeB>;
  Tensor gB =
      local_tile(mB, tile_shape_b,
                 blk_coord_b); // (CTA_TILE_M,CTA_TILE_N,...REST_M,REST_N,...)
  Tensor gB_out =
      local_tile(mB_out, tile_shape_b,
                 blk_coord_b); // (CTA_TILE_M,CTA_TILE_N,...REST_M,REST_N,...)

  //
  // Prepare the TMA_LOAD for A
  //

  auto cta_tma_b = tma_load_b.get_slice(cluster_local_block_id.x); // CTA slice

  Tensor tBgB_x = cta_tma_b.partition_S(gB); // (TMA,TMA_M,TMA_N,REST_M,REST_N)
  Tensor tBsB_x = cta_tma_b.partition_D(sB); // (TMA,TMA_M,TMA_N)

#if 0 
  if (thread0()) {
    print(tma_load_a);
    print("TILE  :  "); print(tile_shape_a); print("\n");
    print("  mB  :  "); print(  mB.data());   print(" o "); print(  mB.layout());   print("\n");
    print("  gB  :  "); print(  gB.data());   print(" o "); print(  gB.layout());   print("\n");
    print("tBgB_x:  "); print(tBgB_x.data()); print(" o "); print(tBgB_x.layout()); print("\n");
    print("  sB  :  "); print(  sB.data());   print(" o "); print(  sB.layout());   print("\n");
    print("tBsB_x:  "); print(tBsB_x.data()); print(" o "); print(tBsB_x.layout()); print("\n");
  }
#endif

  //
  // Perform the TMA_LOAD
  //

  // INPUT: Group the REST_X modes and the TMA_X modes to easily iterate through
  // the tiles
  Tensor tBgB = group_modes<1, rank(tBgB_x)>(tBgB_x); // (TMA,REST)
  Tensor tBsB = group_modes<1, rank(tBsB_x)>(tBsB_x); // (TMA,REST)
  static_assert(size<1>(tBsB) == 1);

  // OUTPUT: Group the CTA_TILE_X modes and REST_X modes for output
  Tensor gB_out_collapsed = group_modes<0, RB>(
      group_modes<RB, rank(gB_out)>(gB_out)); // (CTA_TILE, REST)

#if 0
  if (thread0()) {
    print("tBgB  :  "); print(tBgB.data()); print(" o "); print(tBgB.layout()); print("\n");
    print("tBsB  :  "); print(tBsB.data()); print(" o "); print(tBsB.layout()); print("\n");
    print("gB_out_collapsed  :  "); print(gB_out_collapsed.data()); print(" o "); print(gB_out_collapsed.layout()); print("\n");
  }
#endif

  TiledMma tiled_mma;
  auto thread_mma = tiled_mma.get_thread_slice(threadIdx.x);

  Tensor tCsA = thread_mma.partition_A(sA); // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCsB = thread_mma.partition_B(sB); // (MMA,MMA_N,MMA_K,PIPE)

  // Allocate "fragments/descriptors"
  Tensor tCrA = thread_mma.make_fragment_A(tCsA); // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCrB = thread_mma.make_fragment_B(tCsB); // (MMA,MMA_N,MMA_K,PIPE)
  // auto tCrC = partition_fragment_C(tiled_mma, take<0,2>(tile_shape_c)); //
  // (MMA,MMA_M,MMA_N)
#if 0
  if (thread0()) {
    print("tCsA  :  ");
    print(tCsA.data());
    print(" o ");
    print(tCsA.layout());
    print("\n");
    print("tCsB  :  ");
    print(tCsB.data());
    print(" o ");
    print(tCsB.layout());
    print("\n");
    print("tCrA  :  ");
    print(tCrA.data());
    print(" o ");
    print(tCrA.layout());
    print("\n");
    print("tCrB  :  ");
    print(tCrB.data());
    print(" o ");
    print(tCrB.layout());
    print("\n");
  }
#endif

  Tensor mC = make_tensor(make_gmem_ptr(C), gmem_layout_c);
  auto blk_coord_c = make_coord(uint64_t(blockIdx.x), uint64_t(blockIdx.y),
                                uint64_t(blockIdx.z)); // (m,n,l)

  Tensor gC =
      local_tile(mC, tile_shape_c,
                 blk_coord_c); // (CTA_TILE_M,CTA_TILE_N,...REST_M,REST_N,...)

  //
  // Prepare the TMA_STORE
  //

  Tensor tCgC = thread_mma.partition_C(gC);
  auto tCrC = partition_fragment_C(tiled_mma, tile_shape_c);

#if 0
  if (thread0()) {
    print("tCgC  :  ");
    print(tCgC.data());
    print(" o ");
    print(tCgC.layout());
    print("\n");
    print("tCrC  :  ");
    print(tCrC.data());
    print(" o ");
    print(tCrC.layout());
    print("\n");
  }
#endif

  uint16_t mcast_mask_a = 0;
  uint16_t mcast_mask_b = 0;

  // Issue TmaLoads
  // Maps the tile -> block, value
  if constexpr (cute::is_same_v<TiledCopyA, SM90_TMA_LOAD_MULTICAST>) {
    auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
    for (int n = 0; n < size<1>(block_layout); ++n) {
      mcast_mask_a |=
          (uint16_t(1) << block_layout(cluster_local_block_id.x, n, Int<0>{}));
    }
  }

  if constexpr (cute::is_same_v<TiledCopyB, SM90_TMA_LOAD_MULTICAST>) {
    auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
    for (int m = 0; m < size<0>(block_layout); ++m) {
      mcast_mask_b |=
          (uint16_t(1) << block_layout(m, cluster_local_block_id.y, Int<0>{}));
    }
  }

  cute::prefetch_tma_descriptor(tma_load_a.get_tma_descriptor());
  cute::prefetch_tma_descriptor(tma_load_b.get_tma_descriptor());
  __syncthreads();

//
// Prepare the TMA_LOAD for A
// Loop over the TMA stages, using smem as our buffer
#pragma unroll
  for (int stage = 0; stage < size<1>(tAgA); ++stage) {
    // Set the bytes transferred in this TMA transaction (may involve multiple
    // issues)
    constexpr int kTmaTransactionBytes =
        size(sA) * sizeof_bits_v<TA> / 8 + size(sB) * sizeof_bits_v<TB> / 8;

    cfk::copy(tAgA(_, stage), tBgB(_, stage), tAsA(_, 0), tBsB(_, 0),
              tma_load_a, tma_load_b, tma_load_mbar, mcast_mask_a,
              mcast_mask_b);
    cfk::gemm(tiled_mma, tCrA, tCrB, tCrC);

#ifdef COPYOUTAB
    for (int i = threadIdx.x; i < size(sA); i += blockDim.x) {
      gA_out_collapsed(i, stage) = sA(i);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < size(sB); i += blockDim.x) {
      gB_out_collapsed(i, stage) = sB(i);
    }
    __syncthreads();
#endif
  }

#pragma unroll
  for (int i = 0; i < size(tCrC); ++i) {
    tCgC(i) = tCrC(i);
  }

  __syncthreads();
}

template <typename TA, typename TB, typename TC, typename Alpha, typename Beta>
void gemm(int m, int n, int k, Alpha alpha, TA const *A, int ldA, TB const *B,
          int ldB, Beta beta, TC *C, int ldC, TA *A_out, TB *B_out, int L,
          cudaStream_t stream = 0) {
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);

  using ClusterShape = Shape<_1, _1, _1>;
  // Define TileShapes
  using bM = Int<128>;
  using bN = cute::conditional_t<cute::is_same_v<TA, cutlass::half_t>, Int<256>,
                                 Int<128>>;
  using bK = cute::conditional_t<cute::is_same_v<TA, cutlass::half_t>, Int<64>,
                                 Int<32>>;

  using MmaA = cute::conditional_t<cute::is_same_v<TA, float>, tfloat32_t, TA>;
  using MmaB = cute::conditional_t<cute::is_same_v<TB, float>, tfloat32_t, TB>;

  // row-major for A.
  auto ptr_A = reinterpret_cast<MmaA const *>(A);
  auto ptr_B = reinterpret_cast<MmaB const *>(B);
  auto ptr_A_out = reinterpret_cast<MmaA *>(A_out);
  auto ptr_B_out = reinterpret_cast<MmaB *>(B_out);
  auto tile_shape_a = make_shape(bM{}, bK{});
  auto smem_layout_a =
      tile_to_shape(GMMA::Layout_K_SW64_Atom<MmaA>{}, tile_shape_a);
  Layout gmem_layout_a =
      make_layout(make_shape(M, K, L), make_stride<uint64_t>(K, 1, M * K));
  Tensor gA = make_tensor(ptr_A, gmem_layout_a);
  auto tma_a =
      make_tma_copy(SM90_TMA_LOAD{}, gA, smem_layout_a, tile_shape_a, Int<1>{});
  auto tma_a_mcast = make_tma_copy(SM90_TMA_LOAD_MULTICAST{}, gA, smem_layout_a,
                                   tile_shape_a, size<1>(ClusterShape{}));

  // row-major for B, but transpose, so col-major.
  auto tile_shape_b = make_shape(bN{}, bK{});
  auto smem_layout_b =
      tile_to_shape(GMMA::Layout_K_SW64_Atom<MmaB>{}, tile_shape_b);
  Layout gmem_layout_b =
      make_layout(make_shape(N, K, L), make_stride<uint64_t>(K, 1, N * K));
  Tensor gB = make_tensor(ptr_B, gmem_layout_b);
  auto tma_b =
      make_tma_copy(SM90_TMA_LOAD{}, gB, smem_layout_b, tile_shape_b, Int<1>{});
  auto tma_b_mcast = make_tma_copy(SM90_TMA_LOAD_MULTICAST{}, gB, smem_layout_b,
                                   tile_shape_b, size<0>(ClusterShape{}));

  // col-major for C.
  auto tile_shape_c = make_shape(bM{}, bN{});
  Layout gmem_layout_c =
      make_layout(make_shape(M, N, L), make_stride<uint64_t>(1, M, M * N));
  Tensor gC = make_tensor(C, gmem_layout_c);

  int smem_size = int(sizeof(SharedStorage<MmaA, MmaB, decltype(smem_layout_a),
                                           decltype(smem_layout_b)>));

  // For fp32 types, map to tf32 MMA value type
  using TiledMma = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<MmaA, MmaB, TC, Shape<bM, bN, bK>>(),
      Layout<Shape<_1, _1, _1>>()));

#if 0
  print(tma_a);
  print(tile_shape_a);
  print(smem_layout_a);
  // print (TiledMma{});
#endif

  int cuda_grid_x;
  int cuda_grid_y;
  int cuda_grid_z;
  cudaDeviceGetAttribute(&cuda_grid_x, cudaDevAttrMaxGridDimX, 0);
  cudaDeviceGetAttribute(&cuda_grid_y, cudaDevAttrMaxGridDimY, 0);
  cudaDeviceGetAttribute(&cuda_grid_z, cudaDevAttrMaxGridDimZ, 0);
  // std::cout << cuda_grid_x << " " << cuda_grid_y << " " << cuda_grid_z <<
  // std::endl;

  if (size(ClusterShape{}) == 1) {
    void const *kernel = (void const *)gemm_device<
        TiledMma, ClusterShape, MmaA, decltype(tma_a), decltype(tile_shape_a),
        decltype(gmem_layout_a), decltype(smem_layout_a), MmaB, decltype(tma_b),
        decltype(tile_shape_b), decltype(gmem_layout_b),
        decltype(smem_layout_b), TC, decltype(tile_shape_c),
        decltype(gmem_layout_c)>;
    cfk::utils::set_smem_size(smem_size, kernel);

    dim3 block_dims(size(TiledMma{}));
    dim3 grid_dims(ceil_div(size(M), size(bM{})), ceil_div(size(N), size(bN{})),
                   L);
    dim3 cluster_dims(cute::size<0>(ClusterShape{}),
                      cute::size<1>(ClusterShape{}),
                      cute::size<2>(ClusterShape{}));
    cutlass::ClusterLaunchParams params{grid_dims, block_dims, cluster_dims,
                                        smem_size};

    cutlass::Status status = cutlass::launch_kernel_on_cluster(
        params, kernel, ptr_A, ptr_A_out, tma_a, tile_shape_a, gmem_layout_a,
        smem_layout_a, ptr_B, ptr_B_out, tma_b, tile_shape_b, gmem_layout_b,
        smem_layout_b, C, tile_shape_c, gmem_layout_c);
  } else {
    void const *kernel = (void const *)
        gemm_device<TiledMma, ClusterShape, MmaA, decltype(tma_a_mcast),
                    decltype(tile_shape_a), decltype(gmem_layout_a),
                    decltype(smem_layout_a), MmaB, decltype(tma_b_mcast),
                    decltype(tile_shape_b), decltype(gmem_layout_b),
                    decltype(smem_layout_b), TC, decltype(tile_shape_c),
                    decltype(gmem_layout_c)>;
    cfk::utils::set_smem_size(smem_size, kernel);

    dim3 block_dims(size(TiledMma{}));
    dim3 grid_dims(ceil_div(size(M), size(bM{})), ceil_div(size(N), size(bN{})),
                   L);
    dim3 cluster_dims(cute::size<0>(ClusterShape{}),
                      cute::size<1>(ClusterShape{}),
                      cute::size<2>(ClusterShape{}));
    cutlass::ClusterLaunchParams params{grid_dims, block_dims, cluster_dims,
                                        smem_size};

    cutlass::Status status = cutlass::launch_kernel_on_cluster(
        params, kernel, ptr_A, ptr_A_out, tma_a_mcast, tile_shape_a,
        gmem_layout_a, smem_layout_a, ptr_B, ptr_B_out, tma_b_mcast,
        tile_shape_b, gmem_layout_b, smem_layout_b, C, tile_shape_c,
        gmem_layout_c);
  }
}

#include <cassert>
#include <cstdio>
#include <cstdlib>

template <typename TA, typename TB, typename TC, typename TI>
void test_gemm(int m, int n, int k, int l) {
  cute::device_init(0);

  std::cout << "M = " << m << std::endl;
  std::cout << "N = " << n << std::endl;
  std::cout << "K = " << k << std::endl;
  std::cout << "L = " << l << std::endl;

  thrust::host_vector<TA> h_A(int64_t(m * k) * uint64_t(l));
  thrust::host_vector<TB> h_B(uint64_t(n * k) * uint64_t(l));
  thrust::host_vector<TC> h_C(uint64_t(m * n) * uint64_t(l));

  for (uint64_t j = 0; j < uint64_t(m * k) * uint64_t(l); ++j)
    h_A[j] = static_cast<TA>(2 * (rand() / double(RAND_MAX)) - 1);
  for (uint64_t j = 0; j < uint64_t(n * k) * uint64_t(l); ++j)
    h_B[j] = static_cast<TB>(2 * (rand() / double(RAND_MAX)) - 1);
  for (uint64_t j = 0; j < uint64_t(m * n) * uint64_t(l); ++j)
    h_C[j] = static_cast<TC>(-1);

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;
  thrust::device_vector<TA> d_A_out(m * k);
  thrust::device_vector<TB> d_B_out(n * k);

  TI alpha = TI(1.0);
  TI beta = TI(0.0);

  double gflops = (2.0 * m * n * k * l) * 1e-9;

  const int timing_iterations = 1000;
  GPU_Clock timer;

#if 1
#if defined(CUTLASS_ENABLE_CUBLAS) && CUTLASS_ENABLE_CUBLAS != 0
  //
  // cuBLas
  //

  cublasHandle_t handle;
  cublasCreate(&handle);

  // Run once
  d_C = h_C;
  if (l == 1) {
    blam::cublas::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                       d_A.data().get(), k, d_B.data().get(), k, &beta,
                       d_C.data().get(), m);
  } else {
    blam::cublas::gemm_batch(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                             d_A.data().get(), k, m * k, d_B.data().get(), k,
                             n * k, &beta, d_C.data().get(), m, m * n, l);
  }
  CUTE_CHECK_LAST();

  thrust::host_vector<TC> cublas_result = d_C;

  // Timing iterations
  timer.start();
  if (l == 1) {
    for (int i = 0; i < timing_iterations; ++i) {
      blam::cublas::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                         d_A.data().get(), k, d_B.data().get(), k, &beta,
                         d_C.data().get(), m);
    }
  } else {
    for (int i = 0; i < timing_iterations; ++i) {
      blam::cublas::gemm_batch(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
                               &alpha, d_A.data().get(), k, m * k,
                               d_B.data().get(), k, n * k, &beta,
                               d_C.data().get(), m, m * n, l);
    }
  }
  double cublas_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUBLAS_GEMM:   [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cublas_time,
         cublas_time * 1000);

#else

  std::cout
      << "Verification by comparison with cuBLAS is disabled, "
         "either because the CMake option CUTLASS_ENABLE_CUBLAS "
         "was explicitly set to OFF, or because CMake could not find cuBLAS.  "
         "If you would like to enable verification with cuBLAS, "
         "please set the CMake option CUTLASS_ENABLE_CUBLAS to ON, "
         "rerun CMake, and recompile this example.\n";

#endif // CUTLASS_ENABLE_CUBLAS
#endif
  //
  // CuTe
  //

  // Run once (and check)
  d_C = h_C;
  gemm(m, n, k, alpha, d_A.data().get(), m, d_B.data().get(), n, beta,
       d_C.data().get(), m, d_A_out.data().get(), d_B_out.data().get(), l);
  CUTE_CHECK_LAST();
  thrust::host_vector<TC> cute_result = d_C;
  thrust::host_vector<TA> h_A_out = d_A_out;
  thrust::host_vector<TB> h_B_out = d_B_out;

#ifdef COPYOUTAB
  for (int j = 0; j < m * k; ++j) {
    if (h_A[j] != h_A_out[j]) {
      std::cout << "failed " << h_A[j] << ", " << h_A_out[j] << std::endl;
      break;
    }
    // std::cout << h_A[j] << " " << h_A_out[j] << std::endl;
  }
  for (int j = 0; j < n * k; ++j) {
    if (h_B[j] != h_B_out[j]) {
      std::cout << "failed " << std::endl;
      break;
    }
    // std::cout << h_A[j] << " " << h_A_out[j] << std::endl;
  }
#endif

#if 1
  // Timing iterations
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    gemm(m, n, k, alpha, d_A.data().get(), m, d_B.data().get(), n, beta,
         d_C.data().get(), m, d_A_out.data().get(), d_B_out.data().get(), l);
  }
#endif
  double cute_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time,
         cute_time * 1000);
  cute_result = d_C;

#if 1
#if defined(CUTLASS_ENABLE_CUBLAS) && CUTLASS_ENABLE_CUBLAS != 0
  printf("Empirical Perf: %.1f%%\n", (cublas_time / cute_time) * 100);

  auto host_matrix_to_const_row_major_cute_tensor =
      [](const auto &X, int num_rows, int num_cols, int LDX, long batchStride) {
        const auto shape = cute::Shape<int, int>{num_rows, num_cols};
        const auto strides = cute::Stride<int, int>{LDX, 1};
        return cute::make_tensor(X.data() + batchStride,
                                 cute::make_layout(shape, strides));
      };

  auto host_matrix_to_const_column_major_cute_tensor =
      [](const auto &X, int num_rows, int num_cols, int LDX, long batchStride) {
        const auto shape = cute::Shape<int, int>{num_rows, num_cols};
        const auto strides = cute::Stride<int, int>{1, LDX};
        return cute::make_tensor(X.data() + batchStride,
                                 cute::make_layout(shape, strides));
      };

  auto type_name = cute::is_same_v<TC, cutlass::half_t> ? "half_t" : "float";
  using namespace cute;
  if (l <= 10) {
    for (int i = 0; i < l; ++i) {
      const auto A_view = host_matrix_to_const_row_major_cute_tensor(
          h_A, m, k, k, i * uint64_t(m * k));
      // B^T is k x n, so B is n x k.
      const auto B_view = host_matrix_to_const_row_major_cute_tensor(
          h_B, n, k, k, i * uint64_t(n * k));
      const auto C_computed_view =
          host_matrix_to_const_column_major_cute_tensor(cute_result, m, n, m,
                                                        i * uint64_t(m * n));
      const auto C_expected_view =
          host_matrix_to_const_column_major_cute_tensor(cublas_result, m, n, m,
                                                        i * uint64_t(m * n));

      std::cout << " i " << i << std::endl;
      print_matrix_multiply_mollified_relative_error(
          type_name, A_view, B_view, C_computed_view, C_expected_view);
    }
  }
  for (int64_t j = 0; j < int64_t(m) * int64_t(n) * int64_t(l); ++j) {
    if (cute_result[j] != cublas_result[j]) {
      std::cout << "failed " << cute_result[j] << " " << cublas_result[j]
                << std::endl;
      break;
    }
  }
#endif
#endif // CUTLASS_ENABLE_CUBLAS
}

int main(int argc, char **argv) {
  int type = 1; // 1 means tf32, 2 means half
  if (argc >= 2)
    sscanf(argv[1], "%d", &type);

  int m = 5120;
  if (argc >= 3)
    sscanf(argv[2], "%d", &m);

  int n = 5120;
  if (argc >= 4)
    sscanf(argv[3], "%d", &n);

  int k = 4096;
  if (argc >= 5)
    sscanf(argv[4], "%d", &k);

  int l = 1;
  if (argc >= 6)
    sscanf(argv[5], "%d", &l);

  if (type == 1) {
    test_gemm<float, float, float, float>(m, n, k, l);
  } else if (type == 2) {
    //      test_gemm<cutlass::tfloat32_t, cutlass::tfloat32_t, float, float>(m,
    //      n, k);
  } else if (type == 3) {
    test_gemm<cutlass::half_t, cutlass::half_t, cutlass::half_t,
              cutlass::half_t>(m, n, k, l);
  } else {
    std::cout << "invalid type value (1 | 2 | 3 are the only legal values)";
    exit(-1);
  }

  return 0;
}
