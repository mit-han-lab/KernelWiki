#pragma once
/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

// copy kernel adapted from https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/tiled_copy.cu

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <chrono>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cutlass/numeric_types.h"
#include <cute/arch/cluster_sm90.hpp>
#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

#include "cutlass/detail/layout.hpp"

#include "shared_storage.h"
#include "util.h"

template <class TensorA, class TensorB, class ThreadLayout, class VecLayout, class SmemLayoutA, class SmemLayoutB>
__global__ static void __launch_bounds__(256, 1)
    copySmemFmaNaiveKernel(TensorA const A, TensorB const B, ThreadLayout, VecLayout, SmemLayoutA, SmemLayoutB) {
  using namespace cute;
  using Element = typename TensorA::value_type;

  // Use Shared Storage structure to allocate aligned SMEM addresses.
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorageCopyFmaA<Element, SmemLayoutA, ThreadLayout>;
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(shared_memory);

  Tensor gA = A(make_coord(_, _), blockIdx.x, blockIdx.y); // (bM, bN)
  Tensor gB = B(make_coord(_, _), blockIdx.x, blockIdx.y); // (bM, bN)

  Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem_a.data()), SmemLayoutA{}); // (bM, bN)

  auto tiled_copy_load =
    make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, Element>{},
      ThreadLayout{},
      VecLayout{});

  auto tiled_copy_store =
    make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<32>, Element>{},
      ThreadLayout{});

  // Construct a Tensor corresponding to each thread's slice.
  auto thr_copy_load = tiled_copy_load.get_thread_slice(threadIdx.x);
  auto thr_copy_store = tiled_copy_store.get_thread_slice(threadIdx.x);

  // gmem to smem A
  Tensor tAgA = thr_copy_load.partition_S(gA);
  Tensor tAsA = thr_copy_load.partition_D(sA);

  copy(tiled_copy_load, tAgA, tAsA);

  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();
  // have A vector in smem now

  Tensor accum = make_fragment_like(sA(_, _0{}));
  clear(accum);
  
  static_assert(size<0>(sA) == 1, "Suppose bM = 1.");

  Tensor sO = make_tensor(make_smem_ptr(shared_storage.smem_out.data()), ThreadLayout{});
  Tensor tOsO = thr_copy_store.partition_S(sO);
  int tix = threadIdx.x;
  float mult = static_cast<float>(tix) / 32.f;

  #pragma unroll
  for(int i = 0; i < size<0>(sA); ++i) {
    accum(i) = sA(i, 0);
    #pragma unroll
    for(int j = 0; j < size<1>(sA); ++j) {
      accum(i) += mult * accum(i);
    }
    // dummy store so we don't optimize out compute loop
    sO(tix) = accum(i);
    __syncthreads();
    auto gB_slice = gB(i, _);
    Tensor gO = make_tensor(make_gmem_ptr(gB_slice.data()), ThreadLayout{});
    Tensor tOgO = thr_copy_store.partition_D(gO);
    copy(tiled_copy_store, tOsO, tOgO);

  }

}

template <typename T> void copy_smem_fma_naive(TransposeParams<T> params) {

  using Element = float;
  using namespace cute;

  //
  // Make tensors
  //

  // Will use with N = 1024
  auto tensor_shape = make_shape(params.M, params.N);
  auto gmemLayoutS = make_layout(tensor_shape, LayoutRight{});
  auto gmemLayoutD = make_layout(tensor_shape, LayoutRight{});
  Tensor tensor_S = make_tensor(make_gmem_ptr(params.input), gmemLayoutS);
  Tensor tensor_D = make_tensor(make_gmem_ptr(params.output), gmemLayoutD);
 
  //
  // Tile tensors
  //
  using bM = Int<1>;
  using bN = Int<1024>;

  auto block_shape = make_shape(bM{}, bN{});       // (bM, bN)

  auto smem_layout = make_layout(block_shape, LayoutRight{});

  Tensor tiled_tensor_S =
      tiled_divide(tensor_S, block_shape); // ((bM, bN), m', n')
  Tensor tiled_tensor_D =
      tiled_divide(tensor_D, block_shape); // ((bN, bM), n', m')

  auto threadLayout =
      make_layout(make_shape(Int<1>{}, Int<256>{}), LayoutRight{});
  // auto threadLayout =
  //     make_layout(make_shape(Int<1>{}, Int<128>{}), LayoutRight{});

  auto vec_layout = make_layout(make_shape(Int<1>{}, Int<4>{}));

  //
  // Determine grid and block dimensions
  //

  dim3 gridDim(
      size<1>(tiled_tensor_S),
      size<2>(tiled_tensor_S)); // Grid shape corresponds to modes m' and n'
  dim3 blockDim(size(threadLayout)); // 256 threads

  size_t smem_size = int(sizeof(SharedStorageCopyFmaA<Element, decltype(smem_layout), decltype(threadLayout)>));

  copySmemFmaNaiveKernel<<<gridDim, blockDim, smem_size>>>(tiled_tensor_S, tiled_tensor_D,
                                       threadLayout, vec_layout, smem_layout, smem_layout);
}
