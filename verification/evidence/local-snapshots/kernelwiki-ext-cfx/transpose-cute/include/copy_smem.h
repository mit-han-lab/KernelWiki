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

template <class TensorS, class TensorD, class ThreadLayout, class VecLayout, class SmemLayout>
__global__ static void __launch_bounds__(256, 1)
    copySmemKernel(TensorS const S, TensorD const D, ThreadLayout, VecLayout, SmemLayout) {
  using namespace cute;
  using Element = typename TensorS::value_type;

  // Use Shared Storage structure to allocate aligned SMEM addresses.
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorageCopy<Element, SmemLayout>;
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(shared_memory);

  Tensor gS = S(make_coord(_, _), blockIdx.x, blockIdx.y); // (bM, bN)
  Tensor gD = D(make_coord(_, _), blockIdx.x, blockIdx.y); // (bN, bM)

  Tensor sS = make_tensor(make_smem_ptr(shared_storage.smem.data()), SmemLayout{}); // (bN, bM)

  // SM80_CP_ASYNC_CACHEALWAYS
  auto tiled_copy_load =
    make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, Element>{},
      ThreadLayout{},
      VecLayout{});

  auto tiled_copy_store =
    make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
      ThreadLayout{},
      VecLayout{});

  // Construct a Tensor corresponding to each thread's slice.
  auto thr_copy_load = tiled_copy_load.get_thread_slice(threadIdx.x);
  auto thr_copy_store = tiled_copy_store.get_thread_slice(threadIdx.x);

  Tensor tSgS = thr_copy_load.partition_S(gS);
  Tensor tSsS = thr_copy_load.partition_D(sS);

  Tensor tDsS = thr_copy_store.partition_D(sS);
  Tensor tDgD = thr_copy_store.partition_D(gD);

  copy(tiled_copy_load, tSgS, tSsS);

  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();

  copy(tiled_copy_store, tDsS, tDgD);

}

template <typename T> void copy_smem(TransposeParams<T> params) {

  using Element = float;
  using namespace cute;

  //
  // Make tensors
  //
  auto tensor_shape = make_shape(params.M, params.N);
  auto gmemLayoutS = make_layout(tensor_shape, LayoutRight{});
  auto gmemLayoutD = make_layout(tensor_shape, LayoutRight{});
  Tensor tensor_S = make_tensor(make_gmem_ptr(params.input), gmemLayoutS);
  Tensor tensor_D = make_tensor(make_gmem_ptr(params.output), gmemLayoutD);
 
  //
  // Tile tensors
  //
  // using bM = Int<8>;
  // using bN = Int<256>;

  using bM = Int<1>;
  using bN = Int<2048>;

  auto block_shape = make_shape(bM{}, bN{});       // (bM, bN)

  auto smem_layout = make_layout(block_shape, LayoutRight{});

  Tensor tiled_tensor_S =
      tiled_divide(tensor_S, block_shape); // ((bM, bN), m', n')
  Tensor tiled_tensor_D =
      tiled_divide(tensor_D, block_shape); // ((bN, bM), n', m')

  // auto threadLayout =
  //     make_layout(make_shape(Int<8>{}, Int<32>{}), LayoutRight{});
  auto threadLayout =
      make_layout(make_shape(Int<1>{}, Int<256>{}), LayoutRight{});

  auto vec_layout = make_layout(make_shape(Int<1>{}, Int<4>{}));

  //
  // Determine grid and block dimensions
  //

  dim3 gridDim(
      size<1>(tiled_tensor_S),
      size<2>(tiled_tensor_S)); // Grid shape corresponds to modes m' and n'
  dim3 blockDim(size(threadLayout)); // 256 threads

  size_t smem_size = int(sizeof(SharedStorageCopy<Element, decltype(smem_layout)>));

  copySmemKernel<<<gridDim, blockDim, smem_size>>>(tiled_tensor_S, tiled_tensor_D,
                                       threadLayout, vec_layout, smem_layout);
}
