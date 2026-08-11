/***************************************************************************************************
 * Copyright (c) 2017 - 2023 COLFAX
 * Distributed under MIT License
 **************************************************************************************************/

/*! \file
    \brief FMHA Attention Example.

    This workload computes a fused multi head attention.
    Because it keeps the attention matrix in shared memory, it's both faster
    and uses less global memory.

    The algorithm is based on `"FlashAttention-2: Faster Attention with Better
    Parallelism and Work Partitioning" <https://arxiv.org/abs/2307.08691>`_.

    Many details of the code are explained in the companion paper `"A Case
    Study in CUDA Kernel Fusion: Implementing FlashAttention-2 on NVIDIA
    Hopper Architecture using the CUTLASS Library"`.

*/

/////////////////////////////////////////////////////////////////////////////////////////////////

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>

#include "cutlass/numeric_types.h"
#include <cutlass/cutlass.h>

#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/transform/collective/sm90_wgmma_transpose.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

#include "utils/cuda_launch.hpp"
#include "utils/fmha_cutlass.hpp"
#include "utils/random.hpp"

#include "gemm/copy_tensor.hpp"
#include "gemm/gemm_tensor.hpp"

#include "online_softmax.h"
#include "reg2reg.h"

#include "fmha_nopipe.h"
#include "fmha_pipe_nows.h"
#include "fmha_pipe_ws.h"
#include "ss_helper.h"

// Helper functions for retrieving optimal swizzled layouts
template <typename PrecType, int DIM> constexpr auto getSmemLayoutK() {

	constexpr int headSizeBytes = sizeof(PrecType) * DIM;

	if constexpr (headSizeBytes == 32) {
		return GMMA::Layout_K_SW32_Atom<PrecType>{};
	} else if constexpr (headSizeBytes == 64) {
		return GMMA::Layout_K_SW64_Atom<PrecType>{};
	} else {
		return GMMA::Layout_K_SW128_Atom<PrecType>{};
	}
}

template <typename PrecType, int DIM> constexpr auto getSmemLayoutMN() {

	constexpr int headSizeBytes = sizeof(PrecType) * DIM;

	if constexpr (headSizeBytes == 32) {
		return GMMA::Layout_MN_SW32_Atom<PrecType>{};
	} else if constexpr (headSizeBytes == 64) {
		return GMMA::Layout_MN_SW64_Atom<PrecType>{};
	} else {
		return GMMA::Layout_MN_SW128_Atom<PrecType>{};
	}
}

// Host method that prepares the data structures
// required before calling the DEVICE kernel.
template <typename PrecType, typename Gemm2Type, typename SoftType,
          typename OutputType, int HEADDIM>
void fmhaForwardDevice(int SEQLEN, int KEYLEN, int NUMHEADS, int BATCH,
                       PrecType const *tensorQ, PrecType const *tensorK,
                       Gemm2Type const *tensorV, PrecType *tensorS,
                       OutputType *tensorO, SoftType *miOut,
                       SoftType *sPrimeOut, int iterations, float scale,
                       cudaStream_t stream = 0) {
  using namespace cute;

  // Define shapes (dynamic)
  auto B = int(BATCH);
  auto H = int(NUMHEADS);
  auto M = int(SEQLEN);
  auto N = int(KEYLEN);
  auto K = int(HEADDIM);

  // Define TileShapes
  using bM = Int<kQueriesPerBlock>;
  using bN = Int<kKeysPerBlock>;
  using bK = Int<HEADDIM>;
#if EXECMODE == 2
  using STAGES = Int<1>;
#else
  using STAGES = Int<stageCount>;
#endif

  // Use half_t for computing. float for accumulator.
  using MmaA = PrecType;
  using MmaB = PrecType;
#ifdef GEMM1FP16ACC
  using MmaC = cutlass::half_t;
#else
  using MmaC = SoftType;
#endif

  using Mma2A = Gemm2Type;
  using Mma2B = Gemm2Type;

#ifdef GEMM2FP16ACC
  using Mma2C = cutlass::half_t;
#else
  using Mma2C = SoftType;
#endif

  //
  // All the tensors are stored in BMHK order, with K being the unit-1
  // dimension. For now, n=m.
  //

#if defined(CLUSTERN) && CLUSTERN > 1
#define TMA_LOAD SM90_TMA_LOAD_MULTICAST
  using ClusterShape = Shape<Int<CLUSTERN>, _1, _1>;
#else
#define TMA_LOAD SM90_TMA_LOAD
  using ClusterShape = Shape<_1, _1, _1>;
#endif

  auto ptrQ = reinterpret_cast<PrecType const *>(tensorQ);
  auto ptrK = reinterpret_cast<PrecType const *>(tensorK);
  auto ptrV = reinterpret_cast<Gemm2Type const *>(tensorV);
  auto tileShapeQ = make_shape(bM{}, bK{});
  auto smemLayoutQ =
      tile_to_shape(getSmemLayoutK<MmaA, HEADDIM>(),
                    make_shape(shape<0>(tileShapeQ), shape<1>(tileShapeQ)));
  Layout gmemLayoutQ =
      make_layout(make_shape(M, K, H, B), make_stride(K * H, 1, K, H * M * K));
  Tensor gQ = make_tensor(ptrQ, gmemLayoutQ);
  auto tmaQ =
      make_tma_copy(SM90_TMA_LOAD{}, gQ, smemLayoutQ, tileShapeQ, Int<1>{});

  auto tileShapeK = make_shape(bN{}, bK{});
  // For pipelined FMHA, the third dimension for SMEM K is the number of stages.
  auto smemLayoutK = tile_to_shape(
      getSmemLayoutK<MmaB, HEADDIM>(),
      make_shape(shape<0>(tileShapeK), shape<1>(tileShapeK), STAGES()));
  Layout gmemLayoutK =
      make_layout(make_shape(N, K, H, B), make_stride(K * H, 1, K, H * N * K));
  Tensor gK = make_tensor(ptrK, gmemLayoutK);
  auto tmak = make_tma_copy(TMA_LOAD{}, gK, smemLayoutK(_, _, 0), tileShapeK,
                            size<0>(ClusterShape{}));

  // Use only during debugging, direct writes to GMEM.
  auto tileShapeS = make_shape(bM{}, bN{});
  Layout gmemLayoutS =
      make_layout(make_shape(M, N, H, B), make_stride(N, 1, N * M, H * M * N));
  // Used only for Second matmul with Q and V.
  auto smemLayoutAtomS =
      cute::conditional_return<is_same_v<MmaA, cutlass::half_t>>(
          getSmemLayoutK<MmaA, bN{}>(), GMMA::Layout_K_INTER_Atom<MmaA>{});
  auto smemLayoutS = tile_to_shape(
      smemLayoutAtomS,
      make_shape(shape<0>(tileShapeS), shape<1>(tileShapeS), STAGES()));

// We assume V is NOT transposed in memory by default.
// For now, if we enable V FP8, then we will also transpose V offline.
#ifndef GEMM2FP8
  auto tileShapeV = make_shape(bN{}, bK{});
  auto smemLayoutAtomV = getSmemLayoutK<Mma2B, HEADDIM>();
  // For pipelined FMHA, the third dimension for SMEM V is the number of stages.
  auto smemLayoutV = tile_to_shape(
      smemLayoutAtomV,
      make_shape(shape<0>(tileShapeV), shape<1>(tileShapeV), STAGES()));
  Layout gmemLayoutV =
      make_layout(make_shape(N, K, H, B), make_stride(K * H, 1, K, H * K * N));
  Tensor gV = make_tensor(ptrV, gmemLayoutV);
  auto tmaV = make_tma_copy(TMA_LOAD{}, gV, smemLayoutV(_, _, 0), tileShapeV,
                            size<0>(ClusterShape{}));

  // Layout for Vtranspose. For use in GEMM-II.
  // Note this is the transpose in terms of the view, not in terms of memory.
  auto tileShapeVt = make_shape(bK{}, bN{});
  auto smemLayoutVtFp16 = composition(
      smemLayoutV, make_layout(make_shape(shape<0>(tileShapeVt),
                                          shape<1>(tileShapeVt), STAGES()),
                               make_stride(bN{}, Int<1>{}, bK{} * bN{})));

  auto smemLayoutVtFp8 = tile_to_shape(
      //GMMA::Layout_K_SW64_Atom<Mma2B>{},
      getSmemLayoutK<Mma2B, bN{}>(),
      make_shape(shape<0>(tileShapeVt), shape<1>(tileShapeVt), STAGES()));

  auto smemLayoutVt =
      cute::conditional_return<is_same_v<Mma2B, cutlass::half_t>>(
          smemLayoutVtFp16, smemLayoutVtFp8);

  // The following is only for V FP8 (and may be for TF32 in future).
  auto srcSmemLayoutAtomV = GMMA::Layout_MN_SW64_Atom<Mma2B>{};
  auto srcSmemLayoutV = smemLayoutVtFp16;

  // If V is not assumed transposed in memory, then our choice of majorV changes
  // based on the precision type of V being FP16 or FP8 (-> MN major or K
  // major). This is because transposing the 2nd operand with WGMMA is currently
  // supported for FP16 only. Thus for V FP8, we would need to do the transpose
  // ourselves.
  constexpr auto majorV =
      cute::conditional_return<is_same_v<Mma2B, cutlass::half_t>>(
          GMMA::Major::MN, GMMA::Major::K);
#else
  auto tileShapeV = make_shape(bK{}, bN{});
  auto smemLayoutAtomV = getSmemLayoutK<Mma2B, bN{}>();
  //auto smemLayoutAtomV = GMMA::Layout_K_INTER_Atom<Mma2B>{};
  auto smemLayoutV = tile_to_shape(
      smemLayoutAtomV,
      make_shape(shape<0>(tileShapeV), shape<1>(tileShapeV), STAGES()));
  Layout gmemLayoutV =
      make_layout(make_shape(K, N, H, B), make_stride(N * H, 1, N, H * K * N));
  Tensor gV = make_tensor(ptrV, gmemLayoutV);
  auto tmaV = make_tma_copy(TMA_LOAD{}, gV, smemLayoutV(_, _, 0), tileShapeV,
                            size<0>(ClusterShape{}));

  constexpr auto majorV = GMMA::Major::K;
  // Don't care for this version.
  auto smemLayoutVt = smemLayoutV;
  auto srcSmemLayoutAtomV = smemLayoutAtomV;
  auto srcSmemLayoutV = smemLayoutV;
#endif

  auto tileShapeO = make_shape(bM{}, bK{});
  Layout gmemLayoutO =
      make_layout(make_shape(M, K, H, B), make_stride(K * H, 1, K, H * M * K));
  auto smemLayoutO =
      tile_to_shape(getSmemLayoutK<OutputType, bK{}>(),
                    make_shape(shape<0>(tileShapeQ), shape<1>(tileShapeQ)));
  Tensor gO = make_tensor(tensorO, gmemLayoutO);
  auto tmaO =
      make_tma_copy(SM90_TMA_STORE{}, gO, smemLayoutO, tileShapeO, Int<1>{});

// Enable this flag for 256 threads (or 8 warps) per CTA.
#ifdef CTA256
  using MmaTileShape = Layout<Shape<_2, _1, _1>>;
#else
  using MmaTileShape = Layout<Shape<_1, _1, _1>>;
#endif

#if defined(QINRMEM) && EXECMODE != 2  
  // USE RS version of GMMA for GEMM-I.
  // Disabled for non-pipelined version.  
  using TiledMma0 = decltype(cute::make_tiled_mma(
      rs_op_selector_custom<MmaA, MmaB, MmaC, Shape<bM, bN, bK>>(),
      MmaTileShape{}));
#else
  // USE SS version of GMMA for GEMM-I.
  using TiledMma0 = decltype(cute::make_tiled_mma(
      ss_op_selector_custom<MmaA, MmaB, MmaC, Shape<bM, bN, bK>>(),
      MmaTileShape{}));
#endif

#ifdef SINSMEM
  // USE SS version of GMMA for GEMM-II.
  using TiledMma1 = decltype(cute::make_tiled_mma(
      ss_op_selector_custom<Mma2A, Mma2B, Mma2C, Shape<bM, bK, bN>,
                            GMMA::Major::K, majorV>(),
      MmaTileShape{}));
#else
  // USE RS version of GMMA for GEMM-II (Default).
  using TiledMma1 = decltype(cute::make_tiled_mma(
      rs_op_selector_custom<Mma2A, Mma2B, Mma2C, Shape<bM, bK, bN>,
                            GMMA::Major::K, majorV>(),
      MmaTileShape{}));
#endif

  // col-major for MI and S_prime (used only for verification).
  Layout gmemLayoutMi = make_layout(make_shape(M, H, B), GenColMajor{});

  // We separate out the warp-specialized and non-pipelined 
  // versions using a compiler flag.
  // Default is pipelined and not warp-specialized.
#if !defined(EXECMODE) || EXECMODE == 0
  // Get the ptr to kernel function.
  void const *kernel = (void const *)fmhaForwardPipelinedNoWspl<
      PrecType, MmaC, SoftType, Mma2A, OutputType, TiledMma0, TiledMma1,
      decltype(tmaQ), decltype(tileShapeQ), decltype(gmemLayoutQ),
      decltype(smemLayoutQ), decltype(tmak), decltype(tileShapeK),
      decltype(gmemLayoutK), decltype(smemLayoutK), decltype(tileShapeS),
      decltype(gmemLayoutS), decltype(smemLayoutS), decltype(tmaV),
      decltype(tileShapeV), decltype(gmemLayoutV), decltype(smemLayoutV),
      decltype(smemLayoutVt), decltype(tmaO), decltype(tileShapeO),
      decltype(gmemLayoutO), decltype(smemLayoutO), decltype(gmemLayoutMi),
      ClusterShape>;

  auto ctaSize = size(TiledMma0{});

#elif EXECMODE == 1
  // Get the ptr to kernel function.
  void const *kernel = (void const *)fmhaForwardPipelinedWspl<
      PrecType, MmaC, SoftType, Mma2A, OutputType, TiledMma0, TiledMma1,
      decltype(tmaQ), decltype(tileShapeQ), decltype(gmemLayoutQ),
      decltype(smemLayoutQ), decltype(tmak), decltype(tileShapeK),
      decltype(gmemLayoutK), decltype(smemLayoutK), decltype(tileShapeS),
      decltype(gmemLayoutS), decltype(smemLayoutS), decltype(tmaV),
      decltype(tileShapeV), decltype(gmemLayoutV), decltype(smemLayoutV),
      decltype(smemLayoutVt), decltype(tmaO), decltype(tileShapeO),
      decltype(gmemLayoutO), decltype(smemLayoutO), decltype(gmemLayoutMi),
      ClusterShape>;

  auto ctaSize = size(TiledMma0{}) + NumCopyThreads;

#elif EXECMODE == 2
  // Get the ptr to kernel function.
  void const *kernel = (void const *)fmhaForwardNoPipeline<
      PrecType, MmaC, SoftType, Mma2A, OutputType, TiledMma0, TiledMma1,
      decltype(tmaQ), decltype(tileShapeQ), decltype(gmemLayoutQ),
      decltype(smemLayoutQ), decltype(tmak), decltype(tileShapeK),
      decltype(gmemLayoutK), decltype(smemLayoutK), decltype(tileShapeS),
      decltype(gmemLayoutS), decltype(smemLayoutS), decltype(tmaV),
      decltype(tileShapeV), decltype(gmemLayoutV), decltype(smemLayoutV),
      decltype(smemLayoutVt), decltype(tmaO), decltype(tileShapeO),
      decltype(gmemLayoutO), decltype(smemLayoutO), decltype(gmemLayoutMi),
      ClusterShape>;

  auto ctaSize = size(TiledMma0{});
#endif

  //
  // Define CUDA launch kernel parameters.
  //

  // Compute and set dynamic shared memory size.
  auto smem_size =
      int(sizeof(SharedStorage<MmaA, Mma2A, OutputType, decltype(smemLayoutQ),
                               decltype(smemLayoutK), decltype(smemLayoutS),
                               decltype(smemLayoutV), decltype(smemLayoutO),
                               ClusterShape>));
  cfk::utils::set_smem_size(smem_size, kernel);

  // Set the THREAD BLOCK (CTA) dimensions.
  // #threads in CTA = #threads in MMA (128 by default) + 128 (for WS).
  // For example, this is 3*128 (= 3 warpgroups) for CTA256 and WS.
  dim3 block_dims(ctaSize);

  // Set the GRID dimensions (3-D).
  // First dimension = # of blocks of Q.
  // Second dimension = # of heads.
  // Third dimension = # of batches.
  dim3 grid_dims(ceil_div(size(M), size(bM{})), H, B);

  // Set the CLUSTER dimensions.
  dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}),
                    size<2>(ClusterShape{}));

  // Define the cluster launch parameter structure.
  cutlass::ClusterLaunchParams params{grid_dims, block_dims, cluster_dims,
                                      smem_size, stream};

  // Compute the no. of tiles of K matrix.
  auto nTilesOfK = ceil_div(size(N), size(bN{}));

  // Run the CUDA kernel for preferred number of iterations.
  for (int i = 0; i < iterations; ++i) {
    cutlass::Status status = cutlass::launch_kernel_on_cluster(
        params, kernel, ptrQ, tmaQ, tileShapeQ, gmemLayoutQ, smemLayoutQ, ptrK,
        tmak, tileShapeK, gmemLayoutK, smemLayoutK, tensorS, tileShapeS,
        gmemLayoutS, smemLayoutS, nTilesOfK, tensorV, tmaV, tileShapeV,
        gmemLayoutV, smemLayoutV, smemLayoutVt, tensorO, tmaO, tileShapeO,
        gmemLayoutO, smemLayoutO, miOut, sPrimeOut, gmemLayoutMi, scale);     
  }
}

// Wrapper function for multiple streams.
// Currently, only single stream is used by default.
template <typename PrecType, typename Gemm2Type, typename SoftType,
          typename OutputType, int HEADDIM>
void fmhaForwardDeviceLoop(int SEQLEN, int KEYLEN, int NUMHEADS, int BATCHSIZE,
                           PrecType const *Q, PrecType const *K, Gemm2Type *V,
                           PrecType *S, OutputType *D, SoftType *miOut,
                           SoftType *sPrimeOut, int iterations, int nStreams,
                           float scale) {

  if (nStreams == 1) {
    fmhaForwardDevice<PrecType, Gemm2Type, SoftType, OutputType, HEADDIM>(
        SEQLEN, KEYLEN, NUMHEADS, BATCHSIZE, Q, K, V, S, D, miOut, sPrimeOut,
        iterations, scale);
    return;
  } else {
    auto L = BATCHSIZE / nStreams;
    for (int i = 0; i < nStreams; ++i) {
      cudaStream_t stream;
      cudaStreamCreate(&stream);

      auto offsetQ = i * SEQLEN * NUMHEADS * HEADDIM * L;
      auto offsetK = i * KEYLEN * NUMHEADS * HEADDIM * L;
      auto offsetS = i * SEQLEN * NUMHEADS * KEYLEN * L;
      auto offsetV = i * KEYLEN * NUMHEADS * HEADDIM * L;
      auto offsetD = i * SEQLEN * NUMHEADS * HEADDIM * L;
      auto miOffset = i * SEQLEN * NUMHEADS * L;

      fmhaForwardDevice<PrecType, Gemm2Type, SoftType, OutputType, HEADDIM>(
          SEQLEN, KEYLEN, NUMHEADS, L, Q + offsetQ, K + offsetK, V + offsetV,
          S + offsetS, D + offsetD, miOut + miOffset, sPrimeOut + miOffset,
          iterations, scale, stream);
    }
  }
}

#include <cassert>
#include <cstdio>
#include <cstdlib>

template <typename T> struct TypeConvert {
  __host__ __device__ T operator()(float a) { return T(a); }
};

template <> struct TypeConvert<cutlass::float_e4m3_t> {
  __host__ __device__ cutlass::float_e4m3_t operator()(float a) {
    return cutlass::float_e4m3_t::from_float(a);
  }
};

template <> struct TypeConvert<cutlass::half_t> {
  __host__ __device__ cutlass::half_t operator()(float a) {
    return cutlass::half_t::convert(a);
  }
};

//  The main driver function.
template <typename PrecType, int HEADDIM>
void testFmhaForward(int m, int n, int numHeads, int batchSize, int iterations,
                     bool refCheck, bool printValues, bool printDiffs,
                     int nStreams) {
#ifdef GEMM2FP8  
  using Gemm2Type = PrecType;
#else
  using Gemm2Type = cutlass::half_t;
#endif

  constexpr float kLog2e = float(1.4426950408889634074); // log_2(e) = M_LOG2E
  const float softmax_scale = (1.0f / sqrt(float(HEADDIM)));
  const float scale = softmax_scale * kLog2e;
  cudaDeviceReset();
  cute::device_init(0);
  int k = HEADDIM;
  std::cout << "M = " << m << std::endl;
  std::cout << "N = " << n << std::endl;
  std::cout << "K = " << k << std::endl;
  std::cout << "QBLK = " << kQueriesPerBlock << std::endl;
  std::cout << "KBLK = " << kKeysPerBlock << std::endl;

  auto mLong = uint64_t(m);
  auto nLong = uint64_t(n);
  auto kLong = uint64_t(k);
  auto lLong = uint64_t(numHeads * batchSize);
  std::cout << "L = " << lLong << " : " << numHeads << " * " << batchSize
            << std::endl;

  using OutputType = cutlass::half_t;
  thrust::device_vector<PrecType> devQ(mLong * kLong * lLong);
  thrust::device_vector<PrecType> devK(nLong * kLong * lLong);
  thrust::device_vector<PrecType> devS(mLong * nLong * lLong);
  thrust::device_vector<Gemm2Type> devV(nLong * kLong * lLong);
  thrust::device_vector<Gemm2Type> devVt(nLong * kLong * lLong);
  thrust::device_vector<OutputType> devD(mLong * kLong * lLong);

  uint32_t seed = 3080;

  cutlass::Distribution::Kind initQ;
  cutlass::Distribution::Kind initK;
  cutlass::Distribution::Kind initV;
  if (refCheck) {
    // Change to Uniform for easier debugging
    // initQ = cutlass::Distribution::Uniform;
    // initK = cutlass::Distribution::Uniform;
    // initV = cutlass::Distribution::Uniform;
    initQ = cutlass::Distribution::Gaussian;
    initK = cutlass::Distribution::Gaussian;
    initV = cutlass::Distribution::Gaussian;
  } else {
    initQ = cutlass::Distribution::Gaussian;
    initK = cutlass::Distribution::Gaussian;
    initV = cutlass::Distribution::Gaussian;
  }

  cfk::initialize_rand(devQ.data().get(), devQ.size(), initQ, seed + 1);
  cfk::initialize_rand(devK.data().get(), devK.size(), initK, seed + 2);
  cfk::initialize_rand(devV.data().get(), devV.size(), initV, seed + 3);
  cfk::initialize_const(devS.data().get(), devS.size(), PrecType(-1));
  cfk::initialize_const(devD.data().get(), devD.size(), OutputType(-1));

  using SoftType = float; // SoftType is always float.

  thrust::host_vector<PrecType> hostQ = devQ;
  thrust::host_vector<PrecType> hostK = devK;
  thrust::host_vector<PrecType> hostS = devS;
  thrust::host_vector<Gemm2Type> hostV = devV;
  thrust::host_vector<OutputType> hostD = devD;

// For our experiments with V transposed in memory,
// we view the cost of the transpose as offline.
#ifdef GEMM2FP8
  thrust::host_vector<Gemm2Type> hostVt = hostV;

  int batchStride = m * numHeads * HEADDIM;
  int seqStride = numHeads * HEADDIM;
  int headStride = HEADDIM;
  int headStrideT = m;
  int kStrideT = numHeads * m;

  for (int B = 0; B < batchSize; ++B) {
    for (int M = 0; M < m; ++M) {
      for (int H = 0; H < numHeads; ++H) {
        for (int K = 0; K < HEADDIM; ++K) {
          hostVt[B * batchStride + K * kStrideT + H * headStrideT + M] =
              hostV[B * batchStride + M * seqStride + H * headStride + K];
        }
      }
    }
  }

#ifdef FP8NOSHFL
  thrust::host_vector<Gemm2Type> hostVp = hostVt;

  for (int B = 0; B < batchSize; ++B) {
    for (int H = 0; H < numHeads; ++H) {
      for (int K = 0; K < HEADDIM; ++K) {
        for (int M = 0; M < m; M += 16) {
          for (int r = 0; r < 4; ++r) {
            for (int q = 0; q < 2; ++q) {
              for (int p = 0; p < 2; ++p) {
                // std::cout << p + q * 2 + r * 4 << " " << p + r * 2 + q * 8
                // << std::endl;
                hostVp[B * batchStride + K * kStrideT + H * headStrideT + M +
                       p + r * 2 + q * 8] =
                    hostVt[B * batchStride + K * kStrideT + H * headStrideT +
                           M + p + q * 2 + r * 4];
              }
            }
          }
        }
      }
    }
  }

  devVt = hostVp;
#else
  devVt = hostVt;
#endif
#else
  devVt = devV;
#endif

  thrust::device_vector<SoftType> devMiOut(mLong * lLong);
  thrust::device_vector<SoftType> devSprimeOut(mLong * lLong);

  const int timing_iterations = iterations;
  GPU_Clock timer;

  double fmha_flops =
      double(4 * batchSize * numHeads * mLong * nLong * kLong) / double(1.0e9);

  // Run few times (warmup).
  devS = hostS;
  devD = hostD;
  fmhaForwardDeviceLoop<PrecType, Gemm2Type, SoftType, OutputType, HEADDIM>(
      m, n, numHeads, batchSize, devQ.data().get(), devK.data().get(),
      devVt.data().get(), devS.data().get(), devD.data().get(),
      devMiOut.data().get(), devSprimeOut.data().get(), 10, nStreams, scale);
  CUTE_CHECK_LAST();

  // Timing iterations
  devS = hostS;
  devD = hostD;
  timer.start();
  fmhaForwardDeviceLoop<PrecType, Gemm2Type, SoftType, OutputType, HEADDIM>(
      m, n, numHeads, batchSize, devQ.data().get(), devK.data().get(),
      devVt.data().get(), devS.data().get(), devD.data().get(),
      devMiOut.data().get(), devSprimeOut.data().get(), iterations, nStreams,
      scale);
  double cute_time = timer.seconds() / (float)timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUTE_FMHA:     [%6.1f]Gflop/s  "
         "(%6.4f)ms\n",
         fmha_flops / cute_time, cute_time * 1000);

  thrust::host_vector<SoftType> miHostOut = devMiOut;
  thrust::host_vector<SoftType> sPrimeHostOut = devSprimeOut;
  thrust::host_vector<SoftType> miRefHostOut(miHostOut.size());
  thrust::host_vector<SoftType> sPrimeRefHostOut(sPrimeHostOut.size());

  bool usePreScaling = true;
  bool usePow2 = false;

  using TestPrecType = float;

  thrust::host_vector<TestPrecType> cute_result_S = devS;
  thrust::host_vector<TestPrecType> cute_result_D = devD;
  if (refCheck) {
    // up-cast to float always.
    thrust::device_vector<float> devQFloat(mLong * kLong * lLong);
    thrust::device_vector<float> devKFloat(nLong * kLong * lLong);
    thrust::device_vector<float> devVFloat(nLong * kLong * lLong);
    thrust::transform(devQ.begin(), devQ.end(), devQFloat.begin(),
                      TypeConvert<float>());
    thrust::transform(devK.begin(), devK.end(), devKFloat.begin(),
                      TypeConvert<float>());
    thrust::transform(devV.begin(), devV.end(), devVFloat.begin(),
                      TypeConvert<float>());
    TestAttention<TestPrecType, SoftType> testBed(numHeads, batchSize, k, m);
    testBed.initialize();
    thrust::device_vector<TestPrecType> devSFloat(mLong * nLong * lLong);
    thrust::device_vector<TestPrecType> devDFloat(mLong * kLong * lLong);
    devDFloat = hostD;
    devSFloat = hostS;
    testBed.compute(devQFloat.data().get(), devKFloat.data().get(),
                    devVFloat.data().get(), devSFloat.data().get(),
                    devDFloat.data().get(), miRefHostOut.data(),
                    sPrimeRefHostOut.data(), usePow2, usePreScaling);
    thrust::host_vector<TestPrecType> cublas_result_S = devSFloat;
    thrust::host_vector<TestPrecType> cublas_result_D = devDFloat;

    // For lower precision formats, we want to distinguish between validation
    // errors due to bugs vs. expected errors from low precision computation.
    // In practice, percentage of errors should be far lower than this.
    auto errCountExpected =
        typeid(PrecType) == typeid(cutlass::float_e4m3_t) ? 0.001 : 0.0001;
#ifdef COPYOUTMM0
    // Our intermediate write to S is unscaled. So scale it before checking
    // with CUTLASS.
    if (usePreScaling) {
      for (int j = 0; j < cute_result_S.size(); ++j) {
        cute_result_S[j] =
            PrecType(cutlass::half_t(cute_result_S[j]) * softmax_scale);
      }
    }
    bool gemm1 = cfk::verify_tensor(cute_result_S, cublas_result_S, printValues,
                                    printDiffs);
    std::string result1 = gemm1 ? "Passed" : "Failed";
    std::cout << "gemm-check-1: " << result1 << std::endl;
#endif

#ifdef COPYOUTMI
    // Our intermediate write to MI is scaled with log2e. So un-scale it
    // before checking with CUTLASS.
    if (!usePow2) {
      for (int j = 0; j < miHostOut.size(); ++j) {
        miHostOut[j] = miHostOut[j] * (1.0 / kLog2e);
      }
    }
    bool maxCheck =
        cfk::verify_tensor(miHostOut, miRefHostOut, printValues, printDiffs);
    std::string maxCheckResult = maxCheck ? "Passed" : "Failed";
    std::cout << "max-check: " << maxCheckResult << std::endl;

    // Our intermediate write to sPrime is not reciprocal. So invert it before
    // checking with CUTLASS.
    for (int j = 0; j < sPrimeHostOut.size(); ++j) {
      sPrimeHostOut[j] = (1.0 / sPrimeHostOut[j]);
    }
    bool sumCheck = cfk::verify_tensor(sPrimeHostOut, sPrimeRefHostOut,
                                       printValues, printDiffs);
    std::string sumCheckResult = sumCheck ? "Passed" : "Failed";
    std::cout << "sum-check: " << sumCheckResult << std::endl;
#endif

    bool gemm2 = cfk::verify_tensor(cute_result_D, cublas_result_D, printValues,
                                    printDiffs, errCountExpected);
    std::string result2 = gemm2 ? "Passed" : "Failed";
    std::cout << "gemm-check-2: " << result2 << std::endl;
  }
}

/// Prints the usage statement.
void print_usage() {

  std::cout
      << "fmha_forward "
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage "
         "statement.\n\n"
      << "  --prec-type=<int>           1 for FP16, 2 (default) for FP8 E4M3.\n"         
      << "  --batch-size=<int>          Batch size in multi-head attention "
         "(default=4).\n"
      << "  --dim-size=<int>            Full size of the head dimension "
         "before reshape (default=2048).\n"
      << "  --head-size=<int>           Individual head size "
         "(default=64).\n"
      << "  --seq-length=<int>          Sequence length in multi-head "
         "attention for Q (default=4096).\n"
      << "  --iterations=<int>          Number of profiling iterations to "
         "perform (default=20).\n"
      << "  --num-cuda-streams=<int>    Number of CUDA streams to use "
         "(default=1).\n"
      << "  --reference-check=<bool>    If true, performs reference check.\n"
      << "  --print-values=<bool>       If true, prints the values of the "
         "result (also the intermediate results of gemm-I & softmax).\n";
}

int main(int argc, char const **argv) {

  cutlass::CommandLine cmd(argc, argv);
  // Parses the command line
  if (cmd.check_cmd_line_flag("help")) {
    print_usage();
    return;
  }

  int seqLength, batchSize, dimSize, iterations, nStreams, kHeadSize, precType;  

  bool refCheck, printValues, printDiffs;
  cmd.get_cmd_line_argument("batch-size", batchSize, 4);
  cmd.get_cmd_line_argument("dim-size", dimSize, 2048);
  cmd.get_cmd_line_argument("head-size", kHeadSize, 64);
  cmd.get_cmd_line_argument("seq-length", seqLength, 4096);
  cmd.get_cmd_line_argument("iterations", iterations, 20);
  cmd.get_cmd_line_argument("num-cuda-streams", nStreams, 1);
  cmd.get_cmd_line_argument("reference-check", refCheck, false);
  cmd.get_cmd_line_argument("print-values", printValues, false);
  cmd.get_cmd_line_argument("print-diffs", printDiffs, false);
  cmd.get_cmd_line_argument("prec-type", precType, 2);  

  if (nStreams > batchSize) {
    std::cout << "#max no. of cuda streams <= batchSize" << std::endl;
    exit(-1);
  }
  int numHeads = dimSize / kHeadSize;

  // Instantiate the function template for different HEADDIMS.
  // For now, only half_t and e4m3_t are supported.
  // Though it's simple to also add e5m2_t.
  if (precType == 1) {
    #ifndef GEMM2FP8
    if (kHeadSize == 64) {
      testFmhaForward<cutlass::half_t, 64>(seqLength, seqLength, numHeads,
                                           batchSize, iterations, refCheck,
                                           printValues, printDiffs, nStreams);
    } else if (kHeadSize == 128) {
      testFmhaForward<cutlass::half_t, 128>(seqLength, seqLength, numHeads,
                                            batchSize, iterations, refCheck,
                                            printValues, printDiffs, nStreams);
    } else if (kHeadSize == 256) {
      testFmhaForward<cutlass::half_t, 256>(seqLength, seqLength, numHeads,
                                            batchSize, iterations, refCheck,
                                            printValues, printDiffs, nStreams);
    } else {
      std::cout << "Unsupported head dim: " << kHeadSize << std::endl;
      exit(-1);
    }
    #else
      std::cout << "FP16 not supported when GEMM2FP8 is enabled." << std::endl;
      exit(-1);
    #endif
  }
  // For FP8, we choose e4m3 according to the following philosophy:
  // e4m3 for inference (forward pass), e5m2 for training (backward pass).
  else if (precType == 2) {
#if 1
    if (kHeadSize == 64) {
      testFmhaForward<cutlass::float_e4m3_t, 64>(
          seqLength, seqLength, numHeads, batchSize, iterations, refCheck,
          printValues, printDiffs, nStreams);
    } else if (kHeadSize == 128) {
      testFmhaForward<cutlass::float_e4m3_t, 128>(
          seqLength, seqLength, numHeads, batchSize, iterations, refCheck,
          printValues, printDiffs, nStreams);
    } else if (kHeadSize == 256) {
      testFmhaForward<cutlass::float_e4m3_t, 256>(
          seqLength, seqLength, numHeads, batchSize, iterations, refCheck,
          printValues, printDiffs, nStreams);
    } else {
      std::cout << "Unsupported head dim: " << kHeadSize << std::endl;
      exit(-1);
    }
#endif
  } else {
    std::cout << "Unsupported type: " << precType << std::endl;
    exit(-1);
  }

  return 0;
}
