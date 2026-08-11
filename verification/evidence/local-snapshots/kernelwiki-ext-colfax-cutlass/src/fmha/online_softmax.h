#pragma once

#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/vector.h"
#include "cutlass/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include <cute/tensor.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T> struct MaxOp {
  __device__ inline T operator()(T const &x, T const &y) {
    return x > y ? x : y;
  }
};

template <> struct MaxOp<float> {
  // This is slightly faster
  __device__ inline float operator()(float const &x, float const &y) {
    return max(x, y);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T> struct SumOp {
  __device__ inline T operator()(T const &x, T const &y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int THREADS> struct ShflReduce {
  static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
  template <typename T, typename Operator>
  static __device__ inline T run(T x, Operator &op) {
    constexpr int OFFSET = THREADS / 2;
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
    return ShflReduce<OFFSET>::run(x, op);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <> struct ShflReduce<2> {
  template <typename T, typename Operator>
  static __device__ inline T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
  }
};

template <typename AccumType, typename Fragment0, typename Fragment1>
CUTLASS_DEVICE static void applySoftmaxNormalizer(const Fragment0 &sPrime,
                                                  Fragment1 &accum) {

  using FragValType = typename Fragment1::value_type;
  auto data = accum.data();
  int n = 0;
  int rowId = 0;
#pragma unroll
  for (int i = 0; i < size(shape<1>(accum)); ++i) {
    auto sPrime0 = AccumType(1.0 / sPrime(rowId));
    auto sPrime1 = AccumType(1.0 / sPrime(rowId + 1));

#pragma unroll
    for (int k = 0; k < size(shape<2>(accum)) * size<2>(shape<0>(accum)); ++k) {
      data[n] = FragValType(AccumType(data[n]) * sPrime0);
      n++;
      data[n] = FragValType(AccumType(data[n]) * sPrime0);
      n++;
      data[n] = FragValType(AccumType(data[n]) * sPrime1);
      n++;
      data[n] = FragValType(AccumType(data[n]) * sPrime1);
      n++;
    }
    rowId += 2;
  }
}

template <bool isFirst, typename AccumType, typename Fragment0,
          typename Fragment1, typename Fragment2, typename Fragment3>
CUTLASS_DEVICE static void
onlineSoftmaxAndRescale(Fragment0 &mi, Fragment1 &sPrime, Fragment2 &accum,
                        Fragment3 &accum_o, float scaleFactor) {
  using namespace cute;
  using FragValType = typename Fragment2::value_type;
  using FragValTypeO = typename Fragment3::value_type;
  auto laneId = cutlass::canonical_lane_idx();

  // cute::axpby(scaleFactor, accum, Int<0>{}, accum);

  // First update `mi` to the max per-row
  //
  auto VT = shape<0>(accum); // number of vector elements per tile.
  auto MT = shape<1>(accum); // number of tiles along M.
  auto NT = shape<2>(accum); // number of tiles along N.
  MaxOp<AccumType> maxOp;

  auto data = accum.data();
  auto data_o = accum_o.data();
  int n = 0;
  int no = 0;
  Tensor miPrev = make_fragment_like(mi);
  cute::copy(mi, miPrev);

  int rowId = 0;
#pragma unroll
  for (int i = 0; i < MT; ++i) {
    auto max0 = mi(rowId);
    auto max1 = mi(rowId + 1);

    // Traverse 2-rows + 2-cols (2x2) simultaneously.

#pragma unroll
    for (int k = 0; k < NT * size<2>(VT); ++k) {
      data[n] = FragValType(AccumType(data[n]) * scaleFactor);
      max0 = cutlass::fast_max(max0, AccumType(data[n]));
      n++;

      data[n] = FragValType(AccumType(data[n]) * scaleFactor);
      max0 = cutlass::fast_max(max0, AccumType(data[n]));
      n++;

      data[n] = FragValType(AccumType(data[n]) * scaleFactor);
      max1 = cutlass::fast_max(max1, AccumType(data[n]));
      n++;

      data[n] = FragValType(AccumType(data[n]) * scaleFactor);
      max1 = cutlass::fast_max(max1, AccumType(data[n]));
      n++;
    }
    auto max_quad_0 = ShflReduce<4>::run(max0, maxOp);
    auto max_quad_1 = ShflReduce<4>::run(max1, maxOp);
    mi(rowId) = max_quad_0;
    mi(rowId + 1) = max_quad_1;

    if (!isFirst) {
      auto m_prime_exp0 = exp2f(miPrev(rowId) - max_quad_0);
      sPrime(rowId) *= m_prime_exp0;

      auto m_prime_exp1 = exp2f(miPrev(rowId + 1) - max_quad_1);
      sPrime(rowId + 1) *= m_prime_exp1;

      for (int k = 0; k < size(shape<2>(accum_o)) * size<2>(shape<0>(accum_o));
           ++k) {
        data_o[no] = FragValTypeO(AccumType(data_o[no]) * m_prime_exp0);
        no++;

        data_o[no] = FragValTypeO(AccumType(data_o[no]) * m_prime_exp0);
        no++;

        data_o[no] = FragValTypeO(AccumType(data_o[no]) * m_prime_exp1);
        no++;

        data_o[no] = FragValTypeO(AccumType(data_o[no]) * m_prime_exp1);
        no++;
      }
    }
    rowId += 2;
  }

  SumOp<AccumType> sumOp;
  rowId = 0;
  n = 0;
#pragma unroll
  for (int i = 0; i < MT; ++i) {
    AccumType sum0 = 0.0f;
    AccumType sum1 = 0.0f;
    auto miRow0 = mi(rowId);
    auto miRow1 = mi(rowId + 1);
#pragma unroll
    for (int k = 0; k < NT * size<2>(VT); ++k) {

      auto val0 = AccumType(data[n]);
      val0 = exp2f(val0 - miRow0);
      sum0 += val0;
      data[n++] = val0;

      auto val1 = AccumType(data[n]);
      val1 = exp2f(val1 - miRow0);
      sum0 += val1;
      data[n++] = val1;

      auto val2 = AccumType(data[n]);
      val2 = exp2f(val2 - miRow1);
      sum1 += val2;
      data[n++] = val2;

      auto val3 = AccumType(data[n]);
      val3 = exp2f(val3 - miRow1);
      sum1 += val3;
      data[n++] = val3;
    }
    auto sumQuad0 = ShflReduce<4>::run(sum0, sumOp);
    auto sumQuad1 = ShflReduce<4>::run(sum1, sumOp);
    sPrime(rowId) += sumQuad0;
    sPrime(rowId + 1) += sumQuad1;
    rowId += 2;
  }
}
