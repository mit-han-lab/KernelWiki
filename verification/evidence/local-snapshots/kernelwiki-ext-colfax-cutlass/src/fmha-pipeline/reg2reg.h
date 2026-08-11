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

// Conversion Utility to convert RMEM from one type to another.
// Used for conversion from AccumType to PrecType.
template <typename To_type, typename From_type, typename Fragment>
inline __device__ auto convert_type(Fragment const &tensor) {
  constexpr int numel = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
  // Note: this requires tensor to be "contiguous."
  auto frag =
      convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(
          tensor.data()));
  return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

// Reshape Utility for converting the layout from accumulator of GEMM-I
// to Operand A of GEMM-II.
struct ReshapeTStoTP {
  template <class FragmentC, class FragmentQ>
  __device__ auto operator()(FragmentC &&tC, FragmentQ &&tQ) {

    // get the layout of one row of Q.
    auto layoutQRow = make_ordered_layout(tQ(_, 0, _).layout());
    // get the layout of  M dimension of C.
    auto layoutCM = get<1>(tC.layout());
    return make_layout(get<0>(layoutQRow), layoutCM, get<1>(layoutQRow));
  }
};

// Need this register byte permute/shuffle to match register layout of
// (FP8 downcasted) accumulator of GEMM-I to FP8 operand A of GEMM-II.
struct ReorgCFp8toAFp8{
  int selectorEx0;
  int selectorEx1;  
  int selectorEx4;
  int selectorEx5;
  int upper_map[4] = {0,3,1,2};
  int lower_map[4] = {1,2,0,3};
  
  
CUTLASS_DEVICE ReorgCFp8toAFp8() {
  int laneId = cutlass::canonical_lane_idx();
  
   if (laneId % 4 == 0 || laneId % 4 == 3) {
     selectorEx0 = 0x3210;
     selectorEx1 = 0x7654;
     selectorEx4 = 0x5410;
	   selectorEx5 = 0x7632;
   } else {
     selectorEx0 = 0x7654;
     selectorEx1 = 0x3210;
     selectorEx4 = 0x1054;
	   selectorEx5 = 0x3276;
   }  
   
}

template <typename Fragment>
CUTLASS_DEVICE auto operator()(Fragment &accum) {

  using namespace cute;  

  // First update `mi` to the max per-row
  //
  auto VT = shape<0>(accum); // number of vector elements per tile.
  auto MT = shape<1>(accum); // number of tiles along M.
  auto NT = shape<2>(accum); // number of tiles along N.

  auto data = accum.data();
  int n = 0;

#pragma unroll
  for (int i = 0; i < MT; ++i) {

    // Traverse 2-rows + 2-cols (2x2) simultaneously.

#pragma unroll
    for (int k = 0; k < NT * size<2>(VT) / 2; ++k) {

      auto upper = *reinterpret_cast<uint32_t*>(&data[n]);
      auto lower = *reinterpret_cast<uint32_t*>(&data[n+4]);
      
      auto upper0 = __byte_perm(upper, lower, selectorEx0);
      auto lower0 = __byte_perm(upper, lower, selectorEx1);      
      upper0 = __shfl_sync(uint32_t(-1),upper0, upper_map[threadIdx.x%4],4);
      lower0 = __shfl_sync(uint32_t(-1),lower0, lower_map[threadIdx.x%4],4);
  
      uint32_t *data_32bit = reinterpret_cast<uint32_t *>(&data[n]);
      data_32bit[0] = __byte_perm(upper0, lower0, selectorEx4);
      data_32bit[1] = __byte_perm(upper0, lower0, selectorEx5);
      n += 8;
    }
  }
}
};

// Alternative version: tested to be slightly slower
// struct ReorgCFp8toAFp8{
//   int selectorEx0;
//   int selectorEx1;
//   int selectorEx2;
//   int selectorEx3;
//   int selectorEx4;
//   int selectorEx5;
  
  
// CUTLASS_DEVICE ReorgCFp8toAFp8() {
//   auto laneId = cutlass::canonical_lane_idx();
//    if (laneId % 4 == 0 || laneId % 4 == 1) {
//      selectorEx0 = 0x3210;
//      selectorEx1 = 0x7654;
//    } else {
//      selectorEx0 = 0x7654;
//      selectorEx1 = 0x3210;
//    }
//    if (laneId % 4  == 0 || laneId % 4 == 3) {
// 	   selectorEx2 = 0x3210;
// 	   selectorEx3 = 0x7654;
//    } else {
// 	   selectorEx2 = 0x7654;
// 	   selectorEx3 = 0x3210;
//    }
   
//    if (laneId % 2  == 0) {
// 	   selectorEx4 = 0x5410;
// 	   selectorEx5 = 0x7632;
//    } else { // 1 & 3.
// 	   selectorEx4 = 0x1054;
// 	   selectorEx5 = 0x3276;
//    }

// }

// template <typename Fragment>
// CUTLASS_DEVICE auto operator()(Fragment &accum) {

//   using namespace cute;

//   // First update `mi` to the max per-row
//   //
//   auto VT = shape<0>(accum); // number of vector elements per tile.
//   auto MT = shape<1>(accum); // number of tiles along M.
//   auto NT = shape<2>(accum); // number of tiles along N.

//   auto data = accum.data();
//   int n = 0;

// #pragma unroll
//   for (int i = 0; i < MT; ++i) {

//     // Traverse 2-rows + 2-cols (2x2) simultaneously.

// #pragma unroll
//     for (int k = 0; k < NT * size<2>(VT) / 2; ++k) {

//       auto upper = *reinterpret_cast<uint32_t*>(&data[n]);
//       auto lower = *reinterpret_cast<uint32_t*>(&data[n+4]);
      
//       auto upper0 = __byte_perm(upper, lower, selectorEx0);
//       auto lower0 = __byte_perm(upper, lower, selectorEx1);
//       lower0 = __shfl_xor_sync(uint32_t(-1), lower0, 2);
//       auto upper2 = __byte_perm(upper0, lower0, selectorEx2);
//       auto lower2 = __byte_perm(upper0, lower0, selectorEx3);
//       lower2 = __shfl_xor_sync(uint32_t(-1), lower2, 1);
//       upper = __byte_perm(upper2, lower2, selectorEx4);
//       lower = __byte_perm(upper2, lower2, selectorEx5);
  
//       uint32_t *data_32bit = reinterpret_cast<uint32_t *>(&data[n]);
//       data_32bit[0] = upper;
//       data_32bit[1] = lower;
//       n += 8;
//     }
//   }
// }
// };

// template <typename Fragment>
// CUTLASS_DEVICE static void reorgCFp8toAFp8NoShfl(Fragment &accum) {

//   using namespace cute;
//   auto laneId = cutlass::canonical_lane_idx();

//   // First update `mi` to the max per-row
//   //
//   auto VT = shape<0>(accum); // number of vector elements per tile.
//   auto MT = shape<1>(accum); // number of tiles along M.
//   auto NT = shape<2>(accum); // number of tiles along N.

//   auto data = accum.data();
//   int n = 0;

// #pragma unroll
//   for (int i = 0; i < MT; ++i) {

//     // Traverse 2-rows + 2-cols (2x2) simultaneously.

// #pragma unroll
//     for (int k = 0; k < NT * size<2>(VT) / 2; ++k) {

//       auto tmp0 = data[n + 2];
//       auto tmp1 = data[n + 3];
//       data[n + 2] = data[n + 4];
//       data[n + 3] = data[n + 5];

//       data[n + 4] = tmp0;
//       data[n + 5] = tmp1;
//       n += 8;
//     }
//   }
// }

// // First version (used for debugging)
// // TODO: Remove in future.
// template <typename Fragment>
// CUTLASS_DEVICE static void reorgCFp32toAFp8(Fragment &accum) {

//   using namespace cute;
//   auto laneId = cutlass::canonical_lane_idx();

//   // First update `mi` to the max per-row
//   //
//   auto VT = shape<0>(accum); // number of vector elements per tile.
//   auto MT = shape<1>(accum); // number of tiles along M.
//   auto NT = shape<2>(accum); // number of tiles along N.

//   auto data = accum.data();
//   int n = 0;

//   auto reduceMask = 0x66666666;
// #pragma unroll
//   for (int i = 0; i < MT; ++i) {

//     // Traverse 2-rows + 2-cols (2x2) simultaneously.

// #pragma unroll
//     for (int k = 0; k < NT * size<2>(VT) / 2; ++k) {
//       auto d0 = data[n];
//       auto d1 = data[n + 1];
//       auto d2 = data[n + 2];
//       auto d3 = data[n + 3];

//       auto d4 = data[n + 4];
//       auto d5 = data[n + 5];
//       auto d6 = data[n + 6];
//       auto d7 = data[n + 7];

//       // exhange d4, d5 with d0, d1.
//       float val00 = 0;
//       float val01 = 0;
//       float val10 = 0;
//       float val11 = 0;
//       int srcLane = 0;
//       if (laneId % 2 == 0) {
//         val00 = d4;
//         val01 = d5;
//         val10 = d6;
//         val11 = d7;
//         srcLane = laneId + 1;
//       } else {
//         val00 = d0;
//         val01 = d1;
//         val10 = d2;
//         val11 = d3;
//         srcLane = laneId - 1;
//       }
//       val00 = __shfl_sync(uint32_t(-1), val00, srcLane);
//       val01 = __shfl_sync(uint32_t(-1), val01, srcLane);
//       val10 = __shfl_sync(uint32_t(-1), val10, srcLane);
//       val11 = __shfl_sync(uint32_t(-1), val11, srcLane);

//       if (laneId % 2 == 0) {
//         d4 = val00;
//         d5 = val01;
//         d6 = val10;
//         d7 = val11;
//       } else {
//         d0 = val00;
//         d1 = val01;
//         d2 = val10;
//         d3 = val11;
//       }

//       // exhange d4, d5 with d0, d1.
//       srcLane = 0;
//       if (laneId % 4 == 1) {
//         srcLane = laneId + 1;
//       } else if (laneId % 4 == 2) {
//         srcLane = laneId - 1;
//       }

//       if (laneId % 4 == 1 || laneId % 4 == 2) {
//         d0 = __shfl_sync(reduceMask, d0, srcLane);
//         d1 = __shfl_sync(reduceMask, d1, srcLane);
//         d2 = __shfl_sync(reduceMask, d2, srcLane);
//         d3 = __shfl_sync(reduceMask, d3, srcLane);
//         d4 = __shfl_sync(reduceMask, d4, srcLane);
//         d5 = __shfl_sync(reduceMask, d5, srcLane);
//         d6 = __shfl_sync(reduceMask, d6, srcLane);
//         d7 = __shfl_sync(reduceMask, d7, srcLane);
//       }

//       data[n++] = d0;
//       data[n++] = d1;
//       data[n++] = d4;
//       data[n++] = d5;

//       data[n++] = d2;
//       data[n++] = d3;
//       data[n++] = d6;
//       data[n++] = d7;
//     }
//   }
// }

// template <typename CType, typename AType, typename Fragment>
// CUTLASS_DEVICE static void reorgCtoA(Fragment &accum) {

//   if constexpr (is_same_v<AType, cutlass::half_t>) {
//     // No reorg required; do nothing.
//     return;
//   } else if constexpr (is_same_v<AType, cutlass::float_e4m3_t>) {
//     if constexpr (is_same_v<CType, float>) {
//       reorgCFp32toAFp8(accum);
//     } else if constexpr (is_same_v<CType, cutlass::float_e4m3_t>) {
//       reorgCFp8toAFp8(accum);
//     } else {
//       static_assert(is_same_v<CType, float> ||
//                     is_same_v<CType, cutlass::float_e4m3_t>);
//     }
//   } else {
//     static_assert(is_same_v<AType, cutlass::half_t> ||
//                   is_same_v<AType, cutlass::float_e4m3_t>);
//   }
// }
