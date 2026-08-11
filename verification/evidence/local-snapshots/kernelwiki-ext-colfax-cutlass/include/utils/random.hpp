#pragma once

#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/device/tensor_fill.h"

namespace cfk {
template <typename TIN, typename TOUT = TIN>
void initialize_rand(TIN *ptr, size_t capacity,
                     cutlass::Distribution::Kind dist_kind, uint32_t seed) {

  if (dist_kind == cutlass::Distribution::Uniform) {

    TIN scope_max, scope_min;
    int bits_input = cutlass::sizeof_bits<TIN>::value;
    int bits_output = cutlass::sizeof_bits<TOUT>::value;

    if (bits_input == 1) {
      scope_max = TIN(2);
      scope_min = TIN(0);
    } else if (bits_input <= 8) {
      scope_max = TIN(2);
      scope_min = TIN(-2);
    } else if (bits_output == 16) {
      scope_max = TIN(8);
      scope_min = TIN(-8);
    } else {
      scope_max = TIN(8);
      scope_min = TIN(-8);
    }

    cutlass::reference::device::BlockFillRandomUniform(ptr, capacity, seed,
                                                       scope_max, scope_min, 0);
  } else if (dist_kind == cutlass::Distribution::Gaussian) {

    cutlass::reference::device::BlockFillRandomGaussian(ptr, capacity, seed,
                                                        TIN(), TIN(1.0f));
  } else if (dist_kind == cutlass::Distribution::Sequential) {

    // Fill with increasing elements
    cutlass::reference::device::BlockFillSequential(ptr, capacity, TIN(1),
                                                    TIN());
  }
}

template <typename Element>
void initialize_const(Element *ptr, size_t capacity, const Element &value) {

  // Fill with all 1s
  cutlass::reference::device::BlockFillSequential(ptr, capacity, Element(),
                                                  value);
}

template <typename Element>
bool verify_tensor(thrust::host_vector<Element> vector_Input,
                   thrust::host_vector<Element> vector_Input_Ref,
                   bool printValues = false, bool printDiffs = false,
                   float errCountExpected = 0, int64_t verify_length = -1) {

  int64_t size = (vector_Input.size() < vector_Input_Ref.size())
                     ? vector_Input.size()
                     : vector_Input_Ref.size();
  size = (verify_length == -1) ? size : verify_length;

  // 0.005 for absolute error
  float abs_tol = 5e-3f;
  // 10% for relative error
  float rel_tol = 1e-1f;
  int errCount = 0;
  for (int64_t i = 0; i < size; ++i) {
    if (printValues)
      std::cout << vector_Input[i] << " " << vector_Input_Ref[i] << std::endl;
    float diff = (float)(vector_Input[i] - vector_Input_Ref[i]);
    float abs_diff = fabs(diff);
    float abs_ref = fabs((float)vector_Input_Ref[i] + 1e-5f);
    float relative_diff = abs_diff / abs_ref;
    if ((isnan(vector_Input_Ref[i]) || isnan(abs_diff) || isinf(abs_diff)) ||
        (abs_diff > abs_tol && relative_diff > rel_tol)) {
      if (printDiffs)
        printf("[%d/%d] diff = %f, rel_diff = %f, {computed=%f, ref=%f}.\n",
               int(i), int(size), abs_diff, relative_diff,
               (float)(vector_Input[i]), (float)(vector_Input_Ref[i]));
      errCount++;
      // return false;
    }
  }
  auto errCountComputed = float(errCount) / float(size) * 100;
  printf("Error (percentage) : %f\n", errCountComputed);

  return errCountComputed <= errCountExpected ? true : false;
}

} // namespace cfk
