#pragma once

#include "cutlass/detail/layout.hpp"

// Shared Storage for aligned addresses
template <class Element, class SmemLayout> struct SharedStorageTranspose {
  cute::array_aligned<Element, cute::cosize_v<SmemLayout>,
                      cutlass::detail::alignment_for_swizzle(SmemLayout{})>
      smem;
};

// Shared Storage for aligned addresses
template <class Element, class SmemLayout> struct SharedStorageCopy {
  cute::array_aligned<Element, cute::cosize_v<SmemLayout>>
      smem;
};

// Shared Storage for aligned addresses
template <class Element, class SmemLayoutA, class SmemLayoutB, class ThreadLayout> struct SharedStorageCopyFma {
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutA>> smem_a;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutB>> smem_b;
  cute::array_aligned<Element, cute::cosize_v<ThreadLayout>> smem_out;
};

template <class Element, class SmemLayoutA, class ThreadLayout> struct SharedStorageCopyFmaA {
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutA>> smem_a;
  cute::array_aligned<Element, cute::cosize_v<ThreadLayout>> smem_out;
};