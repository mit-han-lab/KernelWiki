#include <cutlass/cutlass.h>

template <class ElementA, class ElementB, class ElementC, class TileShape_MNK,
          GMMA::Major MajorA = GMMA::Major::K,
          GMMA::Major MajorB = GMMA::Major::K,
          auto... Args // e.g. GMMA::ScaleOut::One, [GMMA::ScaleIn::One,
                       // GMMA::ScaleIn::One] But most commonly leave empty for
                       // defaults
          >
CUTE_HOST_DEVICE constexpr auto ss_op_selector_custom() {
  static_assert(is_static<TileShape_MNK>::value,
                "TileShape_MNK must be static.");
  static_assert(rank(TileShape_MNK{}) == 3, "TileShape_MNK must be rank 3.");
  static_assert(size<0>(TileShape_MNK{}) % 64 == 0,
                "Tile_M must be a multiple of 64.");
  auto Tile_N = size<1>(TileShape_MNK{});

  // FP16 accumulator
  if constexpr (is_same_v<ElementC, half_t> &&
                is_same_v<ElementA, float_e4m3_t> &&
                is_same_v<ElementB, float_e4m3_t>) {

    if constexpr (Tile_N % 256 == 0) {
      return SM90_64x256x32_F16E4M3E4M3_SS_TN<Args...>{};
    } else if constexpr (Tile_N % 192 == 0) {
      return SM90_64x192x32_F16E4M3E4M3_SS_TN<Args...>{};
    } else if constexpr (Tile_N % 128 == 0) {
      return SM90_64x128x32_F16E4M3E4M3_SS_TN<Args...>{};
    } else if constexpr (Tile_N % 96 == 0) {
      return SM90_64x96x32_F16E4M3E4M3_SS_TN<Args...>{};
    } else if constexpr (Tile_N % 64 == 0) {
      return SM90_64x64x32_F16E4M3E4M3_SS_TN<Args...>{};
    } else if constexpr (Tile_N % 32 == 0) {
      return SM90_64x32x32_F16E4M3E4M3_SS_TN<Args...>{};
    } else if constexpr (Tile_N % 16 == 0) {
      return SM90_64x16x32_F16E4M3E4M3_SS_TN<Args...>{};
    } else if constexpr (Tile_N % 8 == 0) {
      return SM90_64x8x32_F16E4M3E4M3_SS_TN<Args...>{};
    } else {
      static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
    }

  } else {

    return cute::GMMA::ss_op_selector<ElementA, ElementB, ElementC,
                                      TileShape_MNK, MajorA, MajorB, Args...>();
  }
}

template <class ElementA, class ElementB, class ElementC, class TileShape_MNK,
          GMMA::Major MajorA = GMMA::Major::K,
          GMMA::Major MajorB = GMMA::Major::K,
          auto... Args // e.g. GMMA::ScaleOut::One, [GMMA::ScaleIn::One,
                       // GMMA::ScaleIn::One] But most commonly leave empty for
                       // defaults
          >
CUTE_HOST_DEVICE constexpr auto rs_op_selector_custom() {
  static_assert(is_static<TileShape_MNK>::value,
                "TileShape_MNK must be static.");
  static_assert(rank(TileShape_MNK{}) == 3, "TileShape_MNK must be rank 3.");
  static_assert(size<0>(TileShape_MNK{}) % 64 == 0,
                "Tile_M must be a multiple of 64.");
  auto Tile_N = size<1>(TileShape_MNK{});

  // FP16 accumulator
  if constexpr (is_same_v<ElementC, half_t> &&
                is_same_v<ElementA, float_e4m3_t> &&
                is_same_v<ElementB, float_e4m3_t>) {

    if constexpr (Tile_N % 256 == 0) {
      return SM90_64x256x32_F16E4M3E4M3_RS_TN<Args...>{};
    } else if constexpr (Tile_N % 192 == 0) {
      return SM90_64x192x32_F16E4M3E4M3_RS_TN<Args...>{};
    } else if constexpr (Tile_N % 128 == 0) {
      return SM90_64x128x32_F16E4M3E4M3_RS_TN<Args...>{};
    } else if constexpr (Tile_N % 96 == 0) {
      return SM90_64x96x32_F16E4M3E4M3_RS_TN<Args...>{};
    } else if constexpr (Tile_N % 64 == 0) {
      return SM90_64x64x32_F16E4M3E4M3_RS_TN<Args...>{};
    } else if constexpr (Tile_N % 32 == 0) {
      return SM90_64x32x32_F16E4M3E4M3_RS_TN<Args...>{};
    } else if constexpr (Tile_N % 16 == 0) {
      return SM90_64x16x32_F16E4M3E4M3_RS_TN<Args...>{};
    } else if constexpr (Tile_N % 8 == 0) {
      return SM90_64x8x32_F16E4M3E4M3_RS_TN<Args...>{};
    } else {
      static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
    }
  } else {

    return cute::GMMA::rs_op_selector<ElementA, ElementB, ElementC,
                                      TileShape_MNK, MajorA, MajorB, Args...>();
  }
}
