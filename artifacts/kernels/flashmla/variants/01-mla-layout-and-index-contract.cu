// provenance: derived from blog-flashmla at DeepSeek FlashMLA commit 71c7379;
// not upstream code
// Scope: V3-family FP8 sparse-decode byte layout and encoded page-index arithmetic.

#include <cassert>
#include <utility>

namespace kernelwiki_flashmla_contract {

constexpr int kNopeFp8Bytes = 512;
constexpr int kScaleCount = 4;
constexpr int kFp32Bytes = 4;
constexpr int kRopeBf16Values = 64;
constexpr int kBf16Bytes = 2;
constexpr int kV3Fp8SparseBytes =
    kNopeFp8Bytes + kScaleCount * kFp32Bytes + kRopeBf16Values * kBf16Bytes;
static_assert(kV3Fp8SparseBytes == 656);

// FlashMLA sparse decode consumes this physical page/offset encoding.
// The caller must handle -1 as an invalid entry before decoding it.
constexpr int encode_page_index(int physical_page, int offset, int page_size) {
  return physical_page * page_size + offset;
}

constexpr std::pair<int, int> decode_page_index(int encoded, int page_size) {
  return {encoded / page_size, encoded % page_size};
}

static_assert(decode_page_index(encode_page_index(7, 13, 64), 64) ==
              std::pair<int, int>{7, 13});

}  // namespace kernelwiki_flashmla_contract
