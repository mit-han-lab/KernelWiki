---
id: blog-flashmla
title: FlashMLA — Multi-head Latent Attention
author: DeepSeek AI
url: https://github.com/deepseek-ai/FlashMLA/tree/71c737929f2567bd0a094ae140f8f60f390b1232
source_category: benchmark-blog
architectures:
- sm100
- sm90
tags:
- mla
- attention
- decode
- prefill
- fp8
- sparse-attention
- tcgen05
- tmem
retrieved_at: 2026-04-27
artifact_dir: artifacts/blogs/flashmla/code
---

## Captured Revision

This source record summarizes DeepSeek FlashMLA commit `71c737929f2567bd0a094ae140f8f60f390b1232` (2026-03-31). It is a repository/benchmark capture; code below is explicitly KernelWiki-derived and is not copied from the upstream implementation.

## Supported Operators

| Operator | Architecture | Mode and format |
|---|---|---|
| Dense decode | SM90 | MQA dimensions `576/512`, BF16 KV cache |
| Sparse decode | SM90/SM100 | MQA dimensions, FP8 KV cache dequantized for BF16 MMA |
| Dense prefill | SM100 | MHA dimensions `192/128` or `128/128`, BF16 inputs |
| Sparse prefill | SM90/SM100 | MQA mode, BF16 Q/KV inputs |

The repository requires CUDA 12.8+, CUDA 12.9+ for SM100, and PyTorch 2.0+.

## Exact V3 FP8 Sparse-Decode Layout

When `is_fp8_kvcache=True` for the DeepSeek-V3-family sparse-decode path, each token uses 512 E4M3 NoPE bytes, four FP32 group scales (16 bytes), and 64 BF16 RoPE values (128 bytes), totaling 656 bytes. Dense decode and other sparse-cache layouts are different.

### V3 FP8 sparse-decode byte check

```python
# KernelWiki-derived contract check; not upstream FlashMLA code.
NOPE_FP8_VALUES = 512
GROUPS = 4
FP32_BYTES = 4
ROPE_BF16_VALUES = 64
BF16_BYTES = 2

V3_FP8_SPARSE_BYTES = (
    NOPE_FP8_VALUES + GROUPS * FP32_BYTES + ROPE_BF16_VALUES * BF16_BYTES
)
assert V3_FP8_SPARSE_BYTES == 656
```

## Sparse Index Contracts

Sparse decode consumes `indices[batch,s_q,topk]` whose entries encode a physical page and offset. Invalid entries are `-1`, and no block table is needed after physical-page encoding. Sparse prefill instead consumes unbatched `indices[s_q,h_kv,topk]` with BF16 Q/KV and accepts negative or out-of-range invalid entries.

### Decode page-index round trip

```python
# KernelWiki-derived contract check; not upstream FlashMLA code.
def encode_page_index(physical_page: int, offset: int, page_size: int) -> int:
    assert physical_page >= 0 and 0 <= offset < page_size
    return physical_page * page_size + offset

def decode_page_index(encoded: int, page_size: int) -> tuple[int, int]:
    assert encoded >= 0 and page_size > 0
    return divmod(encoded, page_size)

assert decode_page_index(encode_page_index(7, 13, 64), 64) == (7, 13)
```

## Source-Reported Performance

- Dense MLA decode, H800 SXM5/CUDA 12.8: up to 3000 GB/s memory-bound and 660 TFLOPS compute-bound.
- Sparse MLA decode: 410 TFLOPS on H800 SXM5/CUDA 12.8 and up to 350 TFLOPS on B200; FP8 describes KV storage and MMA is BF16.
- Dense **MHA** prefill, B200: up to 1460 TFLOPS forward and 1000 TFLOPS backward, reported by NVIDIA.
- Sparse MLA prefill: up to 640 TFLOPS forward on H800 SXM5/CUDA 12.8 and up to 1450 TFLOPS on B200/CUDA 12.9; documented Q/KV inputs are BF16.

The repository does not supply complete shape/timing/sample/variance records for these maxima. They remain qualified author reports rather than structured benchmark entries.
