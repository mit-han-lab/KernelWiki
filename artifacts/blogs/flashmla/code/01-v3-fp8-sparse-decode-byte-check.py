# Extracted from sources/blogs/flashmla.md by scripts/extract_blog_code.py
# Heading: ## Exact V3 FP8 Sparse-Decode Layout > ### V3 FP8 sparse-decode byte check
# Original fence language: python
# See artifacts/blogs/flashmla/code/PROVENANCE.yaml for origin + license metadata.

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
