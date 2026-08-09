# Extracted from sources/blogs/flashmla.md by scripts/extract_blog_code.py
# Heading: ## Sparse Index Contracts > ### Decode page-index round trip
# Original fence language: python
# See artifacts/blogs/flashmla/code/PROVENANCE.yaml for origin + license metadata.

# KernelWiki-derived contract check; not upstream FlashMLA code.
def encode_page_index(physical_page: int, offset: int, page_size: int) -> int:
    assert physical_page >= 0 and 0 <= offset < page_size
    return physical_page * page_size + offset

def decode_page_index(encoded: int, page_size: int) -> tuple[int, int]:
    assert encoded >= 0 and page_size > 0
    return divmod(encoded, page_size)

assert decode_page_index(encode_page_index(7, 13, 64), 64) == (7, 13)
