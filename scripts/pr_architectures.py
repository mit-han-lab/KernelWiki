"""Evidence-only architecture inference shared by PR generation and validation."""

from __future__ import annotations

import re


ARCH_ORDER = {
    arch: index for index, arch in enumerate(
        ("blackwell", "sm80", "sm90", "sm90a", "sm100", "sm100a", "sm103", "sm103a",
         "sm110", "sm110a", "sm120", "sm120a", "sm121", "sm121a")
    )
}


def infer_architectures(title: str, body: str, files: list[str]) -> list[str]:
    """Return only architectures with an explicit PR-text or path signal.

    Exact SM tokens and unambiguous product names are evidence. Family-only
    ``Blackwell`` is not collapsed to SM100 because it also names products in
    the SM103, SM120, and SM121 families.
    """
    text = " ".join((title, body, " ".join(files))).lower()
    found = set()

    for match in re.finditer(
        r"(?<![a-z0-9])sm[_-]?(80|90|100|103|110|120|121)(a)?(?![a-z0-9])",
        text,
    ):
        found.add(f"sm{match.group(1)}{'a' if match.group(2) else ''}")

    product_patterns = {
        # Underscores are word characters to ``\b``, so use alphanumeric
        # boundaries for instruction tokens embedded in file names such as
        # ``gemm_tcgen05.py`` and ``tmem_copy.cu``.
        "blackwell": r"\bblackwell\b|(?:^|/)blackwell(?:/|$)|(?<![a-z0-9])(?:tcgen05|tmem)(?![a-z0-9])",
        "sm80": r"\b(?:ampere|a100)\b",
        "sm90": r"\b(?:hopper|h100|h200|h800|gh100|gh200)\b",
        "sm100": r"\b(?:b100|b200|gb200)\b",
        "sm103": r"\b(?:b300|gb300)\b",
        "sm120": r"\b(?:rtx[ -]?50(?:70|80|90)?|geforce 50(?:70|80|90))\b",
        "sm121": r"\b(?:gb10|dgx spark)\b",
    }
    for arch, pattern in product_patterns.items():
        if re.search(pattern, text):
            found.add(arch)

    return sorted(found, key=lambda arch: ARCH_ORDER[arch])
