"""Derived CPU reference for the GPU Mode NVFP4 dual-GEMM result semantics.

This intentionally does not model NVFP4 encoding, scale layout, or GPU execution.
It checks the two shared-A matrix products and the order of SiLU and multiply.
"""

from __future__ import annotations

import math


Matrix = list[list[float]]


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _project(a: Matrix, b_nk: Matrix) -> Matrix:
    if not a or not b_nk or not a[0]:
        raise ValueError("matrices must be non-empty")
    k = len(a[0])
    if any(len(row) != k for row in a):
        raise ValueError("A must be rectangular")
    if any(len(row) != k for row in b_nk):
        raise ValueError("B must have shape [N,K]")
    return [[sum(x * y for x, y in zip(a_row, b_row)) for b_row in b_nk]
            for a_row in a]


def gated_dual_gemm(a: Matrix, b1_nk: Matrix, b2_nk: Matrix) -> Matrix:
    """Return SiLU(A @ B1.T) * (A @ B2.T)."""
    if len(b1_nk) != len(b2_nk):
        raise ValueError("B1 and B2 must have the same N")
    gate = _project(a, b1_nk)
    up = _project(a, b2_nk)
    return [[g * _sigmoid(g) * u for g, u in zip(g_row, u_row)]
            for g_row, u_row in zip(gate, up)]


def _assert_close(actual: Matrix, expected: Matrix) -> None:
    for actual_row, expected_row in zip(actual, expected):
        for actual_value, expected_value in zip(actual_row, expected_row):
            assert math.isclose(actual_value, expected_value, rel_tol=1e-12,
                                abs_tol=1e-12)


def _self_test() -> None:
    a = [[1.0, -2.0], [0.5, 3.0]]
    b1 = [[2.0, 1.0], [-1.0, 0.5]]
    b2 = [[0.25, 2.0], [1.0, -3.0]]

    gate = _project(a, b1)
    up = _project(a, b2)
    expanded = [[g * _sigmoid(g) * u for g, u in zip(g_row, u_row)]
                for g_row, u_row in zip(gate, up)]
    actual = gated_dual_gemm(a, b1, b2)
    _assert_close(actual, expanded)

    wrong_branch = [[u * _sigmoid(u) * g for g, u in zip(g_row, u_row)]
                    for g_row, u_row in zip(gate, up)]
    assert any(not math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12)
               for actual_row, wrong_row in zip(actual, wrong_branch)
               for x, y in zip(actual_row, wrong_row))


if __name__ == "__main__":
    _self_test()
    print("gated dual GEMM semantic reference: PASS")
