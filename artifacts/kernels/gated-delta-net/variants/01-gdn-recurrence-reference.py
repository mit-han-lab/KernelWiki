"""One-head Gated DeltaNet recurrence; derived, not a tuned GPU kernel.

Reference semantics: FlashInfer commit
7f614b86470180bab2d22e36fd1775791c6bf3e6, trace/templates/gdn.py.
"""

from __future__ import annotations

from math import isclose


Matrix = list[list[float]]


def dot(left: list[float], right: list[float]) -> float:
    assert len(left) == len(right) and left
    return sum(a * b for a, b in zip(left, right))


def outer(left: list[float], right: list[float]) -> Matrix:
    return [[a * b for b in right] for a in left]


def add(left: Matrix, right: Matrix) -> Matrix:
    assert len(left) == len(right) and len(left[0]) == len(right[0])
    return [[a + b for a, b in zip(x, y)] for x, y in zip(left, right)]


def scale_matrix(factor: float, matrix: Matrix) -> Matrix:
    return [[factor * value for value in row] for row in matrix]


def vector_matrix(vector: list[float], matrix: Matrix) -> list[float]:
    assert len(vector) == len(matrix) and matrix
    return [
        sum(vector[row] * matrix[row][column] for row in range(len(matrix)))
        for column in range(len(matrix[0]))
    ]


def gdn_step(
    state: Matrix,
    q: list[float],
    k: list[float],
    v: list[float],
    decay: float,
    beta: float,
    scale: float,
) -> tuple[list[float], Matrix]:
    """Apply the compact gated-delta update to state shaped [K,V]."""
    assert 0.0 <= decay <= 1.0 and 0.0 <= beta <= 1.0
    assert len(state) == len(q) == len(k) and len(state[0]) == len(v)
    decayed = scale_matrix(decay, state)
    retrieved = vector_matrix(k, decayed)
    correction = [beta * (target - old) for target, old in zip(v, retrieved)]
    new_state = add(decayed, outer(k, correction))
    output = [scale * value for value in vector_matrix(q, new_state)]
    return output, new_state


def _expanded_reference(
    state: Matrix, k: list[float], v: list[float], decay: float, beta: float
) -> Matrix:
    decayed = scale_matrix(decay, state)
    old_v = vector_matrix(k, decayed)
    new_v = [beta * target + (1.0 - beta) * old for target, old in zip(v, old_v)]
    removed = scale_matrix(-1.0, outer(k, old_v))
    return add(add(decayed, removed), outer(k, new_v))


def _close(left: Matrix, right: Matrix) -> bool:
    return all(
        isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
        for x, y in zip(left, right)
        for a, b in zip(x, y)
    )


def _self_test() -> None:
    state = [[0.2, -0.3], [0.4, 0.1]]
    q = [0.6, -0.8]
    k = [0.8, 0.6]
    v = [0.5, -0.25]
    decay, beta, output_scale = 0.75, 0.4, 0.5

    output, compact = gdn_step(state, q, k, v, decay, beta, output_scale)
    expanded = _expanded_reference(state, k, v, decay, beta)
    assert _close(compact, expanded)
    assert all(isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(
        output, [output_scale * x for x in vector_matrix(q, expanded)]
    ))

    additive_only = add(scale_matrix(decay, state), outer(k, v))
    assert not _close(compact, additive_only)
    old_output = [output_scale * x for x in vector_matrix(q, scale_matrix(decay, state))]
    assert output != old_output
    print("gated-delta recurrence reference: PASS")


if __name__ == "__main__":
    _self_test()
