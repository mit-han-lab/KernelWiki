# provenance: derived from contest-flashinfer-track-a; not upstream code
# Scope: CPU-checkable semantics only; not an optimized or fused GPU kernel.

from __future__ import annotations

import math
import random


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def deepseek_grouped_topk(
    routing_logits: list[list[float]],
    routing_bias: list[float],
    *,
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
) -> tuple[list[list[int]], list[list[float]]]:
    """Return global expert IDs and combine weights for DeepSeek-V3 routing.

    Bias affects expert selection but not the normalized combine weights.
    """
    if not routing_logits or not routing_logits[0]:
        raise ValueError("routing_logits must have shape [tokens, experts]")
    experts = len(routing_logits[0])
    if any(len(row) != experts for row in routing_logits):
        raise ValueError("routing_logits rows must have equal length")
    if len(routing_bias) != experts:
        raise ValueError("routing_bias must have shape [experts]")
    if experts % n_group:
        raise ValueError("experts must be divisible by n_group")
    group_size = experts // n_group
    if not (1 <= topk_group <= n_group and 1 <= top_k <= topk_group * group_size):
        raise ValueError("invalid grouped top-k parameters")
    if group_size < 2:
        raise ValueError("DeepSeek group scoring requires at least two experts per group")

    all_ids: list[list[int]] = []
    all_weights: list[list[float]] = []
    for row in routing_logits:
        sigmoid_scores = [_sigmoid(value) for value in row]
        selection_scores = [
            score + bias for score, bias in zip(sigmoid_scores, routing_bias, strict=True)
        ]
        group_scores = []
        for group in range(n_group):
            begin = group * group_size
            two_largest = sorted(
                selection_scores[begin : begin + group_size], reverse=True
            )[:2]
            group_scores.append((sum(two_largest), group))
        kept_groups = {
            group
            for _, group in sorted(group_scores, reverse=True)[:topk_group]
        }
        candidates = [
            expert
            for expert in range(experts)
            if expert // group_size in kept_groups
        ]
        chosen = sorted(
            candidates, key=lambda expert: selection_scores[expert], reverse=True
        )[:top_k]
        raw_weights = [sigmoid_scores[expert] for expert in chosen]
        denominator = sum(raw_weights)
        all_ids.append(chosen)
        all_weights.append(
            [
                weight / denominator * routed_scaling_factor
                for weight in raw_weights
            ]
        )
    return all_ids, all_weights


def _matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    if any(len(row) != len(vector) for row in matrix):
        raise ValueError("matrix/vector dimensions do not match")
    return [sum(a * b for a, b in zip(row, vector, strict=True)) for row in matrix]


def local_expert_reference(
    hidden_states: list[list[float]],
    topk_ids: list[list[int]],
    combine_weights: list[list[float]],
    w13: list[list[list[float]]],
    w2: list[list[list[float]]],
    *,
    local_expert_offset: int,
) -> list[list[float]]:
    """Compute local W13/SwiGLU/W2 contributions and weighted accumulation.

    Logical shapes are hidden_states[T,H], w13[E_local,2I,H], and
    w2[E_local,H,I].
    """
    if not hidden_states or not hidden_states[0] or len(w13) != len(w2):
        raise ValueError("empty or mismatched input")
    tokens, hidden = len(hidden_states), len(hidden_states[0])
    if len(topk_ids) != tokens or len(combine_weights) != tokens:
        raise ValueError("top-k rows must match tokens")
    if any(len(ids) != len(weights) for ids, weights in zip(topk_ids, combine_weights, strict=True)):
        raise ValueError("top-k IDs and weights must have equal row lengths")
    output = [[0.0] * hidden for _ in range(tokens)]

    for local_id, (expert_w13, expert_w2) in enumerate(zip(w13, w2, strict=True)):
        if not expert_w13 or len(expert_w13) % 2:
            raise ValueError("w13 rows must equal 2 * intermediate")
        intermediate = len(expert_w13) // 2
        if len(expert_w2) != hidden or any(len(row) != intermediate for row in expert_w2):
            raise ValueError("w2 must have shape [hidden, intermediate]")
        global_id = local_expert_offset + local_id
        for token, hidden_row in enumerate(hidden_states):
            matching_weight = sum(
                weight
                for expert, weight in zip(
                    topk_ids[token], combine_weights[token], strict=True
                )
                if expert == global_id
            )
            if matching_weight == 0.0:
                continue
            projected = _matvec(expert_w13, hidden_row)
            up, gate = projected[:intermediate], projected[intermediate:]
            activated = [
                _sigmoid(gate_value) * gate_value * up_value
                for gate_value, up_value in zip(gate, up, strict=True)
            ]
            expert_output = _matvec(expert_w2, activated)
            for column, value in enumerate(expert_output):
                output[token][column] += matching_weight * value
    return output


def fused_moe_reference(
    hidden_states: list[list[float]],
    routing_logits: list[list[float]],
    routing_bias: list[float],
    w13: list[list[list[float]]],
    w2: list[list[list[float]]],
    *,
    local_expert_offset: int,
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
) -> list[list[float]]:
    topk_ids, combine_weights = deepseek_grouped_topk(
        routing_logits,
        routing_bias,
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=routed_scaling_factor,
    )
    return local_expert_reference(
        hidden_states,
        topk_ids,
        combine_weights,
        w13,
        w2,
        local_expert_offset=local_expert_offset,
    )


def _self_test() -> None:
    logits = [
        [2.0, 1.0, -1.0, -2.0],
        [-1.5, -0.5, 0.5, 1.5],
        [0.1, 0.3, 0.2, 0.0],
    ]
    bias = [0.0, 0.2, 0.4, -0.1]
    ids, weights = deepseek_grouped_topk(
        logits,
        bias,
        top_k=2,
        n_group=2,
        topk_group=1,
        routed_scaling_factor=2.5,
    )
    assert ids == [[1, 0], [2, 3], [2, 3]]
    assert all(math.isclose(sum(row), 2.5, rel_tol=1e-12) for row in weights)

    # Negative control: biased selection scores must not become combine weights.
    first_biased = [_sigmoid(value) + delta for value, delta in zip(logits[0], bias, strict=True)]
    wrong_denominator = sum(first_biased[expert] for expert in ids[0])
    wrong = [first_biased[expert] / wrong_denominator * 2.5 for expert in ids[0]]
    assert any(not math.isclose(a, b) for a, b in zip(weights[0], wrong, strict=True))

    rng = random.Random(7)
    hidden = [[rng.uniform(-1, 1) for _ in range(4)] for _ in range(3)]
    w13 = [
        [[rng.uniform(-1, 1) for _ in range(4)] for _ in range(6)]
        for _ in range(4)
    ]
    w2 = [
        [[rng.uniform(-1, 1) for _ in range(3)] for _ in range(4)]
        for _ in range(4)
    ]
    output = local_expert_reference(
        hidden, ids, weights, w13, w2, local_expert_offset=0
    )
    assert len(output) == 3 and all(len(row) == 4 for row in output)
    assert all(math.isfinite(value) for row in output for value in row)


if __name__ == "__main__":
    _self_test()
    print("fused-moe derived reference: PASS")
