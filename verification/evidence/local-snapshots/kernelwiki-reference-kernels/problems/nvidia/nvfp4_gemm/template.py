from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    """
    Reference implementation of block-scale fp4 gemm
    Args:
        data: Tuple that expands to:
            a: torch.Tensor[float4e2m1fn] of shape [m, k, l],
            b: torch.Tensor[float4e2m1fn] of shape [n, k, l],
            sfa: torch.Tensor[float8_e4m3fnuz] of shape [m, k // 16, l],
            sfb: torch.Tensor[float8_e4m3fnuz] of shape [n, k // 16, l],
            sfa_permuted: torch.Tensor[float8_e4m3fnuz] of shape [32, 4, rest_m, 4, rest_k, l],
            sfb_permuted: torch.Tensor[float8_e4m3fnuz] of shape [32, 4, rest_n, 4, rest_k, l],
            c: torch.Tensor[float16] of shape [m, n, l]
    Returns:
        Tensor containing output in float16
        c: torch.Tensor[float16] of shape [m, n, l]
    """
    # c: [m, n, l] is pre-allocated memory to avoid timing allocation overhead.
    a, b, sfa, sfb, sfa_permuted, sfb_permuted, c = data

    # Your implementation here

    return c