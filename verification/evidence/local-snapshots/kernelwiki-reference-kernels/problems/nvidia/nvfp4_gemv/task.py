import torch
from typing import TypedDict, TypeVar

input_t = TypeVar("input_t", bound=tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor])
output_t = TypeVar("output_t", bound=torch.Tensor)
class TestSpec(TypedDict):
    m: int
    k: int
    l: int
    seed: int
