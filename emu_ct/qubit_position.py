import torch


def dist2(left: torch.tensor, right: torch.tensor) -> torch.Tensor:
    return torch.norm(left - right) ** 2
