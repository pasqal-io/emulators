import torch


def dist2(left: torch.Tensor, right: torch.Tensor) -> float:
    return torch.dist(left, right).item() ** 2


def dist3(left: torch.Tensor, right: torch.Tensor) -> float:
    return torch.dist(left, right).item() ** 3
