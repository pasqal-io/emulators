import torch


def dist2(left: torch.Tensor, right: torch.Tensor) -> float:
    return torch.dist(left, right).item() ** 2


def dist3(left: torch.Tensor, right: torch.Tensor) -> float:
    return torch.dist(left, right).item() ** 3


def deallocate_tensor(t: torch.Tensor) -> None:
    t.set_(source=torch.zeros(0, dtype=t.dtype, device=t.device).untyped_storage())
