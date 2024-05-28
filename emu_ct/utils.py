from typing import List

import torch


def _determine_cutoff_index(d: torch.Tensor, max_error: float) -> int:
    assert max_error > 0
    squared_max_error = max_error * max_error
    acc = 0
    for i in range(d.shape[0] - 1, -1, -1):
        acc += d[i] * d[i]
        if acc > squared_max_error:
            return i + 1
    return d.shape[0]  # type: ignore[no-any-return]


"""
Computes a low-rank approximation svd of m using the Eckart-Young-Mirsky theorem.
"""


def truncated_svd(
    m: torch.Tensor,
    max_error: float = 1e-5,
    max_rank: int = 1024,
    full_matrices: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u, d, vh = torch.linalg.svd(m, full_matrices=full_matrices)
    max_bond = min(
        _determine_cutoff_index(d, max_error),
        max_rank,
    )
    u = u[:, :max_bond]
    d = d[:max_bond]
    vh = vh[:max_bond, :]

    return u, d, vh


"""
Evenly distributes each tensor in the list to a device.
If num_devices_to_use is 0, then all tensors go to CPU.
"""


def assign_devices(tensors: List[torch.Tensor], num_devices_to_use: int) -> None:
    num_devices_to_use = min(len(tensors), num_devices_to_use)

    if num_devices_to_use <= 0:
        for i in range(len(tensors)):
            tensors[i] = tensors[i].to("cpu")
        return

    tensors_per_device = len(tensors) // num_devices_to_use

    if len(tensors) % num_devices_to_use != 0:
        tensors_per_device += 1

    for i in range(len(tensors)):
        tensors[i] = tensors[i].to(f"cuda:{i // tensors_per_device}")
