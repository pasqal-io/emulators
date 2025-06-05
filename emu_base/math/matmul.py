import torch


def matmul_2x2_with_batched(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(right)
    zero = torch.tensor(0, device=right.device)
    one = torch.tensor(1, device=right.device)
    result = result.index_add_(
        1,
        zero,
        right.select(1, 0).unsqueeze(1),
        alpha=left[0, 0],  # type: ignore [arg-type]
    )
    result = result.index_add_(
        1,
        zero,
        right.select(1, 1).unsqueeze(1),
        alpha=left[0, 1],  # type: ignore [arg-type]
    )
    result = result.index_add_(
        1,
        one,
        right.select(1, 0).unsqueeze(1),
        alpha=left[1, 0],  # type: ignore [arg-type]
    )
    result = result.index_add_(
        1,
        one,
        right.select(1, 1).unsqueeze(1),
        alpha=left[1, 1],  # type: ignore [arg-type]
    )
    return result
