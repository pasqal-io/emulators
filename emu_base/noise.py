import torch


def pick_dark_qubits(eta: float, n: int) -> torch.Tensor:
    """
    Randomly pick n booleans such that ℙ(True) = eta

    Returns:
        A BoolTensor of shape (n,) where each element is True with probability eta.
    """
    return torch.rand(n) < eta
