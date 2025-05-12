import torch


def randn_interaction_matrix(
    N: int, dtype: torch.dtype = torch.float64, requires_grad: bool = False
):
    interaction = torch.zeros(N, N, dtype=dtype)
    for i in range(N):
        for j in range(i + 1, N):
            temp = torch.rand(1, dtype=dtype)
            interaction[i, j] = temp
            interaction[j, i] = temp
    interaction.requires_grad = requires_grad
    return interaction


def nn_interaction_matrix(
    N: int, dtype: torch.dtype = torch.float64, requires_grad: bool = False
):
    interaction = torch.zeros(N, N, dtype=dtype)
    for i in range(N - 1):
        interaction[i, i + 1] = 1.0
        interaction[i + 1, i] = 1.0
    interaction.requires_grad = requires_grad
    return interaction
