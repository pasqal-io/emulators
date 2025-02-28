import torch


def randn_interaction_matrix(N: int):
    temp_mat = torch.randn(N, N).fill_diagonal_(0.0)
    interaction = (temp_mat + temp_mat.mT) / 2
    return interaction


def nn_interaction_matrix(N: int):
    interaction = torch.zeros(N, N)
    for i in range(N - 1):
        interaction[i, i + 1] = 1
        interaction[i + 1, i] = 1
    return interaction
