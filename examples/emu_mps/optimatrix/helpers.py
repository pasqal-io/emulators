import pulser
import torch
import random


def reciprocal_dist_matrix(reg: pulser.Register) -> torch.Tensor:
    """
    Matrix 1/r_{ij}. Behaves similarly as 1/r_{ij}^6
    """
    positions = torch.stack(
        [q.as_tensor() for q in reg.qubits.values()],
        dim=0,
        )

    diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    dist = (diff ** 2).sum(-1).pow(0.5)

    mask = dist != 0
    dist_inv = torch.zeros_like(dist)
    dist_inv[mask] = 1.0 / dist[mask]

    return dist_inv


def shuffle_qubits(reg: pulser.Register) -> pulser.Register:
    qubits = reg.qubits
    keys = list(qubits.keys())
    new_values = list(qubits.values())
    random.shuffle(new_values)
    shuffled_reg = pulser.Register(dict(zip(keys, new_values)))
    return shuffled_reg


def permute_sequence_registers(
    reg: pulser.Register, permutation: list
) -> pulser.Register:
    values = list(reg.qubits.values())

    new_register = pulser.Register({i: values[permutation[i]] for i in range(len(values))})
    return new_register
