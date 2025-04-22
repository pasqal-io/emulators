import pulser
import torch
import random


def reciprocal_dist_matrix(reg: pulser.Register) -> torch.Tensor:
    """
    Matrix 1/r_{ij}. Behaves similarly as 1/r_{ij}^6
    """
    qubit_positions = list(reg.qubits.values())

    num_qubits = len(qubit_positions)
    distances = torch.zeros((num_qubits, num_qubits))

    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            x0, y0 = qubit_positions[i]
            x1, y1 = qubit_positions[j]
            value = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** (-1)

            # pulser internal type AbstractArray
            val_num = value.as_tensor().item()
            distances[i, j] = val_num
            distances[j, i] = val_num

    return distances


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

    permuted_reg = [None] * len(values)
    for i, p in enumerate(permutation):
        permuted_reg[i] = values[p]

    new_register = pulser.Register(dict(enumerate(permuted_reg)))
    return new_register
