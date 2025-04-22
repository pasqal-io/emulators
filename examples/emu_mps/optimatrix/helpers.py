import pulser
import numpy as np
import torch
import matplotlib.pyplot as plt
import emu_mps.optimatrix as optimatrix


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


def plot_matrices(matrix1: np.ndarray, matrix2: np.ndarray) -> None:
    # Create a figure with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the first matrix heatmap
    cax1 = axes[0].imshow(matrix1, cmap="Blues")
    axes[0].set_title("Initial matrix")
    fig.colorbar(cax1, ax=axes[0])

    # Plot the second matrix heatmap
    cax2 = axes[1].imshow(matrix2, cmap="Blues")
    axes[1].set_title("Optimised matrix")
    fig.colorbar(cax2, ax=axes[1])

    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.show()


def shuffle_qubits(reg: pulser.Register) -> pulser.Register:
    import random

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
    new_values = optimatrix.permute_list(values, permutation)
    new_register = pulser.Register(dict(enumerate(new_values)))
    return new_register
