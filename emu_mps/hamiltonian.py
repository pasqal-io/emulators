"""
This file deals with creation of the MPO corresponding
to the Hamiltonian of a neutral atoms quantum processor.
"""

import pulser
import torch

from emu_mps.mpo import MPO
from emu_mps.pulser_adapter import get_qubit_positions
from emu_mps.utils import dist2


def _first_factor(gate: torch.Tensor) -> torch.Tensor:
    """
    Creates the first Hamiltonian factor.
    """
    fac = torch.zeros(1, 2, 2, 3, dtype=gate.dtype)
    fac[0, 0, 0, 1] = 1
    fac[0, 1, 1, 1] = 1
    fac[0, 1, 1, 2] = 1
    fac = fac.to(gate.device)
    fac[0, :, :, 0] = gate
    return fac


def _last_factor(gate: torch.Tensor, scale: float | complex) -> torch.Tensor:
    """
    Creates the last Hamiltonian factor.
    """
    fac = torch.zeros(3, 2, 2, 1, dtype=gate.dtype)
    fac[0, 0, 0, 0] = 1
    fac[0, 1, 1, 0] = 1
    fac[2, 1, 1, 0] = scale
    fac = fac.to(gate.device)
    fac[1, :, :, 0] = gate
    return fac


def _left_factor(gate: torch.Tensor, scales: list[float]) -> torch.Tensor:
    """
    Creates the Hamiltonian factors in the left half of the MPS, excepted the first factor.
    """
    index = len(scales)
    fac = torch.zeros(index + 2, 2, 2, index + 3, dtype=gate.dtype)
    for i, val in enumerate(scales):
        fac[i + 2, 1, 1, 0] = val  # rydberg interaction with previous qubits
    fac[1, 1, 1, index + 2] = 1  # rydberg interaction with next qubits
    for i in range(index + 2):
        fac[i, 0, 0, i] = 1
        fac[i, 1, 1, i] = 1  # identity matrix to carry the gates of other qubits
    fac = fac.to(gate.device)
    fac[1, :, :, 0] = gate
    return fac


def _right_factor(gate: torch.Tensor, scales: list[float]) -> torch.Tensor:
    """
    Creates the Hamiltonian factors in the right half of the MPS, excepted the last factor.
    """
    index = len(scales)
    fac = torch.zeros(index + 3, 2, 2, index + 2, dtype=gate.dtype)
    for i, val in enumerate(scales):
        fac[1, 1, 1, i + 2] = val  # rydberg interaction with previous qubits
    fac[2, 1, 1, 0] = 1  # rydberg interaction with next qubits
    for i in range(2, index + 2):
        fac[i + 1, 0, 0, i] = 1
        fac[
            i + 1, 1, 1, i
        ] = 1  # identity matrix to carry previous interactions to the next qubits
    fac[0, 0, 0, 0] = 1
    fac[0, 1, 1, 0] = 1  # identity to carry the next gates to the previous qubits
    fac[1, 0, 0, 1] = 1
    fac[1, 1, 1, 1] = 1  # identity to carry previous gates to next qubits
    fac = fac.to(gate.device)
    fac[1, :, :, 0] = gate
    return fac


def _middle_factor(
    gate: torch.Tensor,
    scales_l: list[float],
    scales_r: list[float],
    scales_mat: list[list[float]],
) -> torch.Tensor:
    """
    Creates the Hamiltonian factor at index ⌊n/2⌋ of the n-qubit MPO.
    """
    assert len(scales_mat) == len(scales_l)
    assert all(len(x) == len(scales_r) for x in scales_mat)

    fac = torch.zeros(len(scales_l) + 2, 2, 2, len(scales_r) + 2, dtype=gate.dtype)
    for i, val in enumerate(scales_r):
        fac[1, 1, 1, i + 2] = val  # rydberg interaction with previous qubits
    for i, val in enumerate(scales_l):
        fac[i + 2, 1, 1, 0] = val  # rydberg interaction with next qubits
    for i, row in enumerate(scales_mat):
        for j, val in enumerate(row):
            fac[
                i + 2, 0, 0, j + 2
            ] = val  # rydberg interaction of previous with next qubits
            fac[i + 2, 1, 1, j + 2] = val
    fac[0, 0, 0, 0] = 1
    fac[0, 1, 1, 0] = 1  # identity to carry the next gates to the previous qubits
    fac[1, 0, 0, 1] = 1
    fac[1, 1, 1, 1] = 1  # identity to carry previous gates to next qubits
    fac = fac.to(gate.device)
    fac[1, :, :, 0] = gate
    return fac


def rydberg_interaction(sequence: pulser.Sequence) -> torch.Tensor:
    """
    Computes the interaction matrix from the qubit positions.
    """
    num_qubits = len(sequence.register.qubit_ids)

    c6 = sequence.device.interaction_coeff

    qubit_positions = get_qubit_positions(sequence.register)
    interaction_matrix = torch.zeros(num_qubits, num_qubits)

    for numi in range(len(qubit_positions)):
        for numj in range(numi + 1, len(qubit_positions)):
            interaction_matrix[numi][numj] = (
                c6 / dist2(qubit_positions[numi], qubit_positions[numj]) ** 3
            )

    return interaction_matrix


def make_H(
    *,
    interaction_matrix: torch.tensor,
    omega: torch.Tensor,
    delta: torch.Tensor,
    phi: torch.Tensor,
    noise: torch.Tensor = torch.zeros(2, 2),
) -> MPO:
    r"""
    Constructs and returns a Matrix Product Operator (MPO) representing the
    neutral atoms Hamiltonian, parameterized by `omega`, `delta`, and `phi`.

    The Hamiltonian H is given by:
    H = ∑ⱼΩⱼ[cos(ϕⱼ)σˣⱼ + sin(ϕⱼ)σʸⱼ] - ∑ⱼΔⱼnⱼ + ∑ᵢ﹥ⱼC⁶/rᵢⱼ⁶ nᵢnⱼ

    If noise is considered, the Hamiltonian includes an additional term to support
    the Monte Carlo WaveFunction algorithm:
    H = ∑ⱼΩⱼ[cos(ϕⱼ)σˣⱼ + sin(ϕⱼ)σʸⱼ] - ∑ⱼΔⱼnⱼ + ∑ᵢ﹥ⱼC⁶/rᵢⱼ⁶ nᵢnⱼ - 0.5i∑ₘ ∑ᵤ Lₘᵘ⁺ Lₘᵘ
    where Lₘᵘ are the Lindblad operators representing the noise, m for noise channel
    and u for the number of atoms

    Args:
        interaction_matrix (torch.Tensor): The interaction matrix describing the interactions
        between qubits.
        omega (torch.Tensor): Rabi frequency Ωⱼ for each qubit.
        delta (torch.Tensor): The detuning value Δⱼ for each qubit.
        phi (torch.Tensor): The phase ϕⱼ corresponding to each qubit.
        noise (torch.Tensor, optional): The single-qubit noise
        term -0.5i∑ⱼLⱼ†Lⱼ applied to all qubits.
        This can be computed using the `compute_noise_from_lindbladians` function.
        Defaults to a zero tensor.

    Returns:
        MPO: A Matrix Product Operator (MPO) representing the specified Hamiltonian.

    Note:
    For more information about the Hamiltonian and its usage, refer to the
    [Pulser documentation](https://pulser.readthedocs.io/en/stable/conventions.html#hamiltonians).

    """
    assert noise.shape == (2, 2)

    nqubits = interaction_matrix.size(dim=1)
    dtype = omega[0].dtype
    device = omega[0].device

    sx = torch.tensor([[0, 0.5], [0.5, 0]], dtype=dtype, device=device)
    sy = torch.tensor([[0, -0.5j], [0.5j, 0]], dtype=dtype, device=device)
    pu = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=dtype, device=device)

    a = torch.tensordot(omega * torch.cos(phi), sx, dims=0)
    c = torch.tensordot(delta, pu, dims=0)
    b = torch.tensordot(omega * torch.sin(phi), sy, dims=0)

    single_qubit_terms = a - b - c + noise

    cores = [_first_factor(single_qubit_terms[0])]

    if nqubits > 2:
        for i in range(1, nqubits // 2):

            cores.append(
                _left_factor(
                    single_qubit_terms[i],
                    [interaction_matrix[j, i] for j in range(i)],
                )
            )

        i = nqubits // 2
        cores.append(
            _middle_factor(
                single_qubit_terms[i],
                [interaction_matrix[j, i] for j in range(i)],
                [interaction_matrix[i, j] for j in range(i + 1, nqubits)],
                [
                    [interaction_matrix[k, j] for j in range(i + 1, nqubits)]
                    for k in range(i)
                ],
            )
        )

        for i in range(nqubits // 2 + 1, nqubits - 1):
            cores.append(
                _right_factor(
                    single_qubit_terms[i],
                    [interaction_matrix[i, j] for j in range(i + 1, nqubits)],
                )
            )

    scale = 1.0
    if nqubits == 2:
        scale = interaction_matrix[0, 1]
    cores.append(
        _last_factor(
            single_qubit_terms[-1],
            scale,
        )
    )
    return MPO(cores)
