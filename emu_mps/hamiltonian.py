"""
This file deals with creation of the MPO corresponding
to the Hamiltonian of a neutral atoms quantum processor.
"""

from emu_base import HamiltonianType
import torch

from emu_mps.mpo import MPO

dtype = torch.complex128  # always complex128
iden_op = torch.eye(2, 2, dtype=dtype)  # dtype is always complex128
n_op = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=dtype)
creation_op = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=dtype)
sx = torch.tensor([[0.0, 0.5], [0.5, 0.0]], dtype=dtype)
sy = torch.tensor([[0.0, -0.5j], [0.5j, 0.0]], dtype=dtype)
pu = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=dtype)


def truncate_factor(
    factor: torch.Tensor,
    left_interactions: torch.Tensor,
    right_interactions: torch.Tensor,
    hamiltonian_type: HamiltonianType,
) -> torch.Tensor:
    if hamiltonian_type == HamiltonianType.XY:
        left_interactions = torch.stack(
            (left_interactions, left_interactions), dim=-1
        ).reshape(-1)
        right_interactions = torch.stack(
            (right_interactions, right_interactions), dim=-1
        ).reshape(-1)
    padding = torch.tensor([True] * 2)
    trunc = factor[torch.cat((padding, left_interactions))]
    return trunc[:, :, :, torch.cat((padding, right_interactions))]


def _first_factor_rydberg(interaction: bool) -> torch.Tensor:
    """
    Creates the first Ising Hamiltonian factor.
    """
    fac = torch.zeros(1, 2, 2, 3 if interaction else 2, dtype=dtype)
    fac[0, :, :, 1] = iden_op
    if interaction:
        fac[0, :, :, 2] = n_op  # number operator

    return fac


def _first_factor_xy(interaction: bool) -> torch.Tensor:
    """
    Creates the first XY Hamiltonian factor.
    """
    fac = torch.zeros(1, 2, 2, 4 if interaction else 2, dtype=dtype)
    fac[0, :, :, 1] = iden_op
    if interaction:
        fac[0, :, :, 2] = creation_op
        fac[0, :, :, 3] = creation_op.T

    return fac


def _last_factor_rydberg(scale: float | complex) -> torch.Tensor:
    """
    Creates the last Ising Hamiltonian factor.
    """
    fac = torch.zeros(3 if scale != 0.0 else 2, 2, 2, 1, dtype=dtype)
    fac[0, :, :, 0] = iden_op
    if scale != 0:
        fac[2, :, :, 0] = scale * n_op

    return fac


def _last_factor_xy(scale: float | complex) -> torch.Tensor:
    """
    Creates the last XY Hamiltonian factor.
    """
    fac = torch.zeros(4 if scale != 0.0 else 2, 2, 2, 1, dtype=dtype)
    fac[0, :, :, 0] = iden_op
    if scale != 0:
        fac[2, :, :, 0] = scale * creation_op.T
        fac[3, :, :, 0] = scale * creation_op

    return fac


def _left_factor_rydberg(
    scales: torch.Tensor,
    left_interactions: torch.Tensor,
    right_interactions: torch.Tensor,
) -> torch.Tensor:
    """
    Creates the Ising Hamiltonian factors in the left half of the MPS, excepted the first factor.
    """
    index = len(scales)
    fac = torch.zeros(index + 2, 2, 2, index + 3, dtype=dtype)
    fac[2 : scales.shape[0] + 2, :, :, 0] = (
        scales.reshape(-1, 1, 1) * n_op
    )  # interaction with previous qubits
    fac[1, :, :, index + 2] = n_op  # interaction with next qubits
    for i in range(index + 2):
        fac[i, :, :, i] = iden_op  # identity matrix to carry the gates of other qubits

    return truncate_factor(
        fac,
        left_interactions,
        right_interactions,
        hamiltonian_type=HamiltonianType.Rydberg,
    )


def _left_factor_xy(
    scales: torch.Tensor,
    left_interactions: torch.Tensor,
    right_interactions: torch.Tensor,
) -> torch.Tensor:
    """
    Creates the XY Hamiltonian factors in the left half of the MPS, excepted the first factor.
    """
    index = len(scales)
    fac = torch.zeros(2 * index + 2, 2, 2, 2 * index + 4, dtype=dtype)

    fac[2 : 2 * scales.shape[0] + 2 : 2, :, :, 0] = (
        scales.reshape(-1, 1, 1) * creation_op.T
    )  # sigma-
    fac[3 : 2 * scales.shape[0] + 3 : 2, :, :, 0] = (
        scales.reshape(-1, 1, 1) * creation_op
    )  # sigma+
    fac[1, :, :, -2] = creation_op
    fac[1, :, :, -1] = creation_op.T
    for i in range(2 * index + 2):
        fac[i, :, :, i] = iden_op  # identity to carry the gates of other qubits

    # duplicate each bool, because each interaction term occurs twice
    return truncate_factor(
        fac, left_interactions, right_interactions, hamiltonian_type=HamiltonianType.XY
    )


def _right_factor_rydberg(
    scales: torch.Tensor,
    left_interactions: torch.Tensor,
    right_interactions: torch.Tensor,
) -> torch.Tensor:
    """
    Creates the Ising Hamiltonian factors in the right half of the MPS, excepted the last factor.
    """
    index = len(scales)
    fac = torch.zeros(index + 3, 2, 2, index + 2, dtype=dtype)
    fac[1, :, :, 2 : scales.shape[0] + 2] = scales * n_op.reshape(
        2, 2, 1
    )  # XY interaction with previous qubits
    fac[2, :, :, 0] = n_op  # XY interaction with next qubits
    for i in range(2, index + 2):
        fac[i + 1, :, :, i] = iden_op
    fac[0, :, :, 0] = iden_op  # identity to carry the next gates to the previous qubits
    fac[1, :, :, 1] = iden_op  # identity to carry previous gates to next qubits

    return truncate_factor(
        fac,
        left_interactions,
        right_interactions,
        hamiltonian_type=HamiltonianType.Rydberg,
    )


def _right_factor_xy(
    scales: torch.Tensor,
    left_interactions: torch.Tensor,
    right_interactions: torch.Tensor,
) -> torch.Tensor:
    """
    Creates the XY Hamiltonian factors in the right half of the MPS, excepted the last factor.
    """
    index = len(scales)
    fac = torch.zeros(2 * index + 4, 2, 2, 2 * index + 2, dtype=dtype)
    fac[1, :, :, 2 : 2 * scales.shape[0] + 2 : 2] = scales * creation_op.reshape(
        2, 2, 1
    )  # XY interaction with previous qubits
    fac[1, :, :, 3 : 2 * scales.shape[0] + 3 : 2] = scales * creation_op.T.reshape(
        2, 2, 1
    )
    fac[2, :, :, 0] = creation_op.T  # s- with next qubits
    fac[3, :, :, 0] = creation_op  # s+ with next qubits
    for i in range(2, index + 2):
        fac[2 * i, :, :, 2 * i - 2] = iden_op
        fac[2 * i + 1, :, :, 2 * i - 1] = iden_op

    # identity to carry the next gates to the previous qubits
    fac[0, :, :, 0] = iden_op
    # identity to carry previous gates to next qubits
    fac[1, :, :, 1] = iden_op

    # duplicate each bool, because each interaction term occurs twice
    return truncate_factor(
        fac, left_interactions, right_interactions, hamiltonian_type=HamiltonianType.XY
    )


def _middle_factor_rydberg(
    scales_l: torch.Tensor,
    scales_r: torch.Tensor,
    scales_mat: torch.Tensor,
    left_interactions: torch.Tensor,
    right_interactions: torch.Tensor,
) -> torch.Tensor:
    """
    Creates the Ising Hamiltonian factor at index ⌊n/2⌋ of the n-qubit MPO.
    """
    assert len(scales_mat) == len(scales_l)
    assert all(len(x) == len(scales_r) for x in scales_mat)

    fac = torch.zeros(len(scales_l) + 2, 2, 2, len(scales_r) + 2, dtype=dtype)
    fac[1, :, :, 2 : scales_r.shape[0] + 2] = scales_r * n_op.reshape(
        2, 2, 1
    )  # rydberg interaction with previous qubits
    fac[2 : scales_l.shape[0] + 2, :, :, 0] = (
        scales_l.reshape(-1, 1, 1) * n_op
    )  # rydberg interaction with next qubits
    x_shape, y_shape = scales_mat.shape
    fac[2 : x_shape + 2, :, :, 2 : y_shape + 2] = scales_mat.reshape(
        x_shape, 1, 1, y_shape
    ) * iden_op.reshape(
        1, 2, 2, 1
    )  # rydberg interaction of previous with next qubits
    fac[0, :, :, 0] = iden_op  # identity to carry the next gates to the previous qubits
    fac[1, :, :, 1] = iden_op  # identity to carry previous gates to next qubits

    return truncate_factor(
        fac,
        left_interactions,
        right_interactions,
        hamiltonian_type=HamiltonianType.Rydberg,
    )


def _middle_factor_xy(
    scales_l: torch.Tensor,
    scales_r: torch.Tensor,
    scales_mat: torch.Tensor,
    left_interactions: torch.Tensor,
    right_interactions: torch.Tensor,
) -> torch.Tensor:
    """
    Creates the XY Hamiltonian factor at index ⌊n/2⌋ of the n-qubit MPO.
    """
    assert len(scales_mat) == len(scales_l)
    assert all(len(x) == len(scales_r) for x in scales_mat)

    fac = torch.zeros(2 * len(scales_l) + 2, 2, 2, 2 * len(scales_r) + 2, dtype=dtype)
    fac[1, :, :, 2 : 2 * scales_r.shape[0] + 2 : 2] = scales_r * creation_op.reshape(
        2, 2, 1
    )  # XY interaction with previous qubits
    fac[1, :, :, 3 : 2 * scales_r.shape[0] + 3 : 2] = scales_r * creation_op.T.reshape(
        2, 2, 1
    )  # XY interaction with previous qubits
    fac[2 : 2 * scales_l.shape[0] + 2 : 2, :, :, 0] = (
        scales_l.reshape(-1, 1, 1) * creation_op.T
    )  # XY interaction with next qubits
    fac[3 : 2 * scales_l.shape[0] + 3 : 2, :, :, 0] = (
        scales_l.reshape(-1, 1, 1) * creation_op
    )  # XY interaction with next qubits
    x_shape, y_shape = scales_mat.shape
    fac[2 : 2 * x_shape + 2 : 2, :, :, 2 : 2 * y_shape + 2 : 2] = scales_mat.reshape(
        x_shape, 1, 1, y_shape
    ) * iden_op.reshape(
        1, 2, 2, 1
    )  # XY interaction of previous with next qubits
    fac[3 : 2 * x_shape + 3 : 2, :, :, 3 : 2 * y_shape + 3 : 2] = scales_mat.reshape(
        x_shape, 1, 1, y_shape
    ) * iden_op.reshape(
        1, 2, 2, 1
    )  # XY interaction of previous with next qubits
    fac[0, :, :, 0] = iden_op  # identity to carry the next gates to the previous qubits
    fac[1, :, :, 1] = iden_op  # identity to carry previous gates to next qubits

    return truncate_factor(
        fac, left_interactions, right_interactions, hamiltonian_type=HamiltonianType.XY
    )


def _get_interactions_to_keep(interaction_matrix: torch.Tensor) -> list[torch.Tensor]:
    """
    returns a list of bool valued tensors,
    indicating which interaction terms to keep for each bond in the MPO
    """
    nqubits = interaction_matrix.size(dim=1)
    middle = nqubits // 2
    interaction_matrix += torch.eye(
        nqubits, nqubits, dtype=interaction_matrix.dtype
    )  # below line fails on all zeros
    interaction_boundaries = torch.tensor(
        [torch.max(torch.nonzero(interaction_matrix[i])) for i in range(middle)]
    )
    interactions_to_keep = [interaction_boundaries[: i + 1] > i for i in range(middle)]

    interaction_boundaries = torch.tensor(
        [
            torch.min(torch.nonzero(interaction_matrix[j]))
            for j in range(middle + 1, nqubits)
        ]
    )
    interactions_to_keep += [
        interaction_boundaries[i - middle :] <= i for i in range(middle, nqubits - 1)
    ]
    return interactions_to_keep


def make_H(
    *,
    interaction_matrix: torch.Tensor,  # depends on Hamiltonian Type
    hamiltonian_type: HamiltonianType,
    num_gpus_to_use: int | None,
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

    make_H constructs an MPO of the appropriate size, but the single qubit terms are left at zero.
    To fill in the appropriate values, call update_H

    Args:
        interaction_matrix (torch.Tensor): The interaction matrix describing the interactions
        between qubits.
        num_gpus_to_use (int): how many gpus to put the Hamiltonian on. See utils.assign_devices
    Returns:
        MPO: A Matrix Product Operator (MPO) representing the specified Hamiltonian.

    Note:
    For more information about the Hamiltonian and its usage, refer to the
    [Pulser documentation](https://pulser.readthedocs.io/en/stable/conventions.html#hamiltonians).

    """

    if hamiltonian_type == HamiltonianType.Rydberg:
        _first_factor = _first_factor_rydberg
        _last_factor = _last_factor_rydberg
        _left_factor = _left_factor_rydberg
        _right_factor = _right_factor_rydberg
        _middle_factor = _middle_factor_rydberg
    elif hamiltonian_type == HamiltonianType.XY:
        _first_factor = _first_factor_xy
        _last_factor = _last_factor_xy
        _left_factor = _left_factor_xy
        _right_factor = _right_factor_xy
        _middle_factor = _middle_factor_xy
    else:
        raise ValueError(f"Unsupported hamiltonian type {hamiltonian_type}")

    nqubits = interaction_matrix.size(dim=1)
    middle = nqubits // 2

    interactions_to_keep = _get_interactions_to_keep(interaction_matrix)

    cores = [_first_factor(interactions_to_keep[0].item())]

    if nqubits > 2:
        for i in range(1, middle):
            cores.append(
                _left_factor(
                    interaction_matrix[:i, i],
                    left_interactions=interactions_to_keep[i - 1],
                    right_interactions=interactions_to_keep[i],
                )
            )

        i = middle
        cores.append(
            _middle_factor(
                interaction_matrix[:i, i],
                interaction_matrix[i, i + 1 :],
                interaction_matrix[:i, i + 1 :],
                interactions_to_keep[i - 1],
                interactions_to_keep[i],
            )
        )

        for i in range(middle + 1, nqubits - 1):
            cores.append(
                _right_factor(
                    interaction_matrix[i, i + 1 :],
                    interactions_to_keep[i - 1],
                    interactions_to_keep[i],
                )
            )
    if nqubits == 2:
        scale = interaction_matrix[0, 1]
    elif interactions_to_keep[-1][0]:
        scale = 1.0
    else:
        scale = 0.0
    cores.append(
        _last_factor(
            scale,
        )
    )
    return MPO(cores, num_gpus_to_use=num_gpus_to_use)


def update_H(
    hamiltonian: MPO,
    omega: torch.Tensor,
    delta: torch.Tensor,
    phi: torch.Tensor,
    noise: torch.Tensor = torch.zeros(2, 2),
) -> None:
    """
    The single qubit operators in the Hamiltonian,
    corresponding to the omega, delta, phi parameters and the aggregated Lindblad operators
    have a well-determined position in the factors of the Hamiltonian.
    This function updates this part of the factors to update the
    Hamiltonian with new parameters without rebuilding the entire thing.
    See make_H for details about the Hamiltonian.

    This is an in-place operation, so this function returns nothing.

    Args:
        omega (torch.Tensor): Rabi frequency Ωⱼ for each qubit.
        delta (torch.Tensor): The detuning value Δⱼ for each qubit.
        phi (torch.Tensor): The phase ϕⱼ corresponding to each qubit.
        noise (torch.Tensor, optional): The single-qubit noise
        term -0.5i∑ⱼLⱼ†Lⱼ applied to all qubits.
        This can be computed using the `compute_noise_from_lindbladians` function.
        Defaults to a zero tensor.
    """

    assert noise.shape == (2, 2)
    nqubits = omega.size(dim=0)

    a = torch.tensordot(omega * torch.cos(phi), sx, dims=0)
    c = torch.tensordot(delta, pu, dims=0)
    b = torch.tensordot(omega * torch.sin(phi), sy, dims=0)

    single_qubit_terms = a + b - c + noise
    factors = hamiltonian.factors

    factors[0][0, :, :, 0] = single_qubit_terms[0]
    for i in range(1, nqubits):
        factors[i][1, :, :, 0] = single_qubit_terms[i]
