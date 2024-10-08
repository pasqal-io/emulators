"""
This file deals with creation of the MPO corresponding
to the Hamiltonian of a neutral atoms quantum processor.
"""

from emu_mps.pulser_adapter import HamiltonianType
import torch

from emu_mps.mpo import MPO

dtype = torch.complex128  # always complex128
iden_op = torch.eye(2, 2, dtype=dtype)  # dtype is always complex128
n_op = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=dtype)
creation_op = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=dtype)
sx = torch.tensor([[0.0, 0.5], [0.5, 0.0]], dtype=dtype)
sy = torch.tensor([[0.0, -0.5j], [0.5j, 0.0]], dtype=dtype)
pu = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=dtype)


def _first_factor_rydberg(gate: torch.Tensor) -> torch.Tensor:
    """
    Creates the first Ising Hamiltonian factor.
    """
    fac = torch.zeros(1, 2, 2, 3, dtype=dtype)
    fac[0, :, :, 1] = iden_op
    fac[0, :, :, 2] = n_op  # number operator

    fac[0, :, :, 0] = gate
    return fac


def _first_factor_xy(gate: torch.Tensor) -> torch.Tensor:
    """
    Creates the first XY Hamiltonian factor.
    """
    fac = torch.zeros(1, 2, 2, 4, dtype=dtype)
    fac[0, :, :, 1] = iden_op
    fac[0, :, :, 2] = creation_op
    fac[0, :, :, 3] = creation_op.T

    fac[0, :, :, 0] = gate
    return fac


def _last_factor_rydberg(gate: torch.Tensor, scale: float | complex) -> torch.Tensor:
    """
    Creates the last Ising Hamiltonian factor.
    """
    fac = torch.zeros(3, 2, 2, 1, dtype=dtype)
    fac[0, :, :, 0] = iden_op
    fac[2, :, :, 0] = scale * n_op

    fac[1, :, :, 0] = gate
    return fac


def _last_factor_xy(gate: torch.Tensor, scale: float | complex) -> torch.Tensor:
    """
    Creates the last XY Hamiltonian factor.
    """
    fac = torch.zeros(4, 2, 2, 1, dtype=dtype)
    fac[0, :, :, 0] = iden_op
    fac[2, :, :, 0] = scale * creation_op.T
    fac[3, :, :, 0] = scale * creation_op

    fac[1, :, :, 0] = gate
    return fac


def _left_factor_rydberg(gate: torch.Tensor, scales: list[float]) -> torch.Tensor:
    """
    Creates the Ising Hamiltonian factors in the left half of the MPS, excepted the first factor.
    """
    index = len(scales)
    fac = torch.zeros(index + 2, 2, 2, index + 3, dtype=dtype)
    for i, val in enumerate(scales):
        fac[i + 2, :, :, 0] = val * n_op  # interaction with previous qubits
    fac[1, :, :, index + 2] = n_op  # interaction with next qubits
    for i in range(index + 2):
        fac[i, :, :, i] = iden_op  # identity matrix to carry the gates of other qubits

    fac[1, :, :, 0] = gate
    return fac


def _left_factor_xy(gate: torch.Tensor, scales: list[float]) -> torch.Tensor:
    """
    Creates the XY Hamiltonian factors in the left half of the MPS, excepted the first factor.
    """
    index = len(scales)
    fac = torch.zeros(2 * index + 2, 2, 2, 2 * index + 4, dtype=dtype)

    for i, val in enumerate(scales):
        fac[2 * i + 2, :, :, 0] = val * creation_op.T  # sigma-
        fac[2 * i + 3, :, :, 0] = val * creation_op  # sigma+
    fac[1, :, :, -2] = creation_op
    fac[1, :, :, -1] = creation_op.T
    for i in range(2 * index + 2):
        fac[i, :, :, i] = iden_op  # identity to carry the gates of other qubits

    fac[1, :, :, 0] = gate
    return fac


def _right_factor_rydberg(gate: torch.Tensor, scales: list[float]) -> torch.Tensor:
    """
    Creates the Ising Hamiltonian factors in the right half of the MPS, excepted the last factor.
    """
    index = len(scales)
    fac = torch.zeros(index + 3, 2, 2, index + 2, dtype=dtype)
    for i, val in enumerate(scales):
        fac[1, :, :, i + 2] = val * n_op  # XY interaction with previous qubits
    fac[2, :, :, 0] = n_op  # XY interaction with next qubits
    for i in range(2, index + 2):
        fac[i + 1, :, :, i] = iden_op
    fac[0, :, :, 0] = iden_op  # identity to carry the next gates to the previous qubits
    fac[1, :, :, 1] = iden_op  # identity to carry previous gates to next qubits

    fac[1, :, :, 0] = gate
    return fac


def _right_factor_xy(gate: torch.Tensor, scales: list[float]) -> torch.Tensor:
    """
    Creates the XY Hamiltonian factors in the right half of the MPS, excepted the last factor.
    """
    index = len(scales)
    fac = torch.zeros(2 * index + 4, 2, 2, 2 * index + 2, dtype=dtype)
    for i, val in enumerate(scales):
        fac[1, :, :, 2 * i + 2] = val * creation_op  # XY interaction with previous qubits
        fac[1, :, :, 2 * i + 3] = val * creation_op.T
    fac[2, :, :, 0] = creation_op.T  # s- with next qubits
    fac[3, :, :, 0] = creation_op  # s+ with next qubits
    for i in range(2, index + 2):
        fac[2 * i, :, :, 2 * i - 2] = iden_op
        fac[2 * i + 1, :, :, 2 * i - 1] = iden_op

    # identity to carry the next gates to the previous qubits
    fac[0, :, :, 0] = iden_op
    # identity to carry previous gates to next qubits
    fac[1, :, :, 1] = iden_op

    fac[1, :, :, 0] = gate
    return fac


def _middle_factor_rydberg(
    gate: torch.Tensor,
    scales_l: list[float],
    scales_r: list[float],
    scales_mat: list[list[float]],
) -> torch.Tensor:
    """
    Creates the Ising Hamiltonian factor at index ⌊n/2⌋ of the n-qubit MPO.
    """
    assert len(scales_mat) == len(scales_l)
    assert all(len(x) == len(scales_r) for x in scales_mat)

    fac = torch.zeros(len(scales_l) + 2, 2, 2, len(scales_r) + 2, dtype=dtype)
    for i, val in enumerate(scales_r):
        fac[1, :, :, i + 2] = val * n_op  # rydberg interaction with previous qubits
    for i, val in enumerate(scales_l):
        fac[i + 2, :, :, 0] = val * n_op  # rydberg interaction with next qubits
    for i, row in enumerate(scales_mat):
        for j, val in enumerate(row):
            fac[i + 2, :, :, j + 2] = (
                val * iden_op
            )  # rydberg interaction of previous with next qubits
    fac[0, :, :, 0] = iden_op  # identity to carry the next gates to the previous qubits
    fac[1, :, :, 1] = iden_op  # identity to carry previous gates to next qubits

    fac[1, :, :, 0] = gate
    return fac


def _middle_factor_xy(
    gate: torch.Tensor,
    scales_l: list[float],
    scales_r: list[float],
    scales_mat: list[list[float]],
) -> torch.Tensor:
    """
    Creates the XY Hamiltonian factor at index ⌊n/2⌋ of the n-qubit MPO.
    """
    assert len(scales_mat) == len(scales_l)
    assert all(len(x) == len(scales_r) for x in scales_mat)

    fac = torch.zeros(2 * len(scales_l) + 2, 2, 2, 2 * len(scales_r) + 2, dtype=dtype)
    for i, val in enumerate(scales_r):
        fac[1, :, :, 2 * i + 2] = val * creation_op  # XY interaction with previous qubits
        fac[1, :, :, 2 * i + 3] = (
            val * creation_op.T
        )  # XY interaction with previous qubits
    for i, val in enumerate(scales_l):
        fac[2 * i + 2, :, :, 0] = val * creation_op.T  # XY interaction with next qubits
        fac[2 * i + 3, :, :, 0] = val * creation_op  # XY interaction with next qubits
    for i, row in enumerate(scales_mat):
        for j, val in enumerate(row):
            fac[2 * i + 2, :, :, 2 * j + 2] = (
                val * iden_op
            )  # XY interaction of previous with next qubits
            fac[2 * i + 3, :, :, 2 * j + 3] = (
                val * iden_op
            )  # XY interaction of previous with next qubits
    fac[0, :, :, 0] = iden_op  # identity to carry the next gates to the previous qubits
    fac[1, :, :, 1] = iden_op  # identity to carry previous gates to next qubits

    fac[1, :, :, 0] = gate
    return fac


def make_H(
    *,
    interaction_matrix: torch.tensor,  # depends on Hamiltonian Type
    omega: torch.Tensor,
    delta: torch.Tensor,
    phi: torch.Tensor,
    hamiltonian_type: HamiltonianType,
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

    a = torch.tensordot(omega * torch.cos(phi), sx, dims=0)
    c = torch.tensordot(delta, pu, dims=0)
    b = torch.tensordot(omega * torch.sin(phi), sy, dims=0)

    single_qubit_terms = a + b - c + noise

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
