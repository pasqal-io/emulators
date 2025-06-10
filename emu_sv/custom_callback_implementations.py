import torch

from pulser.backend import (
    CorrelationMatrix,
    EmulationConfig,
    EnergySecondMoment,
    EnergyVariance,
    Occupation,
)
from emu_sv.density_matrix_state import DensityMatrix
from emu_sv.state_vector import StateVector
from emu_sv.dense_operator import DenseOperator
from emu_sv.hamiltonian import RydbergHamiltonian
from emu_sv.lindblad_operator import RydbergLindbladian

dtype = torch.float64


def qubit_occupation_sv_impl(
    self: Occupation,
    *,
    config: EmulationConfig,
    state: StateVector,
    hamiltonian: DenseOperator,
) -> torch.Tensor:
    """
    Custom implementation of the occupation ❬ψ|nᵢ|ψ❭ for the state vector solver.
    """
    nqubits = state.n_qudits
    occupation = torch.zeros(nqubits, dtype=dtype, device=state.vector.device)
    for i in range(nqubits):
        state_tensor = state.vector.view(2**i, 2, -1)
        # nᵢ is a projector and therefore nᵢ == nᵢnᵢ
        # ❬ψ|nᵢ|ψ❭ == ❬ψ|nᵢnᵢ|ψ❭ == ❬ψ|nᵢ * nᵢ|ψ❭ == ❬ϕ|ϕ❭ == |ϕ|**2
        occupation[i] = torch.linalg.vector_norm(state_tensor[:, 1]) ** 2
    return occupation.cpu()


def qubit_occupation_sv_den_mat_impl(
    self: Occupation,
    *,
    config: EmulationConfig,
    state: DensityMatrix,
    hamiltonian: DenseOperator,
) -> torch.Tensor:
    """
    Custom implementation of the occupation nᵢ observable for density matrix.
    The observable nᵢ is given by: I ⊗ ... ⊗  nᵢ ⊗ ...  ⊗I
    where nᵢ is the occupation operator for qubit i.
    The expectation value is given by: <nᵢ> = Tr(ρ nᵢ).

    The output will be a tensor of size (nqubits,), where each element will be the
    expectation value of the occupation operator for each qubit.
    In case of 3 atoms, the output will be a tensor of size (3,), where each element
    will be <nᵢ> = Tr(ρnᵢ), or [ <n₁>, <n₂>, <n₃> ].
    """
    nqubits = state.n_qudits
    occupation = torch.zeros(nqubits, dtype=dtype, device=state.matrix.device)
    diag_state_tensor = state.matrix.diagonal()
    for i in range(nqubits):
        state_tensor = diag_state_tensor.view(2**i, 2, 2 ** (nqubits - i - 1))[:, 1, :]
        occupation[i] = state_tensor.sum().real
    return occupation.cpu()


def correlation_matrix_sv_impl(
    self: CorrelationMatrix,
    *,
    config: EmulationConfig,
    state: StateVector,
    hamiltonian: DenseOperator,
) -> torch.Tensor:
    """
    Custom implementation of the density-density correlation ❬ψ|nᵢnⱼ|ψ❭
      for the state vector solver.
    TODO: extend to arbitrary two-point correlation ❬ψ|AᵢBⱼ|ψ❭
    """
    nqubits = state.n_qudits
    correlation = torch.zeros(nqubits, nqubits, dtype=dtype, device=state.vector.device)

    for i in range(nqubits):
        select_i = state.vector.view(2**i, 2, -1)
        select_i = select_i[:, 1]
        correlation[i, i] = torch.linalg.vector_norm(select_i) ** 2
        for j in range(i + 1, nqubits):  # select the upper triangle
            select_i = select_i.view(2**i, 2 ** (j - i - 1), 2, -1)
            select_ij = select_i[:, :, 1, :]
            correlation[i, j] = torch.linalg.vector_norm(select_ij) ** 2
            correlation[j, i] = correlation[i, j]

    return correlation.cpu()


def correlation_matrix_sv_den_mat_impl(
    self: CorrelationMatrix,
    *,
    config: EmulationConfig,
    state: DensityMatrix,
    hamiltonian: DenseOperator,
) -> torch.Tensor:
    """
    Custom implementation of the density-density correlation <nᵢnⱼ> = Tr(ρ nᵢnⱼ)
    in the case of Lindblad noise
    """
    nqubits = state.n_qudits
    correlation = torch.zeros(nqubits, nqubits, dtype=dtype)
    state_diag_matrix = state.matrix.diagonal()
    for i in range(nqubits):  # applying ni
        shapei = (2**i, 2, 2 ** (nqubits - i - 1))
        state_diag_ni = state_diag_matrix.view(*shapei)[:, 1, :]
        correlation[i, i] = state_diag_ni.sum().real  # diagonal
        for j in range(i + 1, nqubits):
            shapeij = (2**i, 2 ** (j - i - 1), 2, 2 ** (nqubits - 1 - j))
            state_diag_ni_nj = state_diag_ni.view(*shapeij)[:, :, 1, :]
            correlation[i, j] = state_diag_ni_nj.sum().real
            correlation[j, i] = correlation[i, j]
    return correlation.cpu()


def energy_variance_sv_impl(
    self: EnergyVariance,
    *,
    config: EmulationConfig,
    state: StateVector,
    hamiltonian: RydbergHamiltonian,
) -> torch.Tensor:
    """
    Custom implementation of the energy variance ❬ψ|H²|ψ❭-❬ψ|H|ψ❭² for the state vector solver.
    """
    hstate = hamiltonian * state.vector
    h_squared = torch.vdot(hstate, hstate).real
    energy = torch.vdot(state.vector, hstate).real
    en_var: torch.Tensor = h_squared - energy**2
    return en_var.cpu()


def energy_variance_sv_den_mat_impl(
    self: EnergyVariance,
    *,
    config: EmulationConfig,
    state: DensityMatrix,
    hamiltonian: RydbergLindbladian,
) -> torch.Tensor:
    """
    Custom implementation of the energy variance tr(ρH²)-tr(ρH)² for the
    lindblad equation solver.
    """
    h_dense_matrix = hamiltonian.h_eff(state.matrix)  # Hρ
    gpu = state.matrix.is_cuda
    h_squared_dense_mat = hamiltonian.expect(
        DensityMatrix(h_dense_matrix, gpu=gpu)
    )  # tr(ρH²)
    en_var: torch.Tensor = h_squared_dense_mat - hamiltonian.expect(state) ** 2  # tr(ρH)²
    return en_var.cpu()


def energy_second_moment_sv_impl(
    self: EnergySecondMoment,
    *,
    config: EmulationConfig,
    state: StateVector,
    hamiltonian: RydbergHamiltonian,
) -> torch.Tensor:
    """
    Custom implementation of the second moment of energy ❬ψ|H²|ψ❭
    for the state vector solver.
    """
    hstate = hamiltonian * state.vector
    en_2_mom: torch.Tensor = torch.vdot(hstate, hstate).real
    return en_2_mom.cpu()


def energy_second_moment_den_mat_impl(
    self: EnergyVariance,
    *,
    config: EmulationConfig,
    state: DensityMatrix,
    hamiltonian: RydbergLindbladian,
) -> torch.Tensor:
    """
    Custom implementation of the second moment of energy tr(ρH²) for the
    lindblad equation solver.
    """
    h_dense_matrix = hamiltonian.h_eff(state.matrix)  # Hρ
    gpu = state.matrix.is_cuda

    return hamiltonian.expect(
        DensityMatrix(h_dense_matrix, gpu=gpu)
    ).cpu()  # tr(ρH²) = tr(ρ H H)
