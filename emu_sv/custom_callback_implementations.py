import torch

from pulser.backend import (
    CorrelationMatrix,
    EmulationConfig,
    EnergySecondMoment,
    EnergyVariance,
    Occupation,
    Energy,
)

from emu_sv.state_vector import StateVector
from emu_sv.dense_operator import DenseOperator
from emu_sv.hamiltonian import RydbergHamiltonian


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
    occupation = torch.zeros(nqubits, dtype=torch.float64, device=state.vector.device)
    for i in range(nqubits):
        state_tensor = state.vector.view(2**i, 2, -1)
        # nᵢ is a projector and therefore nᵢ == nᵢnᵢ
        # ❬ψ|nᵢ|ψ❭ == ❬ψ|nᵢnᵢ|ψ❭ == ❬ψ|nᵢ * nᵢ|ψ❭ == ❬ϕ|ϕ❭ == |ϕ|**2
        occupation[i] = torch.linalg.vector_norm(state_tensor[:, 1]) ** 2
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
    correlation = torch.zeros(
        nqubits, nqubits, dtype=torch.float64, device=state.vector.device
    )

    for i in range(nqubits):
        select_i = state.vector.view(2**i, 2, -1)
        select_i = select_i[:, 1]
        for j in range(i, nqubits):  # select the upper triangle
            if i == j:
                value = torch.linalg.vector_norm(select_i) ** 2
                correlation[j, j] = value
            else:
                select_i = select_i.view(2**i, 2 ** (j - i - 1), 2, -1)
                select_ij = select_i[:, :, 1, :]
                value = torch.linalg.vector_norm(select_ij) ** 2
                correlation[i, j] = value
                correlation[j, i] = value

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


def energy_sv_impl(
    self: Energy,
    *,
    config: EmulationConfig,
    state: StateVector,
    hamiltonian: RydbergHamiltonian,
) -> torch.Tensor:
    """
    Custom implementation of the energy ❬ψ|H|ψ❭ for the state vector solver.
    """
    en: torch.Tensor = hamiltonian.expect(state)
    return en
