import math
import torch

from emu_base.base_classes.config import BackendConfig
from emu_base.base_classes.default_callbacks import (
    QubitDensity,
    EnergyVariance,
    SecondMomentOfEnergy,
    CorrelationMatrix,
)
from emu_base.base_classes.operator import Operator

from emu_sv import StateVector
from emu_sv.hamiltonian import RydbergHamiltonian


def qubit_density_sv_impl(
    self: QubitDensity, config: BackendConfig, t: int, state: StateVector, H: Operator
) -> torch.Tensor:
    """
    Custom implementation of the qubit density ❬ψ|nᵢ|ψ❭ for the state vector solver.
    """
    nqubits = int(math.log2(len(state.vector)))
    state_tensor = state.vector.reshape((2,) * nqubits)

    qubit_density = torch.zeros(nqubits, dtype=torch.float64, device=state_tensor.device)
    for i in range(nqubits):
        qubit_density[i] = state_tensor.select(i, 1).norm() ** 2
    return qubit_density


def correlation_matrix_sv_impl(
    self: CorrelationMatrix,
    config: BackendConfig,
    t: int,
    state: StateVector,
    H: Operator,
) -> torch.Tensor:
    """
    Custom implementation of the density-density correlation ❬ψ|nᵢnⱼ|ψ❭ for the state vector solver.

    TODO: extend to arbitrary two-point correlation ❬ψ|AᵢBⱼ|ψ❭
    """
    nqubits = int(math.log2(len(state.vector)))
    state_tensor = state.vector.reshape((2,) * nqubits)

    correlation = torch.zeros(
        nqubits, nqubits, dtype=torch.float64, device=state_tensor.device
    )

    for i in range(nqubits):
        select_i = state_tensor.select(i, 1)
        for j in range(i, nqubits):  # select the upper triangle
            if i == j:
                value = select_i.norm() ** 2
            else:
                value = select_i.select(j - 1, 1).norm() ** 2

            correlation[i, j] = value
            correlation[j, i] = value
    return correlation


def energy_variance_sv_impl(
    self: EnergyVariance,
    config: BackendConfig,
    t: int,
    state: StateVector,
    H: RydbergHamiltonian,
) -> torch.Tensor:
    """
    Custom implementation of the energy variance ❬ψ|H²|ψ❭-❬ψ|H|ψ❭² for the state vector solver.
    """
    hstate = H * state.vector
    h_squared = torch.vdot(hstate, hstate).real
    energy = torch.vdot(state.vector, hstate).real
    energy_variance: torch.Tensor = h_squared - energy**2
    return energy_variance


def second_moment_sv_impl(
    self: SecondMomentOfEnergy,
    config: BackendConfig,
    t: int,
    state: StateVector,
    H: RydbergHamiltonian,
) -> torch.Tensor:
    """
    Custom implementation of the second moment of energy ❬ψ|H²|ψ❭
    for the state vector solver.
    """
    hstate = H * state.vector
    return torch.vdot(hstate, hstate).real
