import math
from typing import Any

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
) -> Any:

    num_qubits = int(math.log2(len(state.vector)))
    state_tensor = state.vector.reshape((2,) * num_qubits)
    return [(state_tensor.select(i, 1).norm() ** 2).item() for i in range(num_qubits)]


def correlation_matrix_sv_impl(
    self: CorrelationMatrix,
    config: BackendConfig,
    t: int,
    state: StateVector,
    H: Operator,
) -> Any:
    """'Sparse' implementation of <𝜓| nᵢ nⱼ | 𝜓 >"""
    num_qubits = int(math.log2(len(state.vector)))
    state_tensor = state.vector.reshape((2,) * num_qubits)

    correlation_matrix = [[0.0] * num_qubits for _ in range(num_qubits)]

    for numi in range(num_qubits):
        select_i = state_tensor.select(numi, 1)
        for numj in range(numi, num_qubits):  # select the upper triangle
            if numi == numj:
                value = (select_i.norm() ** 2).item()
            else:
                value = (select_i.select(numj - 1, 1).norm() ** 2).item()

            correlation_matrix[numi][numj] = value
            correlation_matrix[numj][numi] = value
    return correlation_matrix


def energy_variance_sv_impl(
    self: EnergyVariance,
    config: BackendConfig,
    t: int,
    state: StateVector,
    H: RydbergHamiltonian,
) -> Any:
    hstate = H * state.vector
    h_squared = torch.vdot(hstate, hstate)
    h_state = torch.vdot(state.vector, hstate)
    return (h_squared.real - h_state.real**2).item()


def second_momentum_sv_impl(
    self: SecondMomentOfEnergy,
    config: BackendConfig,
    t: int,
    state: StateVector,
    H: RydbergHamiltonian,
) -> Any:

    hstate = H * state.vector
    h_squared = torch.vdot(hstate, hstate)
    return (h_squared.real).item()
