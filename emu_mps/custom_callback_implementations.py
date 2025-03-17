import math
import torch

from pulser.backend.default_observables import (
    CorrelationMatrix,
    EnergySecondMoment,
    EnergyVariance,
    Occupation,
    Energy
)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from emu_mps.mps_config import MPSConfig
    from emu_mps.mps import MPS
    from emu_mps.mpo import MPO



def qubit_occupation_mps_impl(
    self: Occupation,
    *,
    config:'MPSConfig',
    state: 'MPS',
    H: 'MPO',
) -> torch.Tensor:
    """
    Custom implementation of the qubit density ❬ψ|nᵢ|ψ❭ for the state vector solver.
    """
    op = torch.tensor([[[0., 1.],[0., 1.]]], dtype=torch.complex128, device = state.factors[0].device)
    return state.expect_batch(op).real.reshape(-1).to("cpu")

def correlation_matrix_mps_impl(
    self: CorrelationMatrix,
    *,
    config:'MPSConfig',
    state: 'MPS',
    H: 'MPO',
) -> torch.Tensor:
    """
    Custom implementation of the density-density correlation ❬ψ|nᵢnⱼ|ψ❭ for the state vector solver.

    TODO: extend to arbitrary two-point correlation ❬ψ|AᵢBⱼ|ψ❭
    """
    return state.get_correlation_matrix().to("cpu")


def energy_variance_mps_impl(
    self: EnergyVariance,
    *,
    config:'MPSConfig',
    state: 'MPS',
    H: 'MPO',
) -> torch.Tensor:
    """
    Custom implementation of the energy variance ❬ψ|H²|ψ❭-❬ψ|H|ψ❭² for the state vector solver.
    """
    h_squared = H @ H
    return (h_squared.expect(state).real - H.expect(state).real ** 2).to("cpu")# type: ignore[no-any-return]


def energy_second_moment_mps_impl(
    self: EnergySecondMoment,
    *,
    config:'MPSConfig',
    state: 'MPS',
    H: 'MPO',
) -> torch.Tensor:
    """
    Custom implementation of the second moment of energy ❬ψ|H²|ψ❭
    for the state vector solver.
    """
    H_square = H @ H
    return H_square.expect(state).real.to("cpu")

def energy_mps_impl(
    self: Energy,
    *,
    config:'MPSConfig',
    state: 'MPS',
    H: 'MPO',
) -> torch.Tensor:
    """
    Custom implementation of the second moment of energy ❬ψ|H²|ψ❭
    for the state vector solver.
    """
    return H.expect(state).real.to("cpu")