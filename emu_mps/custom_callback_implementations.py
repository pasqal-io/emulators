import torch

from pulser.backend.default_observables import (
    CorrelationMatrix,
    EnergySecondMoment,
    EnergyVariance,
    Occupation,
    Energy,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emu_mps.mps_config import MPSConfig
    from emu_mps.mps import MPS
    from emu_mps.mpo import MPO


def qubit_occupation_mps_impl(
    self: Occupation,
    *,
    config: "MPSConfig",
    state: "MPS",
    hamiltonian: "MPO",
) -> torch.Tensor:
    """
    Custom implementation of the occupation ❬ψ|nᵢ|ψ❭ for the EMU-MPS.
    """
    dim = state.dim
    op = torch.zeros(1, dim, dim, dtype=torch.complex128, device=state.factors[0].device)
    op[0, 1, 1] = 1.0
    return state.expect_batch(op).real.view(-1).cpu()


def correlation_matrix_mps_impl(
    self: CorrelationMatrix,
    *,
    config: "MPSConfig",
    state: "MPS",
    hamiltonian: "MPO",
) -> torch.Tensor:
    """
    Custom implementation of the density-density correlation ❬ψ|nᵢnⱼ|ψ❭ for the EMU-MPS.

    TODO: extend to arbitrary two-point correlation ❬ψ|AᵢBⱼ|ψ❭
    """
    return state.get_correlation_matrix().cpu()


def energy_variance_mps_impl(
    self: EnergyVariance,
    *,
    config: "MPSConfig",
    state: "MPS",
    hamiltonian: "MPO",
) -> torch.Tensor:
    """
    Custom implementation of the energy variance ❬ψ|H²|ψ❭-❬ψ|H|ψ❭² for the EMU-MPS.
    """
    h_squared = hamiltonian @ hamiltonian
    h_2 = h_squared.expect(state).cpu()
    h = hamiltonian.expect(state).cpu()
    en_var = h_2 - h**2
    return en_var.real  # type: ignore[no-any-return]


def energy_second_moment_mps_impl(
    self: EnergySecondMoment,
    *,
    config: "MPSConfig",
    state: "MPS",
    hamiltonian: "MPO",
) -> torch.Tensor:
    """
    Custom implementation of the second moment of energy ❬ψ|H²|ψ❭
    for the EMU-MPS.
    """
    h_square = hamiltonian @ hamiltonian
    h_2 = h_square.expect(state).cpu()
    assert torch.allclose(h_2.imag, torch.zeros_like(h_2.imag), atol=1e-4)
    return h_2.real


def energy_mps_impl(
    self: Energy,
    *,
    config: "MPSConfig",
    state: "MPS",
    hamiltonian: "MPO",
) -> torch.Tensor:
    """
    Custom implementation of the second moment of energy ❬ψ|H²|ψ❭
    for the EMU-MPS.
    """
    h = hamiltonian.expect(state)
    assert torch.allclose(h.imag, torch.zeros_like(h.imag), atol=1e-4)
    return h.real
