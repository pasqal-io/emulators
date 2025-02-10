import torch

from emu_base.math.krylov_exp import krylov_exp
from emu_sv.hamiltonian import RydbergHamiltonian


def evolve_sv_rydberg(
    dt: float,
    ham: RydbergHamiltonian,
    state_vector: torch.Tensor,
    krylov_tolerance: float,
) -> torch.Tensor:
    op = lambda x: -1j * dt * (ham * x)
    return krylov_exp(
        op,
        state_vector,
        norm_tolerance=krylov_tolerance,
        exp_tolerance=krylov_tolerance,
    )
