import torch

from emu_base.math.krylov_exp import krylov_exp
from emu_sv.hamiltonian import RydbergHamiltonian


def do_time_step(
    dt: float,
    omegas: torch.Tensor,
    deltas: torch.Tensor,
    phis: torch.Tensor,
    full_interaction_matrix: torch.Tensor,
    state_vector: torch.Tensor,
    krylov_tolerance: float,
) -> tuple[torch.Tensor, RydbergHamiltonian]:
    ham = RydbergHamiltonian(
        omegas=omegas,
        deltas=deltas,
        phis=phis,
        interaction_matrix=full_interaction_matrix,
        device=state_vector.device,
    )
    op = lambda x: -1j * dt * (ham * x)
    return (
        krylov_exp(
            op,
            state_vector,
            norm_tolerance=krylov_tolerance,
            exp_tolerance=krylov_tolerance,
        ),
        ham,
    )
