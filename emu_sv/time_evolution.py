from typing import Optional
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
    linblad_ops: Optional[list[torch.Tensor]],
) -> tuple[torch.Tensor, RydbergHamiltonian]:
    ham = RydbergHamiltonian(
        omegas=omegas,
        deltas=deltas,
        phis=phis,
        interaction_matrix=full_interaction_matrix,
        device=state_vector.device,
    )
    op = lambda x: -1j * dt * (ham * x)

    def op_noise(x: torch.Tensor) -> torch.Tensor:
        ham_eff: torch.Tensor = ham * (x * -1.0j)
        return dt * (ham_eff + ham_eff.T.conj())

    # for the noise, something like this
    # def op(x):
    #    num_qubits = int(math.log2(len(x)))
    #    ham_eff = -1.0j * const * sigma_x @ x - 0.5 * l1dag @ l1 @ x
    #
    #    return 10*(ham_eff + ham_eff.T.conj() + l1 @ x @ l1dag)

    return (
        krylov_exp(
            op,
            state_vector,
            norm_tolerance=krylov_tolerance,
            exp_tolerance=krylov_tolerance,
        ),
        ham,
    )
