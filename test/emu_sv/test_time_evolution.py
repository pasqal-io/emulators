import torch
import pytest
from test.utils_testing import (
    dense_rydberg_hamiltonian,
    nn_interaction_matrix,
    randn_interaction_matrix,
)
from emu_sv.time_evolution import do_time_step

dtype = torch.complex128
device = "cpu"


@pytest.mark.parametrize(
    ("N", "krylov_tolerance"),
    [(3, 1e-10), (5, 1e-12), (7, 1e-10), (8, 1e-12)],
)
def test_forward_no_phase(N: int, krylov_tolerance: float) -> None:
    torch.manual_seed(1337)
    omegas = torch.randn(N)
    deltas = torch.randn(N)
    phis = torch.zeros_like(omegas)
    interaction = nn_interaction_matrix(N)
    ham_params = (omegas, deltas, phis, interaction)

    state = torch.randn(2**N, dtype=dtype, device=device)
    state /= state.norm()

    H = dense_rydberg_hamiltonian(*ham_params).to(device)
    dt = 1.0  # 1 μs big time step
    ed = torch.linalg.matrix_exp(-1j * dt * H) @ state
    krylov, _ = do_time_step(
        dt,
        *ham_params,
        state,
        krylov_tolerance,
        linblad_ops=None,
    )
    assert torch.allclose(ed, krylov, atol=krylov_tolerance)


@pytest.mark.parametrize(
    ("N", "krylov_tolerance"),
    [(3, 1e-10), (5, 1e-12), (7, 1e-10), (8, 1e-12)],
)
def test_forward_with_phase(N: int, krylov_tolerance: float) -> None:
    torch.manual_seed(1337)
    omegas, deltas, phis = torch.randn(3, N)  # unpack a 3*N tensor
    interaction = randn_interaction_matrix(N)
    ham_params = (omegas, deltas, phis, interaction)

    state = torch.randn(2**N, dtype=dtype, device=device)
    state /= state.norm()

    H = dense_rydberg_hamiltonian(*ham_params).to(device)
    dt = 1.0  # 1 μs big time step
    ed = torch.linalg.matrix_exp(-1j * dt * H) @ state
    krylov, _ = do_time_step(
        dt,
        *ham_params,
        state,
        krylov_tolerance,
        linblad_ops=None,
    )
    assert torch.allclose(ed, krylov, atol=krylov_tolerance)
