import torch
import pytest
from test.utils_testing import dense_rydberg_hamiltonian
from emu_sv.time_evolution import do_time_step

dtype = torch.complex128
device = "cpu"


def randn_interaction(N: int):
    temp_mat = torch.randn(N, N).fill_diagonal_(0.0)
    interactions = (temp_mat + temp_mat.mT) / 2
    return interactions


def nn_interaction(N: int):
    interactions = torch.zeros(N, N)
    for i in range(N - 1):
        interactions[i, i + 1] = 1
        interactions[i + 1, i] = 1
    return interactions


@pytest.mark.parametrize(
    ("N", "krylov_tolerance"),
    [(3, 1e-10), (5, 1e-12), (7, 1e-10), (8, 1e-12)],
)
def test_forward_no_phase(N: int, krylov_tolerance: float):
    torch.manual_seed(1337)
    omegas = torch.randn(N)
    deltas = torch.randn(N)
    phis = torch.zeros_like(omegas)
    interactions = nn_interaction(N)
    ham_params = (omegas, deltas, phis, interactions)

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
    )
    assert torch.allclose(ed, krylov, atol=krylov_tolerance)


@pytest.mark.parametrize(
    ("N", "krylov_tolerance"),
    [(3, 1e-10), (5, 1e-12), (7, 1e-10), (8, 1e-12)],
)
def test_forward_with_phase(N: int, krylov_tolerance: float):
    torch.manual_seed(1337)
    omegas = torch.randn(N)
    deltas = torch.randn(N)
    phis = torch.randn(N)
    interactions = randn_interaction(N)
    ham_params = (omegas, deltas, phis, interactions)

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
    )
    assert torch.allclose(ed, krylov, atol=krylov_tolerance)
