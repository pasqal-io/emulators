import torch
from test.utils_testing import dense_rydberg_hamiltonian
from emu_sv.time_evolution import do_time_step
from emu_sv.sv_config import SVConfig

dtype = torch.complex128
dtype_params = torch.float64
device = "cpu"


def test_forward():
    torch.manual_seed(1337)
    N = 8
    omegas = torch.randn(N, dtype=dtype_params)
    deltas = torch.randn(N, dtype=dtype_params)
    phis = torch.zeros_like(omegas)
    interactions = torch.zeros(N, N, dtype=dtype_params)
    for i in range(N - 1):
        interactions[i, i + 1] = 1
        interactions[i + 1, i] = 1

    state = torch.randn(2**N, dtype=dtype, device=device)
    state /= state.norm()

    sv_config = SVConfig()

    h = dense_rydberg_hamiltonian(interactions, omegas, deltas, phis).to(device)
    dt = 1.0  # 1 μs
    ed = torch.linalg.matrix_exp(-1j * dt * h) @ state
    krylov, _ = do_time_step(
        dt,
        omegas,
        deltas,
        phis,
        interactions,
        state,
        sv_config.krylov_tolerance,
    )
    assert torch.allclose(ed, krylov)
