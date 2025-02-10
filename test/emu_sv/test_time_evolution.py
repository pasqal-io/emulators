import torch
from functools import reduce

from emu_sv.hamiltonian import RydbergHamiltonian
from emu_sv.time_evolution import evolve_sv_rydberg
from emu_sv.sv_config import SVConfig

dtype = torch.complex128
dtype_params = torch.float64

torch.manual_seed(1337)


device = "cpu"
# device = "cuda"


def single_gate(i: int, nqubits: int, g: torch.Tensor):
    matrices = [torch.eye(2, 2, dtype=dtype)] * nqubits
    matrices[i] = g
    return reduce(torch.kron, matrices)


def sigma_x(i: int, nqubits: int) -> torch.Tensor:
    σ_x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
    return single_gate(i, nqubits, σ_x)


def pu(i, nqubits):
    n = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=dtype)
    return single_gate(i, nqubits, n)


def n(i, j, nqubits):
    n = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=dtype)
    matrices = [torch.eye(2, 2, dtype=dtype)] * nqubits
    matrices[i] = n
    matrices[j] = n
    return reduce(torch.kron, matrices)


def sv_hamiltonian(
    inter_matrix: torch.Tensor, omega: list[torch.Tensor], delta: list[torch.Tensor]
) -> torch.Tensor:
    n_qubits = inter_matrix.size(dim=1)
    device = omega[0].device
    h = torch.zeros(2**n_qubits, 2**n_qubits, dtype=dtype, device=device)
    for i in range(n_qubits):
        h = h + omega[i] * sigma_x(i, n_qubits) / 2
        h = h - delta[i] * pu(i, n_qubits).to(dtype=dtype, device=device)

        for j in range(i + 1, n_qubits):
            h = h + inter_matrix[i, j] * n(i, j, n_qubits).to(dtype=dtype, device=device)

    return h


def test_forward():
    N = 8

    omega = torch.randn(N, dtype=dtype_params)
    delta = torch.randn(N, dtype=dtype_params)
    interactions = torch.zeros(N, N, dtype=dtype_params)
    for i in range(N - 1):
        interactions[i, i + 1] = 1
        interactions[i + 1, i] = 1

    grad = torch.randn(2**N, dtype=dtype).to(device)
    grad /= grad.norm()
    state = torch.randn(2**N, dtype=dtype).to(device)
    state /= state.norm()

    sv_config = SVConfig()

    h = sv_hamiltonian(interactions, omega, delta).to(device)
    dt = 1.0
    ed = torch.linalg.matrix_exp(-1j * dt * h) @ state
    ham = RydbergHamiltonian(
        omegas=omega,
        deltas=delta,
        interaction_matrix=interactions,
        device=device,
    )
    krylov = evolve_sv_rydberg(
        dt,
        ham,
        state,  # .reshape((2,) * N),
        sv_config.krylov_tolerance,
    )  # .reshape(-1)
    assert torch.allclose(ed, krylov)
