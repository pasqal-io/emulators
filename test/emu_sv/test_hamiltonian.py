import torch

from emu_sv.hamiltonian import RydbergHamiltonian

from functools import reduce

dtype = torch.complex128
# device = "cuda"
device = "cpu"


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
    "Rydberg hamiltonian not complex part"
    n_qubits = inter_matrix.size(dim=1)
    device = omega[0].device
    h = torch.zeros(2**n_qubits, 2**n_qubits, dtype=dtype, device=device)
    for i in range(n_qubits):
        h += omega[i] * sigma_x(i, n_qubits).to(dtype=dtype, device=device) / 2
        h -= delta[i] * pu(i, n_qubits).to(dtype=dtype, device=device)

        for j in range(i + 1, n_qubits):
            h += inter_matrix[i, j] * n(i, j, n_qubits).to(dtype=dtype, device=device)

    return h


def test_dense_vs_sparse():
    N = 8
    torch.manual_seed(1337)
    omega = torch.randn(N, dtype=dtype, device=device)
    delta = torch.randn(N, dtype=dtype, device=device)
    interaction_matrix = torch.randn((N, N))

    h = sv_hamiltonian(interaction_matrix, omega, delta).to(device)
    v = torch.randn((2,) * N, dtype=dtype, device=device)

    res_dense = h @ v.reshape(-1)

    h_custom = RydbergHamiltonian(
        omegas=omega, deltas=delta, interaction_matrix=interaction_matrix, device=device
    )

    res_sparse = (h_custom * v).reshape(-1)

    assert torch.allclose(res_sparse, res_dense)

    diag_sparse = h_custom.diag
    diag_dense = torch.diagonal(h).to(dtype=dtype)

    assert torch.allclose(diag_sparse.reshape(-1), diag_dense)
