import torch
from functools import reduce
from emu_sv.hamiltonian import RydbergHamiltonian

dtype = torch.complex128
device = "cpu"


def single_gate(i: int, nqubits: int, g: torch.Tensor):
    matrices = [torch.eye(2, 2, dtype=dtype)] * nqubits
    matrices[i] = g
    return reduce(torch.kron, matrices)


def sigma_x(i: int, nqubits: int) -> torch.Tensor:
    σ_x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
    return single_gate(i, nqubits, σ_x)


def sigma_y(i: int, nqubits: int) -> torch.Tensor:
    σ_y = torch.tensor([[0.0, -1j], [1j, 0.0]], dtype=dtype)
    return single_gate(i, nqubits, σ_y)


def pu(i, nqubits):
    n = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=dtype)
    return single_gate(i, nqubits, n)


def nn(i, j, nqubits):
    n = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=dtype)
    matrices = [torch.eye(2, 2, dtype=dtype)] * nqubits
    matrices[i] = n
    matrices[j] = n
    return reduce(torch.kron, matrices)


def sv_hamiltonian(
    interaction_matrix: torch.Tensor,
    omegas: torch.Tensor,
    deltas: torch.Tensor,
    phis: torch.Tensor,
) -> torch.Tensor:
    """
    Dense Rydberg Hamiltonian for testing:
        H = ∑ⱼΩⱼ/2[cos(ϕⱼ)σˣⱼ + sin(ϕⱼ)σʸⱼ] - ∑ⱼΔⱼnⱼ + ∑ᵢ﹥ⱼUᵢⱼnᵢnⱼ
    """
    N = interaction_matrix.size(dim=1)
    device = omegas[0].device
    h = torch.zeros(2**N, 2**N, dtype=dtype, device=device)
    for i in range(N):
        h += (
            omegas[i]
            * torch.cos(phis[i])
            * sigma_x(i, N).to(dtype=dtype, device=device)
            / 2
        )
        h += (
            omegas[i]
            * torch.sin(phis[i])
            * sigma_y(i, N).to(dtype=dtype, device=device)
            / 2
        )
        h -= deltas[i] * pu(i, N).to(dtype=dtype, device=device)

        for j in range(i + 1, N):
            h += interaction_matrix[i, j] * nn(i, j, N).to(dtype=dtype, device=device)
    return h


def test_dense_vs_sparse():
    N = 8
    torch.manual_seed(1337)
    params_dtype = torch.float64
    omegas = torch.randn(N, dtype=params_dtype, device=device)
    deltas = torch.randn(N, dtype=params_dtype, device=device)
    phis = torch.randn(N, dtype=params_dtype, device=device)
    interaction_matrix = torch.randn(N, N, dtype=params_dtype)

    h = sv_hamiltonian(interaction_matrix, omegas, deltas, phis).to(device)
    v = torch.randn((2,) * N, dtype=dtype, device=device)

    res_dense = h @ v.reshape(-1)

    h_custom = RydbergHamiltonian(
        omegas=omegas,
        deltas=deltas,
        phis=phis,
        interaction_matrix=interaction_matrix,
        device=device,
    )

    res_sparse = (h_custom * v).reshape(-1)

    assert torch.allclose(res_sparse, res_dense)

    diag_sparse = h_custom.diag
    diag_dense = torch.diagonal(h).to(dtype=dtype)

    assert torch.allclose(diag_sparse.reshape(-1), diag_dense)
