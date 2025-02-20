import torch
from functools import reduce
import pytest

from emu_sv.hamiltonian import RydbergHamiltonian

dtype = torch.complex128
params_dtype = torch.float64
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


@pytest.mark.parametrize(
    ("N", "make_phases"),
    [(5, torch.rand), (7, torch.rand), (3, torch.zeros), (8, torch.zeros)],
)
def test_dense_vs_sparse(N: int, make_phases):
    torch.manual_seed(1337)
    omegas = torch.randn(N, dtype=params_dtype, device=device)
    deltas = torch.randn(N, dtype=params_dtype, device=device)
    phis = make_phases(N, dtype=params_dtype, device=device)
    interaction_matrix = torch.randn(N, N, dtype=params_dtype)

    ham_dense = sv_hamiltonian(interaction_matrix, omegas, deltas, phis).to(device)
    ham = RydbergHamiltonian(
        omegas=omegas,
        deltas=deltas,
        phis=phis,
        interaction_matrix=interaction_matrix,
        device=device,
    )

    # test hamiltonian diagonal terms
    diag_sparse = ham.diag
    diag_dense = torch.diagonal(ham_dense).to(dtype=dtype)
    assert torch.allclose(diag_sparse.reshape(-1), diag_dense)

    # test H_dense @ |ψ❭ == H|ψ❭
    state = torch.randn(2**N, dtype=dtype, device=device)

    res_dense = ham_dense @ state
    res_sparse = ham * state
    assert torch.allclose(res_sparse, res_dense)


def test_call_real_complex():
    torch.manual_seed(1337)
    N = 2
    omegas = torch.randn(N, dtype=params_dtype, device=device)
    deltas = torch.randn(N, dtype=params_dtype, device=device)
    interaction_matrix = torch.randn(N, N, dtype=params_dtype)

    ham_w_phase = RydbergHamiltonian(
        omegas=omegas,
        deltas=deltas,
        phis=torch.randn(2, dtype=params_dtype, device=device),
        interaction_matrix=interaction_matrix,
        device=device,
    )
    assert (
        ham_w_phase._apply_sigma_operators == ham_w_phase._apply_sigma_operators_complex
    )

    ham_zero_phase = RydbergHamiltonian(
        omegas=omegas,
        deltas=deltas,
        phis=torch.zeros(2, dtype=params_dtype, device=device),
        interaction_matrix=interaction_matrix,
        device=device,
    )
    assert (
        ham_zero_phase._apply_sigma_operators
        == ham_zero_phase._apply_sigma_operators_real
    )
