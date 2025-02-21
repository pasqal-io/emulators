import torch
import pytest
from test.utils_testing import dense_rydberg_hamiltonian
from emu_sv.hamiltonian import RydbergHamiltonian

dtype = torch.complex128
params_dtype = torch.float64
device = "cpu"


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

    ham_dense = dense_rydberg_hamiltonian(interaction_matrix, omegas, deltas, phis).to(
        device
    )
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
