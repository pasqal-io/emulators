import torch
import pytest
from unittest.mock import patch
from test.utils_testing import (
    dense_rydberg_hamiltonian,
    nn_interaction_matrix,
    randn_interaction_matrix,
)
from emu_sv.state_vector import StateVector
from emu_sv.hamiltonian import RydbergHamiltonian

dtype = torch.complex128
device = "cpu"


def mock_apply_sigma_operators(result, vec):
    pass


@pytest.mark.parametrize("N", [3, 5, 7, 8])
def test_dense_vs_sparse_no_phase(N: int) -> None:
    torch.manual_seed(1337)
    omegas = torch.randn(N)
    deltas = torch.randn(N)
    phis = torch.zeros(N)
    interaction_matrix = nn_interaction_matrix(N)

    ham_dense = dense_rydberg_hamiltonian(omegas, deltas, phis, interaction_matrix).to(
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
    assert torch.allclose(diag_sparse.reshape(-1), diag_dense, atol=1e-12)

    # test H_dense @ |ψ❭ == H*|ψ❭
    state = torch.randn(2**N, dtype=dtype, device=device)

    res_dense = ham_dense @ state
    res_sparse = ham * state
    assert torch.allclose(res_sparse, res_dense, atol=1e-12)


@pytest.mark.parametrize("N", [2, 4, 7, 9])
def test_dense_vs_sparse_with_phase(N: int) -> None:
    torch.manual_seed(1337)
    omegas = torch.randn(N)
    deltas = torch.randn(N)
    phis = torch.randn(N)
    interaction_matrix = randn_interaction_matrix(N)

    ham_dense = dense_rydberg_hamiltonian(omegas, deltas, phis, interaction_matrix).to(
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
    assert torch.allclose(diag_sparse.reshape(-1), diag_dense, atol=1e-12)

    # test H_dense @ |ψ❭ == H*|ψ❭
    state = torch.randn(2**N, dtype=dtype, device=device)

    res_dense = ham_dense @ state
    res_sparse = ham * state
    assert torch.allclose(res_sparse, res_dense, atol=1e-12)


def test_call_sigma_real_complex() -> None:
    torch.manual_seed(1337)
    N = 2
    omegas = torch.randn(N)
    deltas = torch.randn(N)
    interaction_matrix = torch.randn(N, N)
    state = StateVector.make(N)

    with patch.object(
        RydbergHamiltonian,
        "_apply_sigma_operators_complex",
        side_effect=mock_apply_sigma_operators,
    ):
        ham_w_phase = RydbergHamiltonian(
            omegas=omegas,
            deltas=deltas,
            phis=torch.randn(2),
            interaction_matrix=interaction_matrix,
            device=state.vector.device,
        )
        ham_w_phase * state.vector
        assert ham_w_phase.complex
        assert ham_w_phase._apply_sigma_operators_complex.called_once()

    with patch.object(
        RydbergHamiltonian,
        "_apply_sigma_operators_real",
        side_effect=mock_apply_sigma_operators,
    ):
        ham_zero_phase = RydbergHamiltonian(
            omegas=omegas,
            deltas=deltas,
            phis=torch.zeros(2),
            interaction_matrix=interaction_matrix,
            device=state.vector.device,
        )
        ham_zero_phase * state.vector
        assert not ham_zero_phase.complex
        assert ham_zero_phase._apply_sigma_operators_real.called_once()
