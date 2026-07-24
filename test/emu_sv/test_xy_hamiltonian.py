import torch
import pytest
from test.utils_testing import (
    dense_xy_hamiltonian,
    nn_interaction_matrix,
    randn_interaction_matrix,
)
from test.utils_testing.utils_dense_hamiltonians import single_gate
from emu_sv.state_vector import StateVector
from emu_sv.xy_hamiltonian import XYHamiltonian

dtype = torch.complex128
device = "cpu"


@pytest.mark.parametrize("N", [3, 5, 7, 8])
def test_dense_vs_sparse_no_phase(N: int) -> None:
    torch.manual_seed(1337)
    omegas = torch.randn(N)
    deltas = torch.randn(N)
    phis = torch.zeros(N)
    interaction_matrix = nn_interaction_matrix(N)

    ham_dense = dense_xy_hamiltonian(omegas, deltas, phis, interaction_matrix).to(device)
    ham = XYHamiltonian(
        omegas=omegas,
        deltas=deltas,
        phis=phis,
        interaction_matrix=interaction_matrix,
        device=device,
        noise=None,
    )

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

    ham_dense = dense_xy_hamiltonian(omegas, deltas, phis, interaction_matrix).to(device)
    ham = XYHamiltonian(
        omegas=omegas,
        deltas=deltas,
        phis=phis,
        interaction_matrix=interaction_matrix,
        device=device,
        noise=None,
    )

    # test H_dense @ |ψ❭ == H*|ψ❭
    state = torch.randn(2**N, dtype=dtype, device=device)

    res_dense = ham_dense @ state
    res_sparse = ham * state
    assert torch.allclose(res_sparse, res_dense, atol=1e-12)


def test_input_state_unchanged() -> None:
    torch.manual_seed(1337)
    N = 4
    ham = XYHamiltonian(
        omegas=torch.randn(N),
        deltas=torch.randn(N),
        phis=torch.randn(N),
        interaction_matrix=randn_interaction_matrix(N),
        device=device,
        noise=None,
    )
    state = torch.randn(2**N, dtype=dtype, device=device)
    state_copy = state.clone()
    ham * state
    assert torch.equal(state, state_copy)


@pytest.mark.parametrize("N", [2, 5, 8])
def test_dense_vs_sparse_with_noise(N: int) -> None:
    torch.manual_seed(1337)
    omegas = torch.randn(N)
    deltas = torch.randn(N)
    phis = torch.randn(N)
    interaction_matrix = randn_interaction_matrix(N)
    # -0.5i∑ⱼLⱼ†Lⱼ for some generic single-qubit lindbladians
    noise = torch.randn(2, 2, dtype=dtype)

    ham_dense = dense_xy_hamiltonian(omegas, deltas, phis, interaction_matrix).to(device)
    for i in range(N):
        ham_dense += single_gate(i, N, noise).to(device)
    ham = XYHamiltonian(
        omegas=omegas,
        deltas=deltas,
        phis=phis,
        interaction_matrix=interaction_matrix,
        device=device,
        noise=noise,
    )

    # test H_dense @ |ψ❭ == H*|ψ❭
    state = torch.randn(2**N, dtype=dtype, device=device)

    res_dense = ham_dense @ state
    res_sparse = ham * state
    assert torch.allclose(res_sparse, res_dense, atol=1e-12)


def test_expect():
    torch.manual_seed(1337)
    N = 3
    omegas = torch.randn(N, dtype=torch.float64)
    deltas = torch.randn(N, dtype=torch.float64)
    phis = torch.randn(N, dtype=torch.float64)
    interaction_matrix = randn_interaction_matrix(N)
    eigenstates = ("r", "g")
    state = StateVector.from_state_amplitudes(
        eigenstates=eigenstates, amplitudes={"rrr": 1, "rgg": 1}
    )
    h = XYHamiltonian(
        omegas=omegas,
        deltas=deltas,
        phis=phis,
        interaction_matrix=interaction_matrix,
        device=state.data.device,
        noise=None,
    )
    ham_dense = dense_xy_hamiltonian(omegas, deltas, phis, interaction_matrix).to(
        state.data.device
    )
    expected = torch.vdot(state.data, ham_dense @ state.data).real
    assert torch.allclose(h.expect(state), expected, atol=1e-12)
