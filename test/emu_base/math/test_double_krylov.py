import torch
from emu_base.math.double_krylov import double_krylov, double_krylov_2
from emu_sv.hamiltonian import RydbergHamiltonian
from test.utils_testing.utils_dense_hamiltonians import dense_rydberg_hamiltonian
from pytest import mark

dtype = torch.complex128
dtype_params = torch.float64


def frechet_exp(A: torch.Tensor, E: torch.Tensor):
    """Returns exp([[A,E],[0,A]])"""
    big_mat = torch.block_diag(A, A)
    sizeA = A.shape[0]
    big_mat[:sizeA, sizeA:] = E
    big_exp = torch.linalg.matrix_exp(big_mat)
    return big_exp[:sizeA, sizeA:]


@mark.parametrize(
    "N, tolerance",
    [(n, tol) for n in [5, 7] for tol in [1e-8, 1e-10, 1e-12]],
)
def test_double_krylov(N, tolerance):
    torch.manual_seed(1337)
    omegas = torch.randn(N, dtype=dtype_params)
    deltas = torch.randn(N, dtype=dtype_params)
    phis = torch.randn(N, dtype=dtype_params)
    interactions = torch.zeros(N, N, dtype=dtype_params)
    for i in range(N):
        for j in range(i + 1, N):
            interactions[i, j] = 1 / abs(j - i)
    state = torch.randn(2**N, dtype=dtype)
    state = state / state.norm()

    grad = torch.randn(2**N, dtype=dtype)

    dt, iteration_count = 1.0, 80
    ham = RydbergHamiltonian(
        omegas=omegas,
        deltas=deltas,
        phis=phis,
        interaction_matrix=interactions,
        device=state.device,
    )

    op = lambda x: -1j * dt * (ham * x)
    lanczos_vectors_even, odd_block, eT = double_krylov(
        op, grad, state, iteration_count, tolerance
    )

    even_block = torch.stack(lanczos_vectors_even)
    Hess_L = eT[1 : 2 * odd_block.shape[0] : 2, : 2 * even_block.shape[0] : 2]
    # L = V_odd @ Hess_L @ V_even*
    L = odd_block.mT @ Hess_L @ even_block.conj()

    Hsv = dense_rydberg_hamiltonian(omegas, deltas, phis, interactions)
    E = state.unsqueeze(-1) @ grad.conj().unsqueeze(0)
    expected_L = frechet_exp(-1j * dt * Hsv, E)

    assert torch.allclose(L, expected_L, atol=tolerance)


@mark.parametrize(
    "N, tolerance",
    [(n, tol) for n in [5, 7] for tol in [1e-8, 1e-10, 1e-12]],
)
def test_double_krylov_2(N, tolerance):
    torch.manual_seed(1337)
    omegas = torch.randn(N, dtype=dtype_params)
    deltas = torch.randn(N, dtype=dtype_params)
    phis = torch.randn(N, dtype=dtype_params)
    interactions = torch.zeros(N, N, dtype=dtype_params)
    for i in range(N):
        for j in range(i + 1, N):
            interactions[i, j] = 1 / abs(j - i)
    state = torch.randn(2**N, dtype=dtype)
    state = state / state.norm()

    grad = torch.randn(2**N, dtype=dtype)

    dt = 1.0
    ham = RydbergHamiltonian(
        omegas=omegas,
        deltas=deltas,
        phis=phis,
        interaction_matrix=interactions,
        device=state.device,
    )

    op = lambda x: -1j * dt * (ham * x)
    lanczos_vectors_even, lanczos_vectors_odd, eT = double_krylov_2(
        op, grad, state, tolerance
    )

    odd_block = torch.stack(lanczos_vectors_odd)
    even_block = torch.stack(lanczos_vectors_even)
    Hess_L = eT[1 : 2 * odd_block.shape[0] : 2, : 2 * even_block.shape[0] : 2]
    # L = V_odd @ Hess_L @ V_even*
    L = odd_block.mT @ Hess_L @ even_block.conj()

    Hsv = dense_rydberg_hamiltonian(omegas, deltas, phis, interactions)
    E = state.unsqueeze(-1) @ grad.conj().unsqueeze(0)
    expected_L = frechet_exp(-1j * dt * Hsv, E)

    assert torch.allclose(L, expected_L, atol=tolerance)
