import torch
from emu_base.math.double_krylov import double_krylov
from emu_sv.hamiltonian import RydbergHamiltonian
from test.utils_testing.utils_dense_hamiltonians import dense_rydberg_hamiltonian
from test.utils_testing.utils_interaction_matrix import randn_interaction_matrix
from pytest import mark

dtype = torch.complex128
dtype_params = torch.float64
# to test locally on GPU just change device here
device = "cpu"


def frechet_exp(A: torch.Tensor, E: torch.Tensor):
    """
    Returns the Frechet derivative of exp(A) with
    respect to the displacement E, computed as
        [[exp(A), Dexp(A,E)],[0, exp(A)]] = exp([[A,E],[0,A]])
    """
    big_mat = torch.block_diag(A, A)
    sizeA = A.shape[0]
    big_mat[:sizeA, sizeA:] = E
    big_exp = torch.linalg.matrix_exp(big_mat)
    return big_exp[:sizeA, sizeA:]


@mark.parametrize(
    "N, tolerance",
    [(n, tol) for n in [3, 4, 5, 7] for tol in [1e-8, 1e-10, 1e-12]],
)
def test_double_krylov(N, tolerance):
    torch.manual_seed(1337)
    omegas = torch.randn(N, dtype=dtype_params)
    deltas = torch.randn(N, dtype=dtype_params)
    phis = torch.randn(N, dtype=dtype_params)
    interactions = randn_interaction_matrix(N)
    state = torch.randn(2**N, dtype=dtype, device=device)
    state = state / state.norm()

    grad = torch.randn(2**N, dtype=state.dtype, device=state.device)

    dt = 1.0  # big timestep 1 μs
    ham = RydbergHamiltonian(
        omegas=omegas,
        deltas=deltas,
        phis=phis,
        interaction_matrix=interactions,
        device=device,
    )

    op = lambda x: -1j * dt * (ham * x)
    lanczos_vectors_state, dS, lanczos_vectors_grad = double_krylov(
        op, state, grad, tolerance
    )

    state_block = torch.stack(lanczos_vectors_state)
    grad_block = torch.stack(lanczos_vectors_grad)
    dU = state_block.mT @ dS @ grad_block.conj()

    Hsv = dense_rydberg_hamiltonian(omegas, deltas, phis, interactions).to(device=device)
    # build E = |state❭❬grad|
    E = state.unsqueeze(-1) @ grad.conj().unsqueeze(0)
    expected_dU = frechet_exp(-1j * dt * Hsv, E)

    assert torch.allclose(dU, expected_dU, atol=tolerance, rtol=0)
