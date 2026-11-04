import torch
import pytest
from unittest.mock import patch
from test.utils_testing import (
    EPS,
    dense_rydberg_hamiltonian,
    nn_interaction_matrix,
    randn_interaction_matrix,
    requires_kernel,
)
from emu_sv.state_vector import StateVector
from emu_sv.hamiltonian import RydbergHamiltonian, make_interaction_diagonal
from emu_sv import hamiltonian_kernels as kernels
from emu_sv.time_evolution import EvolveStateVector

dtype = torch.complex128
device = "cpu"


def mock_apply_sigma_operators(result, vec):
    pass


@pytest.mark.usefixtures("restore_rng_state")
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
        noise=None,
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


@pytest.mark.usefixtures("restore_rng_state")
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
        noise=None,
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


@pytest.mark.usefixtures("restore_rng_state")
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
            device=state.data.device,
            noise=None,
        )
        ham_w_phase * state.data
        assert ham_w_phase.complex
        ham_w_phase._apply_sigma_operators_complex.assert_called_once()

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
            device=state.data.device,
            noise=None,
        )
        ham_zero_phase * state.data
        assert not ham_zero_phase.complex
        ham_zero_phase._apply_sigma_operators_real.assert_called_once()


@pytest.mark.usefixtures("restore_rng_state")
def test_expect():
    torch.manual_seed(1337)
    N = 3
    omegas = torch.randn(N, dtype=torch.float64)
    deltas = torch.randn(N, dtype=torch.float64)
    interaction_matrix = torch.randn(N, N, dtype=torch.float64)
    interaction_matrix = interaction_matrix + interaction_matrix.T
    eigenstates = ("r", "g")
    state = StateVector.from_state_amplitudes(
        eigenstates=eigenstates, amplitudes={"rrr": 1, "rgg": 1}
    )
    h0 = RydbergHamiltonian(
        omegas=omegas,
        deltas=deltas,
        phis=torch.zeros(N),
        interaction_matrix=interaction_matrix,
        device=state.data.device,
        noise=None,
    )
    h = RydbergHamiltonian(
        omegas=omegas,
        deltas=deltas,
        phis=torch.zeros(N),
        interaction_matrix=interaction_matrix,
        device=state.data.device,
        noise=torch.tensor([[0.0, 0.0], [0.0, -0.5j]], dtype=torch.complex128),
    )
    assert torch.allclose(
        h0.expect(state),
        0.5
        * (
            -2 * deltas[0]
            - deltas[1]
            - deltas[2]
            + interaction_matrix[0, 1]
            + interaction_matrix[0, 2]
            + interaction_matrix[1, 2]
        ),
    )
    assert h0.expect(state) == h.expect(state)


@pytest.mark.usefixtures("restore_rng_state")
def test_interaction_diagonal_is_kept_between_steps() -> None:
    torch.manual_seed(1337)
    N = 4
    stepper = EvolveStateVector()
    first = randn_interaction_matrix(N)
    diag = stepper.interaction_diagonal(first, torch.device(device))
    assert stepper.interaction_diagonal(first.clone(), torch.device(device)) is diag

    zeros = torch.zeros(N)
    for interaction_matrix in [randn_interaction_matrix(N), first]:
        ham = RydbergHamiltonian(
            omegas=zeros,
            deltas=zeros,
            phis=zeros,
            interaction_matrix=interaction_matrix,
            device=device,
            noise=None,
            interaction_diag=stepper.interaction_diagonal(
                interaction_matrix, torch.device(device)
            ),
        )
        dense = dense_rydberg_hamiltonian(zeros, zeros, zeros, interaction_matrix)
        assert torch.allclose(ham.diag, torch.diagonal(dense).to(dtype), atol=1e-12)


@pytest.mark.skipif(
    not torch.cuda.is_available() or kernels.triton is None,
    reason="the fused kernel needs CUDA and Triton",
)
@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize(
    "N, with_phase, with_noise",
    [(n, p, nz) for n in [8, 11] for p in [False, True] for nz in [False, True]],
)
def test_fused_kernel_matches_torch(N: int, with_phase: bool, with_noise: bool) -> None:
    torch.manual_seed(1337)
    noise = (
        torch.tensor([[-0.1j, 0.2 + 0.1j], [0.3j, -0.4j]], dtype=dtype)
        if with_noise
        else None
    )
    omegas = torch.randn(N, dtype=torch.float64)
    deltas = torch.randn(N, dtype=torch.float64)
    phis = torch.randn(N, dtype=torch.float64) if with_phase else torch.zeros(N)
    interaction_matrix = randn_interaction_matrix(N)
    ham = RydbergHamiltonian(
        omegas=omegas,
        deltas=deltas,
        phis=phis,
        interaction_matrix=interaction_matrix,
        device="cuda",
        noise=noise,
    )
    state = torch.randn(2**N, dtype=dtype, device="cuda")
    assert ham._use_kernel(state)
    fused = ham * state
    fused_op = ham.make_op(-0.3j)(state)
    with patch.object(kernels, "is_available", return_value=False):
        reference = ham * state
    assert torch.allclose(fused, reference, atol=1e-12)
    assert torch.allclose(fused_op, -0.3j * reference, atol=1e-12)

    # the kernel is not differentiable: autograd falls back to torch
    ham_grad = RydbergHamiltonian(
        omegas=omegas.clone().requires_grad_(),
        deltas=deltas,
        phis=phis,
        interaction_matrix=interaction_matrix,
        device="cuda",
        noise=noise,
    )
    assert not ham_grad._use_kernel(state)
    with torch.no_grad():
        assert ham_grad._use_kernel(state)


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("with_phase", [False, True])
@pytest.mark.parametrize("input_dtype", [torch.float64, torch.complex128])
def test_spectral_bounds(with_phase: bool, input_dtype: torch.dtype) -> None:
    """
    d is the real diagonal of H; the bounds are min(d) - R and max(d) + R with
    R = sum_j |Omega_j| / 2 (Gershgorin), whatever the phases; they contain the spectrum.
    Negative and zero Omega, detunings of both signs, complex inputs as from the adapter.
    """
    torch.manual_seed(1337)
    N = 5
    omegas = torch.randn(N, dtype=torch.float64)
    omegas[0] = -omegas[0].abs() - 0.5
    omegas[2] = 0.0
    deltas = torch.randn(N, dtype=torch.float64)
    deltas[1] = deltas[1].abs() + 0.5
    deltas[3] = -deltas[3].abs() - 0.5
    phis = torch.randn(N, dtype=torch.float64) if with_phase else torch.zeros(N)
    interaction_matrix = randn_interaction_matrix(N)
    interaction_diag = make_interaction_diagonal(interaction_matrix, device)
    interaction_diag_copy = interaction_diag.clone()
    dense = dense_rydberg_hamiltonian(omegas, deltas, phis, interaction_matrix)
    d_dense = torch.diagonal(dense).real
    radius = omegas.abs().sum().item() / 2
    eigs = torch.linalg.eigvalsh(dense)

    ham = RydbergHamiltonian(
        omegas=omegas.to(input_dtype),
        deltas=deltas.to(input_dtype),
        phis=phis.to(input_dtype),
        interaction_matrix=interaction_matrix,
        device=device,
        noise=None,
        interaction_diag=interaction_diag,
    )
    d, lower, upper = ham.spectral_bounds()

    assert d.dtype == torch.float64
    assert isinstance(lower, float) and isinstance(upper, float)
    # the stepper keeps interaction_diag between steps: it must not be modified
    assert torch.equal(interaction_diag, interaction_diag_copy)
    # Budget: each diagonal entry sums at most N + N(N-1)/2 terms, both here and in the
    # dense reference: |error| <= 2 * n_terms * eps * S, S = sum |Delta| + sum |U|.
    # R sums N terms: N eps R. atol = 2x the budget. Bugs (R doubled or halved, Delta
    # left out) move the bounds by >= 0.25.
    n_terms = N + N * (N - 1) // 2
    scale = deltas.abs().sum().item() + interaction_matrix.abs().sum().item() / 2
    atol = 2 * (2 * n_terms * EPS * scale + N * EPS * radius)
    torch.testing.assert_close(d, d_dense, rtol=0.0, atol=atol)
    assert lower == pytest.approx(d_dense.min().item() - radius, rel=0.0, abs=atol)
    assert upper == pytest.approx(d_dense.max().item() + radius, rel=0.0, abs=atol)
    # eigvalsh error <~ dim eps ||H||
    slack = 2**N * EPS * eigs.abs().max().item()
    assert lower <= eigs[0].item() + slack
    assert upper >= eigs[-1].item() - slack


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize(
    "dev", ["cpu", pytest.param("cuda", marks=requires_kernel, id="cuda")]
)
@pytest.mark.parametrize("with_phase", [False, True])
def test_shifted_interaction_diag(dev: str, with_phase: bool) -> None:
    """
    The Chebyshev steps shift H by a constant c through `interaction_diag - c`:
    both the torch path and the fused kernel then apply H - c.
    """
    torch.manual_seed(1337)
    N = 8  # the kernel needs N >= MIN_QUBITS
    omegas = torch.randn(N, dtype=torch.float64)
    deltas = torch.randn(N, dtype=torch.float64)
    phis = torch.randn(N, dtype=torch.float64) if with_phase else torch.zeros(N)
    interaction_matrix = randn_interaction_matrix(N)
    shift = 2.75
    ham = RydbergHamiltonian(
        omegas=omegas,
        deltas=deltas,
        phis=phis,
        interaction_matrix=interaction_matrix,
        device=dev,
        noise=None,
        interaction_diag=make_interaction_diagonal(interaction_matrix, dev) - shift,
    )
    state = torch.randn(2**N, dtype=dtype, device=dev)
    assert ham._use_kernel(state) == (dev == "cuda")

    dense = dense_rydberg_hamiltonian(omegas, deltas, phis, interaction_matrix)
    expected = (dense - shift * torch.eye(2**N, dtype=dtype)) @ state.cpu()
    # Budget per entry: the dense matrix sums <= N^2 terms per entry, the sparse product
    # N + 1 products and the shifted diagonal N + N(N-1)/2 terms, all bounded by
    # S = sum|Delta| + sum|U| + |c| + sum|Omega|/2 times max|psi|. atol = 4 N^2 eps S max|psi|.
    scale = (
        deltas.abs().sum() + interaction_matrix.abs().sum() / 2 + shift
    ).item() + omegas.abs().sum().item() / 2
    atol = 4 * N**2 * EPS * scale * state.abs().max().item()
    torch.testing.assert_close((ham * state).cpu(), expected, rtol=0.0, atol=atol)
