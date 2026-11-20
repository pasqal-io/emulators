import logging
import torch
import pytest
from unittest.mock import MagicMock, patch
from test.utils_testing import (
    dense_rydberg_hamiltonian,
    nn_interaction_matrix,
    randn_interaction_matrix,
)
from emu_sv.state_vector import StateVector
from emu_sv.hamiltonians import FusedRydbergHamiltonian, RydbergHamiltonian
from emu_sv.hamiltonians import fused_rydberg_hamiltonian as fused
from emu_sv.time_evolution import EvolveStateVector, Lanczos
from emu_base.math.krylov_exp import DEFAULT_MAX_KRYLOV_DIM

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
    # neither the exponentiator nor the Hamiltonian type are used
    stepper = EvolveStateVector(Lanczos(), DEFAULT_MAX_KRYLOV_DIM, RydbergHamiltonian)
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


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("N, with_phase", [(3, False), (5, True), (6, False)])
def test_spectral_bounds_and_shifted(N: int, with_phase: bool) -> None:
    torch.manual_seed(1337)
    omegas = torch.randn(N, dtype=torch.float64)
    deltas = torch.randn(N, dtype=torch.float64)
    phis = torch.randn(N, dtype=torch.float64) if with_phase else torch.zeros(N)
    interaction_matrix = randn_interaction_matrix(N)
    ham = RydbergHamiltonian(
        omegas=omegas,
        deltas=deltas,
        phis=phis,
        interaction_matrix=interaction_matrix,
        device=device,
        noise=None,
    )
    dense = dense_rydberg_hamiltonian(omegas, deltas, phis, interaction_matrix).to(dtype)

    d, lower, upper, offdiag_bound, im_lower, im_upper = ham.spectral_bounds()
    assert torch.allclose(d, torch.diagonal(dense), atol=1e-12)
    assert offdiag_bound == pytest.approx(omegas.abs().sum().item() / 2)
    assert (im_lower, im_upper) == (0.0, 0.0)  # no noise
    eigvals = torch.linalg.eigvalsh(dense)
    assert lower <= eigvals.min().item() and eigvals.max().item() <= upper

    shift = 0.7
    state = torch.randn(2**N, dtype=dtype)
    expected = (dense - shift * torch.eye(2**N, dtype=dtype)) @ state
    assert torch.allclose(ham.shifted(shift) * state, expected, atol=1e-12)
    # the diagonal from spectral_bounds is handed over, shifted in place
    shifted = ham.shifted(shift, d)
    assert shifted._diag is d
    assert torch.allclose(shifted * state, expected, atol=1e-12)
    # and H itself is unchanged
    assert torch.allclose(ham * state, dense @ state, atol=1e-12)


def test_spectral_bounds_of_diagonal_hamiltonian() -> None:
    N = 3
    ham = RydbergHamiltonian(
        omegas=torch.zeros(N, dtype=torch.float64),
        deltas=torch.randn(N, dtype=torch.float64),
        phis=torch.zeros(N, dtype=torch.float64),
        interaction_matrix=randn_interaction_matrix(N),
        device=device,
        noise=None,
    )
    d, lower, upper, offdiag_bound, _, _ = ham.spectral_bounds()
    assert offdiag_bound == 0.0
    assert (lower, upper) == (d.real.min().item(), d.real.max().item())


@pytest.mark.usefixtures("restore_rng_state")
def test_spectral_bounds_and_shifted_noise() -> None:
    """
    With jump noise H = H₀ - iΓ: the real bounds still enclose the Hermitian part,
    the imaginary bounds are the exact spectrum extrema of -Γ, and `shifted` accepts
    a complex shift (the imaginary part goes into the noise term).
    """
    torch.manual_seed(1337)
    N = 4
    omegas = torch.randn(N, dtype=torch.float64)
    deltas = torch.randn(N, dtype=torch.float64)
    phis = torch.randn(N, dtype=torch.float64)
    interaction_matrix = randn_interaction_matrix(N)
    # a non-diagonal L†L, so the noise has off-diagonal entries
    lindblad = 0.6 * torch.tensor([[0.3, 1.0], [0.1j, -0.2]], dtype=dtype)
    noise = -0.5j * lindblad.mH @ lindblad
    ham = RydbergHamiltonian(
        omegas=omegas,
        deltas=deltas,
        phis=phis,
        interaction_matrix=interaction_matrix,
        device=device,
        noise=noise,
    )
    dense = dense_rydberg_hamiltonian(
        omegas, deltas, phis, interaction_matrix, noise=noise
    ).to(dtype)

    d, lower, upper, offdiag_bound, im_lower, im_upper = ham.spectral_bounds()
    assert torch.allclose(d, torch.diagonal(dense), atol=1e-12)
    assert offdiag_bound == pytest.approx(omegas.abs().sum().item() / 2)
    # the Hermitian part of H is the noiseless Hamiltonian: real bounds unchanged
    eig_re = torch.linalg.eigvalsh((dense + dense.mH) / 2)
    assert lower <= eig_re.min().item() and eig_re.max().item() <= upper
    # the skew part is -Γ with Γ = ∑ⱼ gⱼ: the imaginary bounds are its exact extrema
    eig_im = torch.linalg.eigvalsh((dense - dense.mH) / 2j)
    assert im_lower == pytest.approx(eig_im.min().item())
    assert im_upper == pytest.approx(eig_im.max().item())
    assert im_upper < 0.0  # Γ is positive definite here

    shift = 0.7 + 1j * (im_lower + im_upper) / 2
    state = torch.randn(2**N, dtype=dtype)
    expected = (dense - shift * torch.eye(2**N, dtype=dtype)) @ state
    assert torch.allclose(ham.shifted(shift) * state, expected, atol=1e-12)
    shifted = ham.shifted(shift, d)
    assert shifted._diag is d
    assert not shifted.interaction_diag.is_complex()
    assert torch.allclose(shifted * state, expected, atol=1e-12)
    # and H itself is unchanged
    assert torch.allclose(ham * state, dense @ state, atol=1e-12)


def test_kernel_probe_failure_falls_back() -> None:
    """Triton can import fine and still fail to compile, e.g. without a C compiler."""
    launcher = MagicMock(side_effect=RuntimeError("no C compiler"))
    # the "emulators" logger does not propagate, so caplog does not see it
    with (
        patch.object(fused, "_add_kernel", {(1,): launcher}, create=True),
        patch.object(logging.getLogger("emulators"), "warning") as warning,
    ):
        assert not fused._kernels_work.__wrapped__(torch.device("cpu"))
    launcher.assert_called_once()
    assert "no C compiler" in warning.call_args.args[0]


@pytest.mark.skipif(
    not torch.cuda.is_available() or fused.triton is None,
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
    params = dict(
        omegas=omegas,
        deltas=deltas,
        phis=phis,
        interaction_matrix=interaction_matrix,
        device="cuda",
        noise=noise,
    )
    assert FusedRydbergHamiltonian.is_available(torch.device("cuda"), N)
    ham = FusedRydbergHamiltonian(**params)
    reference_ham = RydbergHamiltonian(**params)
    state = torch.randn(2**N, dtype=dtype, device="cuda")
    assert ham._use_kernel(state.requires_grad)
    fused = ham * state
    fused_op = ham.make_op(-0.3j)(state)
    # the kernel copies non-contiguous input instead of reading it as contiguous
    strided = torch.randn(2 * 2**N, dtype=dtype, device="cuda")[::2]
    fused_strided = ham.make_op(-0.3j)(strided)
    reference = reference_ham * state
    reference_strided = reference_ham * strided
    assert torch.allclose(fused, reference, atol=1e-12)
    assert torch.allclose(fused_op, -0.3j * reference, atol=1e-12)
    d, *bounds = ham.spectral_bounds()
    reference_d, *reference_bounds = reference_ham.spectral_bounds()
    assert d.dtype == (torch.complex128 if with_noise else torch.float64)
    assert torch.allclose(d, reference_d if with_noise else reference_d.real, atol=1e-12)
    assert bounds == pytest.approx(reference_bounds)
    shift = 0.7 - 0.2j if with_noise else 0.7
    shifted = ham.shifted(shift, d)
    assert type(shifted) is FusedRydbergHamiltonian
    assert shifted._kernel_diag is d and shifted._diag is None
    assert shifted._kernel_buffer is ham._kernel_buffer
    assert torch.allclose(shifted * state, reference - shift * state, atol=1e-12)
    assert torch.allclose(fused_strided, -0.3j * reference_strided, atol=1e-12)

    # the kernel is not differentiable: autograd falls back to torch
    ham_grad = FusedRydbergHamiltonian(
        **{**params, "omegas": omegas.clone().requires_grad_()}
    )
    assert not ham_grad._use_kernel(state.requires_grad)
    with torch.no_grad():
        assert ham_grad._use_kernel(state.requires_grad)
