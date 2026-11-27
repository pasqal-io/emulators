import torch
from typing import Type
from unittest.mock import patch
import pytest
from test.utils_testing import (
    dense_rydberg_hamiltonian,
    randn_interaction_matrix,
)
from emu_base.math.krylov_exp import DEFAULT_MAX_KRYLOV_DIM, krylov_exp_impl
from emu_sv.hamiltonians import RydbergHamiltonian, FusedRydbergHamiltonian
from emu_sv.time_evolution import Chebyshev, EvolveStateVector, Exponentiation, Lanczos

# to test locally on GPU just change device here
device = "cpu"

# forward step with either algorithm; the backward pass is Lanczos-based in both cases
with_exponentiators = pytest.mark.parametrize(
    "exponentiator",
    [pytest.param(Lanczos(), id="lanczos"), pytest.param(Chebyshev(), id="chebyshev")],
)


def do_dense_time_step(
    dt: float,
    omegas: torch.Tensor,
    deltas: torch.Tensor,
    phis: torch.Tensor,
    interaction_matrix: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    H = dense_rydberg_hamiltonian(omegas, deltas, phis, interaction_matrix).to(
        state.device
    )
    return torch.linalg.matrix_exp(-1j * dt * H) @ state


def get_randn_ham_params(
    nqubits: int, with_phase: bool = False, dtype: torch.dtype = torch.float64, **kwargs
) -> tuple[torch.Tensor]:
    omegas = torch.randn(nqubits, dtype=dtype, **kwargs)
    deltas = torch.randn(nqubits, dtype=dtype, **kwargs)
    if with_phase:
        phis = torch.randn(nqubits, dtype=dtype, **kwargs)
    else:
        phis = torch.zeros(nqubits, dtype=dtype)
    interaction = randn_interaction_matrix(nqubits, **kwargs)
    return omegas, deltas, phis, interaction


def get_randn_state(
    nqubits: int, dtype: torch.dtype = torch.complex128, **kwargs
) -> torch.Tensor:
    state = torch.randn(2**nqubits, dtype=dtype, **kwargs)
    return state / state.norm()


@with_exponentiators
@pytest.mark.parametrize("requires_grad", [True, False])
def test_forward_with_requires_grad(requires_grad, exponentiator: Exponentiation):
    """test that index_add is called in a no_grad context in forward"""
    ham_params = get_randn_ham_params(
        1, with_phase=False, requires_grad=not requires_grad
    )
    state_in = get_randn_state(1, device=device, requires_grad=requires_grad)
    state_out, _ = EvolveStateVector(
        exponentiator, DEFAULT_MAX_KRYLOV_DIM, RydbergHamiltonian
    ).apply(0.3, *ham_params, state_in, 1e-5, None)
    assert state_out.requires_grad


@with_exponentiators
@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize(
    "N, tolerance, with_phase",
    [
        (n, tol, wp)
        for n in [3, 5, 8]
        for tol in [1e-8, 1e-10, 1e-12]
        for wp in [False, True]
    ],
)
def test_forward_accuracy(
    N: int, tolerance: float, with_phase: bool, exponentiator: Exponentiation
) -> None:
    torch.manual_seed(1337)
    ham_params = get_randn_ham_params(N, with_phase=with_phase)
    state_in = get_randn_state(N, device=device)
    dt = 1.0  # 1 μs big time step

    expected = do_dense_time_step(dt, *ham_params, state_in)
    state_out, _ = EvolveStateVector(
        exponentiator, DEFAULT_MAX_KRYLOV_DIM, RydbergHamiltonian
    ).apply(dt, *ham_params, state_in, tolerance, None)
    assert torch.allclose(expected, state_out, atol=tolerance)


@with_exponentiators
@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize(
    "N, tolerance",
    [(n, tol) for n in [3, 5, 8] for tol in [1e-8, 1e-10, 1e-12]],
)
def test_backward_accuracy(N, tolerance, exponentiator: Exponentiation):
    torch.manual_seed(1337)
    ham_params = get_randn_ham_params(N, with_phase=True, requires_grad=True)
    state_in = get_randn_state(N, device=device, requires_grad=True)
    dt = 1.0  # big timestep 1 μs

    # arbitrary vector to construct a scalar
    r = torch.randn(2**N, dtype=state_in.dtype, device=state_in.device)
    r *= 0.71 / r.norm()

    state_out, _ = EvolveStateVector(
        exponentiator, DEFAULT_MAX_KRYLOV_DIM, RydbergHamiltonian
    ).apply(dt, *ham_params, state_in, tolerance, None)
    scalar = torch.vdot(r, state_out).real
    grads = torch.autograd.grad(scalar, (*ham_params, state_in))

    expected = do_dense_time_step(dt, *ham_params, state_in)
    expected_scalar = torch.vdot(r, expected).real
    expected_grads = torch.autograd.grad(expected_scalar, (*ham_params, state_in))

    for g, eg in zip(grads, expected_grads):
        assert torch.allclose(g, eg, rtol=tolerance)


@pytest.mark.parametrize(
    "hamiltonian_type", [RydbergHamiltonian, FusedRydbergHamiltonian]
)
@with_exponentiators
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("N", [8, 9])
def test_forward_backward_accuracy_cuda(
    N: int, exponentiator: Exponentiation, hamiltonian_type: Type
) -> None:
    """On CUDA with Triton, both passes apply H with the fused kernel."""
    torch.manual_seed(1337)
    tolerance = 1e-10
    ham_params = get_randn_ham_params(N, with_phase=True, requires_grad=True)
    state_in = get_randn_state(N, device="cuda", requires_grad=True)
    dt = 1.0
    r = torch.randn(2**N, dtype=state_in.dtype, device=state_in.device)
    r *= 0.71 / r.norm()

    state_out, _ = EvolveStateVector(
        exponentiator, DEFAULT_MAX_KRYLOV_DIM, hamiltonian_type
    ).apply(dt, *ham_params, state_in, tolerance, None)
    grads = torch.autograd.grad(torch.vdot(r, state_out).real, (*ham_params, state_in))

    expected = do_dense_time_step(dt, *ham_params, state_in)
    expected_grads = torch.autograd.grad(
        torch.vdot(r, expected).real, (*ham_params, state_in)
    )
    assert torch.allclose(expected, state_out, atol=tolerance)
    for g, eg in zip(grads, expected_grads):
        assert torch.allclose(g, eg, rtol=tolerance)


@with_exponentiators
@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("with_phase", [False, True])
@pytest.mark.parametrize("dt", [0.01, 1.0, 10.0])
def test_forward_accuracy_dt(
    dt: float, with_phase: bool, exponentiator: Exponentiation
) -> None:
    torch.manual_seed(1337)
    N, tolerance = 5, 1e-10
    ham_params = get_randn_ham_params(N, with_phase=with_phase)
    state_in = get_randn_state(N, device=device)

    expected = do_dense_time_step(dt, *ham_params, state_in)
    stepper = EvolveStateVector(exponentiator, DEFAULT_MAX_KRYLOV_DIM, RydbergHamiltonian)
    state_out, _ = stepper.apply(dt, *ham_params, state_in, tolerance, None)
    assert torch.allclose(expected, state_out, atol=tolerance)


@with_exponentiators
def test_diagonal_hamiltonian(exponentiator: Exponentiation) -> None:
    """Omega = 0: H is diagonal and the step is a phase per basis state"""
    N, dt = 3, 0.7
    omegas, deltas, phis, interaction = get_randn_ham_params(N)
    omegas = torch.zeros_like(omegas)
    state_in = get_randn_state(N, device=device)

    expected = do_dense_time_step(dt, omegas, deltas, phis, interaction, state_in)
    stepper = EvolveStateVector(exponentiator, DEFAULT_MAX_KRYLOV_DIM, RydbergHamiltonian)
    state_out, _ = stepper.apply(
        dt, omegas, deltas, phis, interaction, state_in, 1e-10, None
    )
    assert torch.allclose(expected, state_out, atol=1e-12)


# ---- non-Hermitian H - iΓ (Monte Carlo jump noise) ----

import math  # noqa: E402

from emu_base.jump_lindblad_operators import compute_noise_from_lindbladians  # noqa: E402
from emu_sv.hamiltonians import RydbergLindbladian  # noqa: E402
from emu_sv.time_evolution.exponentiation import CHEBYSHEV_MAX_LOG_GROWTH  # noqa: E402


def make_noise(kind: str, rate: float = 0.5) -> torch.Tensor:
    """The single-qubit noise term -½i∑L†L for a few representative channels."""
    c128 = torch.complex128
    if kind == "relaxation":  # L†L diagonal, not proportional to the identity
        op = torch.zeros(2, 2, dtype=c128)
        op[0, 1] = math.sqrt(rate)
        ops = [op]
    elif kind == "dephasing":  # L†L proportional to the identity: rho stays 1
        ops = [math.sqrt(rate / 2) * torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=c128)]
    else:  # eff_noise with a non-diagonal L†L: noise off-diagonals, complex shift
        assert kind == "eff_noise"
        ops = [math.sqrt(rate) * torch.tensor([[0.3, 1.0], [0.1j, -0.2]], dtype=c128)]
    return compute_noise_from_lindbladians(ops)


def make_noisy_hamiltonian(
    ham_params: tuple[torch.Tensor, ...], noise: torch.Tensor
) -> RydbergHamiltonian:
    omegas, deltas, phis, interaction = ham_params
    return RydbergHamiltonian(
        omegas=omegas,
        deltas=deltas,
        phis=phis,
        interaction_matrix=interaction,
        device=torch.device(device),
        noise=noise,
    )


@with_exponentiators
@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("noise_kind", ["relaxation", "dephasing", "eff_noise"])
@pytest.mark.parametrize("dt", [1.0, -0.3])
def test_forward_accuracy_noise(
    dt: float, noise_kind: str, exponentiator: Exponentiation
) -> None:
    """
    exp(-i dt (H - iΓ)) for the non-Hermitian effective Hamiltonian of the Monte
    Carlo stepper, forward and backward in time (the jump root finder steps back).
    """
    torch.manual_seed(1337)
    N, tolerance = 5, 1e-10
    ham_params = get_randn_ham_params(N, with_phase=True)
    noise = make_noise(noise_kind)
    state_in = get_randn_state(N, device=device)

    h = dense_rydberg_hamiltonian(*ham_params, noise=noise).to(state_in.device)
    expected = torch.linalg.matrix_exp(-1j * dt * h) @ state_in
    ham = make_noisy_hamiltonian(ham_params, noise)
    state_out = exponentiator(ham, dt, state_in, tolerance)
    assert torch.allclose(expected, state_out, atol=10 * tolerance)


@with_exponentiators
@pytest.mark.usefixtures("restore_rng_state")
def test_diagonal_hamiltonian_noise(exponentiator: Exponentiation) -> None:
    """Omega = 0 with a diagonal L†L: H - iΓ is diagonal, the step is exact."""
    torch.manual_seed(1337)
    N, dt = 3, 0.7
    omegas, deltas, phis, interaction = get_randn_ham_params(N)
    omegas = torch.zeros_like(omegas)
    noise = make_noise("relaxation")
    state_in = get_randn_state(N, device=device)

    h = dense_rydberg_hamiltonian(omegas, deltas, phis, interaction, noise=noise)
    expected = torch.linalg.matrix_exp(-1j * dt * h.to(state_in.device)) @ state_in
    ham = make_noisy_hamiltonian((omegas, deltas, phis, interaction), noise)
    state_out = exponentiator(ham, dt, state_in, 1e-10)
    assert torch.allclose(expected, state_out, atol=1e-12)


@pytest.mark.usefixtures("restore_rng_state")
def test_chebyshev_substeps_strong_noise() -> None:
    """
    Decay strong enough that the per-substep cap on |α| ln ρ (roundoff growth of the
    Chebyshev recurrence off the real axis) forces several substeps.
    """
    torch.manual_seed(1337)
    N, dt, tolerance = 5, 1.0, 1e-10
    omegas, deltas, phis, interaction = get_randn_ham_params(N, with_phase=True)
    omegas, deltas, interaction = 40 * omegas, 40 * deltas, 40 * interaction
    noise = make_noise("relaxation", rate=30.0)
    state_in = get_randn_state(N, device=device)
    ham = make_noisy_hamiltonian((omegas, deltas, phis, interaction), noise)

    # the scenario must exercise the growth cap, not just CHEBYSHEV_MAX_ALPHA
    _, lower, upper, _, im_lower, im_upper = ham.spectral_bounds()
    radius = (upper - lower) / 2
    epsilon = (im_upper - im_lower) / (2 * radius)
    s = epsilon * (epsilon + math.sqrt(epsilon * epsilon + 4.0)) / 2
    rho = math.sqrt(1.0 + s) + math.sqrt(s)
    assert radius * dt * math.log(rho) > CHEBYSHEV_MAX_LOG_GROWTH

    h = dense_rydberg_hamiltonian(omegas, deltas, phis, interaction, noise=noise)
    expected = torch.linalg.matrix_exp(-1j * dt * h.to(state_in.device)) @ state_in
    state_out = Chebyshev()(ham, dt, state_in, tolerance)
    assert torch.allclose(expected, state_out, atol=10 * tolerance)


def test_chebyshev_rejects_lindbladian() -> None:
    """Density-matrix evolution stays Lanczos-only."""
    nqubits = 1
    ones = torch.ones(nqubits, dtype=torch.float64)
    lindblad = torch.zeros(2, 2, dtype=torch.complex128)
    lindblad[0, 1] = 0.1
    lindbladian = RydbergLindbladian(
        omegas=ones,
        deltas=ones,
        phis=torch.zeros(nqubits, dtype=torch.float64),
        pulser_lindblads=[lindblad],
        interaction_matrix=torch.zeros(nqubits, nqubits, dtype=torch.float64),
        device=torch.device(device),
    )
    state = torch.eye(2, dtype=torch.complex128).view(-1) / math.sqrt(2)
    with pytest.raises(ValueError, match="density-matrix"):
        Chebyshev()(lindbladian, 0.1, state, 1e-10)


@pytest.mark.parametrize("dt", [0.7, -0.3, 40.0])
def test_chebyshev_no_drive_offdiagonal_noise(dt: float) -> None:
    """
    Omega = Delta = U = 0 with a non-diagonal L†L: the real part of the numerical
    range is a single point (zero real half-width), so H - c is anti-Hermitian and is
    scaled by an imaginary number. dt = 40 forces substeps.
    """
    N = 3
    zeros = torch.zeros(N, dtype=torch.float64)
    interaction = torch.zeros(N, N, dtype=torch.float64)
    noise = make_noise("eff_noise")
    state_in = get_randn_state(N, device=device)

    h = dense_rydberg_hamiltonian(zeros, zeros, zeros, interaction, noise=noise)
    expected = torch.linalg.matrix_exp(-1j * dt * h.to(state_in.device)) @ state_in
    ham = make_noisy_hamiltonian((zeros, zeros, zeros, interaction), noise)
    state_out = Chebyshev()(ham, dt, state_in, 1e-10)
    assert torch.allclose(expected, state_out, atol=1e-9)


def test_chebyshev_tiny_drive_offdiagonal_noise() -> None:
    """
    A tiny drive next to off-diagonal noise: ε = (noise half-width)/r is huge, and
    so is ρ, whose power ρ^start in the coefficient truncation must not overflow.
    """
    N, dt = 3, 0.7
    omegas = torch.full((N,), 1e-6, dtype=torch.float64)
    zeros = torch.zeros(N, dtype=torch.float64)
    interaction = torch.zeros(N, N, dtype=torch.float64)
    noise = make_noise("eff_noise")
    state_in = get_randn_state(N, device=device)

    h = dense_rydberg_hamiltonian(omegas, zeros, zeros, interaction, noise=noise)
    expected = torch.linalg.matrix_exp(-1j * dt * h.to(state_in.device)) @ state_in
    ham = make_noisy_hamiltonian((omegas, zeros, zeros, interaction), noise)
    state_out = Chebyshev()(ham, dt, state_in, 1e-10)
    assert torch.allclose(expected, state_out, atol=1e-9)


@pytest.mark.parametrize("max_krylov_dim", [8, 100])
def test_lanczos_substeps_large_step(max_krylov_dim: int) -> None:
    """A step whose Krylov iteration count would exceed the cap is split a priori
    into sub-steps, and stays accurate."""
    torch.manual_seed(1337)
    nqubits = 4
    omegas, deltas, phis, interaction = get_randn_ham_params(nqubits, with_phase=True)
    omegas *= 10.0  # a wide spectrum
    state_in = get_randn_state(nqubits)
    dt = 2.0
    tolerance = 1e-10
    ham = RydbergHamiltonian(
        omegas, deltas, phis, interaction, torch.device(device), noise=None
    )
    lanczos = Lanczos(max_krylov_dim)

    substeps = lanczos.a_priori_substeps(ham, dt, tolerance, max_krylov_dim)
    assert substeps > 1
    assert substeps >= lanczos.a_priori_substeps(ham, dt, tolerance, 100)
    assert lanczos.a_priori_substeps(ham, dt, tolerance, 1000) == 1

    state_out = lanczos(ham, dt, state_in.clone(), tolerance)
    expected = do_dense_time_step(dt, omegas, deltas, phis, interaction, state_in)
    assert torch.allclose(state_out, expected, atol=1e-9, rtol=0)


def test_lanczos_memory_cap() -> None:
    """The Krylov dimension is capped by the memory available on the device."""
    torch.manual_seed(1337)
    nqubits = 4
    omegas, deltas, phis, interaction = get_randn_ham_params(nqubits)
    state_in = get_randn_state(nqubits)
    ham = RydbergHamiltonian(
        omegas, deltas, phis, interaction, torch.device(device), noise=None
    )
    lanczos = Lanczos(100)
    vector_bytes = state_in.numel() * state_in.element_size()
    module = "emu_sv.time_evolution.exponentiation"

    # room for 6 basis vectors + KRYLOV_EXTRA_VECTORS in LANCZOS_MEMORY_FRACTION
    with patch(f"{module}.available_memory", return_value=8 * vector_bytes / 0.8):
        assert lanczos.krylov_cap(ham, state_in) == 6
        state_out = lanczos(ham, 1.0, state_in.clone(), 1e-10)
    expected = do_dense_time_step(1.0, omegas, deltas, phis, interaction, state_in)
    assert torch.allclose(state_out, expected, atol=1e-9, rtol=0)

    with (
        patch(f"{module}.available_memory", return_value=3 * vector_bytes / 0.8),
        pytest.raises(MemoryError),
    ):
        lanczos.krylov_cap(ham, state_in)

    with patch(f"{module}.available_memory", return_value=None):
        assert lanczos.krylov_cap(ham, state_in) == 100


@pytest.mark.parametrize("failure", ["non-convergence", "oom"])
def test_lanczos_retries_with_halved_step(failure: str) -> None:
    """A sub-step that does not converge, or runs out of GPU memory, is halved."""
    torch.manual_seed(1337)
    nqubits = 3
    omegas, deltas, phis, interaction = get_randn_ham_params(nqubits)
    state_in = get_randn_state(nqubits)
    ham = RydbergHamiltonian(
        omegas, deltas, phis, interaction, torch.device(device), noise=None
    )
    dts = []

    def failing_once(op, v, **kwargs):
        if not dts and failure == "oom":
            dts.append(None)
            raise torch.cuda.OutOfMemoryError("simulated")
        if not dts:  # too small a cap for the first attempt
            kwargs["max_krylov_dim"] = 2
        result = krylov_exp_impl(op, v, **kwargs)
        dts.append(result.converged)
        return result

    with patch(
        "emu_sv.time_evolution.exponentiation.krylov_exp_impl", side_effect=failing_once
    ) as spy:
        state_out = Lanczos(100)(ham, 1.0, state_in.clone(), 1e-10)

    expected = do_dense_time_step(1.0, omegas, deltas, phis, interaction, state_in)
    assert torch.allclose(state_out, expected, atol=1e-9, rtol=0)
    # the failed attempt, then the two halves
    assert spy.call_count == 3
    assert dts[1:] == [True, True]


def test_lanczos_gives_up_after_max_splits() -> None:
    torch.manual_seed(1337)
    nqubits = 3
    omegas, deltas, phis, interaction = get_randn_ham_params(nqubits)
    state_in = get_randn_state(nqubits)
    ham = RydbergHamiltonian(
        omegas, deltas, phis, interaction, torch.device(device), noise=None
    )

    def never_converges(op, v, **kwargs):
        kwargs["max_krylov_dim"] = 1
        return krylov_exp_impl(op, v, **kwargs)

    with (
        patch(
            "emu_sv.time_evolution.exponentiation.krylov_exp_impl",
            side_effect=never_converges,
        ),
        pytest.raises(RecursionError, match="halving"),
    ):
        Lanczos(100)(ham, 1.0, state_in.clone(), 1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the CUDA allocator")
@pytest.mark.parametrize("cap", [3, 6])
def test_lanczos_memory_accounting(cap: int) -> None:
    """A Krylov call holds at most cap + max(KRYLOV_EXTRA_VECTORS, working_vectors)
    state-sized tensors beyond its input, and the operators their `working_vectors`,
    as `Lanczos.krylov_cap` assumes. On top of that, cuBLAS may take a workspace
    (a few MB, independent of the state) from the allocator for the duration of
    each dot product; it is measured and allowed for, on vectors large enough for
    it to be a fraction of one."""
    from emu_base.math.krylov_exp import KRYLOV_EXTRA_VECTORS
    from emu_sv.hamiltonians import RydbergLindbladian

    torch.manual_seed(1337)
    cuda = torch.device("cuda")

    def peak_vectors(fn, reference: torch.Tensor) -> float:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        fn()
        torch.cuda.synchronize()
        return (torch.cuda.max_memory_allocated() - base) / (
            reference.numel() * reference.element_size()
        )

    nqubits = 20  # 16 MB vectors; the density matrix below has the same size
    omegas, deltas, phis, interaction = get_randn_ham_params(nqubits, with_phase=True)
    state = get_randn_state(nqubits).to(cuda)
    noise = -0.5j * torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.complex128)
    lindblad = [torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.complex128)]
    rho = torch.randn(2**10, 2**10, dtype=torch.complex128, device=cuda)
    cases = [
        (RydbergHamiltonian(omegas, deltas, phis, interaction, cuda, noise=None), state),
        (RydbergHamiltonian(omegas, deltas, phis, interaction, cuda, noise=noise), state),
        (
            FusedRydbergHamiltonian(omegas, deltas, phis, interaction, cuda, noise=None),
            state,
        ),
        (
            RydbergLindbladian(
                omegas=omegas[:10],
                deltas=deltas[:10],
                phis=phis[:10],
                pulser_lindblads=lindblad,
                interaction_matrix=interaction[:10, :10],
                device=cuda,
            ),
            rho,
        ),
    ]
    for ham, vector in cases:
        op = ham.make_op(-1j * 0.1)
        hermitian = isinstance(ham, RydbergHamiltonian) and ham.noise is None
        # warm up: lazily cached diagonals and anything a library keeps around
        krylov_exp_impl(op, vector.clone(), hermitian, 1e-14, 1e-14, max_krylov_dim=2)
        workspace = peak_vectors(
            lambda: torch.tensordot(vector.conj(), vector, dims=vector.dim()), vector
        )
        assert workspace < 1.0
        op_peak = peak_vectors(lambda: op(vector), vector)
        assert op_peak == pytest.approx(ham.working_vectors, abs=workspace + 0.01)
        for _ in range(2):  # the second call converges within the cap, or not
            result = peak_vectors(
                lambda: krylov_exp_impl(
                    op,
                    vector.clone(),
                    is_hermitian=hermitian,
                    exp_tolerance=1e-14,
                    norm_tolerance=1e-14,
                    max_krylov_dim=cap,
                ),
                vector,
            )
            bound = 1 + cap + max(KRYLOV_EXTRA_VECTORS, ham.working_vectors)
            assert result <= bound + workspace + 0.01
