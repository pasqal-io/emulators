import torch
from typing import Type
import pytest
from test.utils_testing import (
    dense_rydberg_hamiltonian,
    randn_interaction_matrix,
)
from emu_base.math.krylov_exp import DEFAULT_MAX_KRYLOV_DIM
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
