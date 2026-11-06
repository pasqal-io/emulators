import cmath
import inspect
import math
from unittest.mock import patch

import torch
import pytest
from test.utils_testing import (
    EPS,
    chebyshev_step_budget,
    dense_rydberg_hamiltonian,
    matrix_exp_error,
    randn_interaction_matrix,
    requires_kernel,
)
from emu_base.math.chebyshev import chebyshev_exp, exp_chebyshev_coefficients
from emu_base.math.double_krylov import double_krylov
from emu_base.math.krylov_exp import krylov_exp
from emu_sv import hamiltonian_kernels as kernels
from emu_sv.hamiltonian import RydbergHamiltonian, make_interaction_diagonal
from emu_sv.time_evolution import (
    EvolveDensityMatrix,
    EvolveMonteCarlo,
    EvolveStateVector,
)
from emu_sv.time_evolution.torch import DifferentiableStateVectorEvolution

# to test locally on GPU just change device here
device = "cpu"


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


@pytest.mark.parametrize("requires_grad", [True, False])
def test_forward_with_requires_grad(requires_grad):
    """test that index_add is called in a no_grad context in forward"""
    ham_params = get_randn_ham_params(
        1, with_phase=False, requires_grad=not requires_grad
    )
    state_in = get_randn_state(1, device=device, requires_grad=requires_grad)
    state_out, _ = EvolveStateVector().apply(0.3, *ham_params, state_in, 1e-5, None)
    assert state_out.requires_grad


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
def test_forward_accuracy(N: int, tolerance: float, with_phase: bool) -> None:
    torch.manual_seed(1337)
    ham_params = get_randn_ham_params(N, with_phase=with_phase)
    state_in = get_randn_state(N, device=device)
    dt = 1.0  # 1 μs big time step

    expected = do_dense_time_step(dt, *ham_params, state_in)
    state_out, _ = EvolveStateVector().apply(dt, *ham_params, state_in, tolerance, None)
    assert torch.allclose(expected, state_out, atol=tolerance)


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize(
    "N, tolerance",
    [(n, tol) for n in [3, 5, 8] for tol in [1e-8, 1e-10, 1e-12]],
)
def test_backward_accuracy(N, tolerance):
    torch.manual_seed(1337)
    ham_params = get_randn_ham_params(N, with_phase=True, requires_grad=True)
    state_in = get_randn_state(N, device=device, requires_grad=True)
    dt = 1.0  # big timestep 1 μs

    # arbitrary vector to construct a scalar
    r = torch.randn(2**N, dtype=state_in.dtype, device=state_in.device)
    r *= 0.71 / r.norm()

    state_out, _ = EvolveStateVector().apply(dt, *ham_params, state_in, tolerance, None)
    scalar = torch.vdot(r, state_out).real
    grads = torch.autograd.grad(scalar, (*ham_params, state_in))

    expected = do_dense_time_step(dt, *ham_params, state_in)
    expected_scalar = torch.vdot(r, expected).real
    expected_grads = torch.autograd.grad(expected_scalar, (*ham_params, state_in))

    for g, eg in zip(grads, expected_grads):
        assert torch.allclose(g, eg, rtol=tolerance)


# Chebyshev steps (emu_sv.time_evolution.torch): on CPU they are forced through the
# last argument of DifferentiableStateVectorEvolution.apply; on a GPU with the fused
# kernel, EvolveStateVector selects them by default.

dtype = torch.complex128
# tolerance of the Chebyshev series relative to krylov_tolerance (SVConfig docstring)
CHEBYSHEV_TOLERANCE_FACTOR = 1e-2
CHEBYSHEV = "emu_sv.time_evolution.torch.chebyshev_exp"
KRYLOV = "emu_sv.time_evolution.torch.krylov_exp"


def chebyshev_call_args(call) -> dict:
    """Arguments of one recorded call of chebyshev_exp, by name."""
    return inspect.signature(chebyshev_exp).bind(*call.args, **call.kwargs).arguments


def series_applications(spy) -> int:
    """Number of H products of the recorded chebyshev_exp calls: K per call (see the
    chebyshev_exp docstring)."""
    calls = (chebyshev_call_args(call) for call in spy.call_args_list)
    return sum(
        len(exp_chebyshev_coefficients(a["alpha"], a["tolerance"])) - 1 for a in calls
    )


def off_centre_params(nqubits: int, with_phase: bool) -> tuple[torch.Tensor, ...]:
    """Random parameters whose diagonal is not centred on 0 (shift c != 0)."""
    omegas, deltas, phis, interaction = get_randn_ham_params(nqubits, with_phase)
    return omegas, deltas + 3.0, phis, interaction


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("N", [3, 5, 8])
@pytest.mark.parametrize("tolerance", [1e-8, 1e-10, 1e-12])
@pytest.mark.parametrize("with_phase", [False, True])
@pytest.mark.parametrize("dt", [1.0, -0.7])
def test_chebyshev_step_accuracy(
    N: int, tolerance: float, with_phase: bool, dt: float
) -> None:
    torch.manual_seed(1337)
    ham_params = off_centre_params(N, with_phase)
    state_in = get_randn_state(N, device=device)
    h = dense_rydberg_hamiltonian(*ham_params)
    expected = torch.linalg.matrix_exp(-1j * dt * h) @ state_in
    # Budget: truncation tolerance of the series, rounding of the recurrence computed
    # from the oracle coefficients for alpha = r dt, matrix_exp (chebyshev_step_budget).
    budget = chebyshev_step_budget(
        h, ham_params[0], dt, tolerance * CHEBYSHEV_TOLERANCE_FACTOR
    )

    with patch(CHEBYSHEV, autospec=True, side_effect=chebyshev_exp) as cheb:
        state_out, _ = DifferentiableStateVectorEvolution.apply(
            dt, *ham_params, state_in, tolerance, None, True
        )

    # the option went through forward to evolve (Krylov would be accurate too)
    assert cheb.call_count == 1
    args = chebyshev_call_args(cheb.call_args)
    assert args["tolerance"] == pytest.approx(
        tolerance * CHEBYSHEV_TOLERANCE_FACTOR, rel=1e-12, abs=0.0
    )
    torch.testing.assert_close(state_out, expected, rtol=0.0, atol=3 * budget)


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("use_chebyshev", [False, True])
@pytest.mark.parametrize("zero_omega", [False, True])
def test_evolve_dispatch(use_chebyshev: bool, zero_omega: bool) -> None:
    """use_chebyshev=False runs Lanczos, also at Omega = 0 (shortcut only on Chebyshev)."""
    torch.manual_seed(1337)
    omegas, deltas, phis, interaction = off_centre_params(3, with_phase=True)
    if zero_omega:
        omegas[:] = 0.0
    ham_params = omegas, deltas, phis, interaction
    state_in = get_randn_state(3, device=device)

    with (
        patch(CHEBYSHEV, autospec=True, side_effect=chebyshev_exp) as cheb,
        patch(KRYLOV, autospec=True, side_effect=krylov_exp) as krylov,
    ):
        DifferentiableStateVectorEvolution.apply(
            0.5, *ham_params, state_in, 1e-10, None, use_chebyshev
        )

    assert cheb.call_count == int(use_chebyshev and not zero_omega)
    assert krylov.call_count == int(not use_chebyshev)


@pytest.mark.usefixtures("restore_rng_state")
def test_chebyshev_step_returns_unshifted_hamiltonian() -> None:
    """The shift c stays local to the step: observables use H, not H - c."""
    torch.manual_seed(1337)
    N = 4
    omegas, deltas, phis, interaction = off_centre_params(N, with_phase=True)
    interaction_diag = make_interaction_diagonal(interaction, device)
    interaction_diag_copy = interaction_diag.clone()
    state_in = get_randn_state(N, device=device)

    _, ham = DifferentiableStateVectorEvolution.apply(
        1.0, omegas, deltas, phis, interaction, state_in, 1e-10, interaction_diag, True
    )

    assert ham.interaction_diag is interaction_diag
    assert torch.equal(interaction_diag, interaction_diag_copy)
    dense = dense_rydberg_hamiltonian(omegas, deltas, phis, interaction)
    # Budget per entry: <= N^2 summed terms bounded by S max|psi| (dense and sparse);
    # atol = 4 N^2 eps S max|psi|. A shift left in the Hamiltonian moves it by |c| >= 1.
    scale = (deltas.abs().sum() + interaction.abs().sum() / 2 + omegas.abs().sum()).item()
    atol = 4 * N**2 * EPS * scale * state_in.abs().max().item()
    torch.testing.assert_close(ham * state_in, dense @ state_in, rtol=0.0, atol=atol)


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("last_omega", [0.0, None, 1e-9])
def test_chebyshev_step_zero_omega(last_omega: float | None) -> None:
    """
    With Omega = 0 on every qubit, H is diagonal and the step is exactly
    exp(-i dt d) psi: no series, no Lanczos. With one qubit driven (random Omega, or
    Omega = 1e-9, whose omission would cost ~1e-10 >> 1e-12), the series runs.
    """
    torch.manual_seed(1337)
    N, dt = 4, 1.0
    omegas, deltas, phis, interaction = off_centre_params(N, with_phase=True)
    omegas[: N - 1] = 0.0
    if last_omega is not None:
        omegas[-1] = last_omega
    zero_omega = last_omega == 0.0
    state_in = get_randn_state(N, device=device)
    state_copy = state_in.clone()
    h = dense_rydberg_hamiltonian(omegas, deltas, phis, interaction)
    expected = torch.linalg.matrix_exp(-1j * dt * h) @ state_in
    if zero_omega:
        # Budget per entry: the diagonal sums <= N + N(N-1)/2 terms, here and in the dense
        # reference: phase error <= |dt| 2 n_terms eps S; exp and product: 4 eps;
        # plus the matrix_exp estimate. atol = 3x the budget.
        n_terms = N + N * (N - 1) // 2
        scale = (deltas.abs().sum() + interaction.abs().sum() / 2).item()
        budget = abs(dt) * 2 * n_terms * EPS * scale + 4 * EPS + matrix_exp_error(h, dt)
    else:
        budget = chebyshev_step_budget(h, omegas, dt, 1e-10 * CHEBYSHEV_TOLERANCE_FACTOR)

    with (
        patch(CHEBYSHEV, autospec=True, side_effect=chebyshev_exp) as cheb,
        patch(KRYLOV, autospec=True, side_effect=krylov_exp) as krylov,
    ):
        state_out, _ = DifferentiableStateVectorEvolution.apply(
            dt, omegas, deltas, phis, interaction, state_in, 1e-10, None, True
        )

    assert cheb.call_count == int(not zero_omega)
    assert krylov.call_count == 0
    assert torch.equal(state_in, state_copy)
    assert state_out.data_ptr() != state_in.data_ptr()
    torch.testing.assert_close(state_out, expected, rtol=0.0, atol=3 * budget)


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize(
    "dt, substeps",
    [
        (500.0, 1),  # alpha = 2000 exactly: one step
        (500.0 + 2**-30, 2),  # alpha just above 2000
        (1125.0, 3),  # alpha = 4500
    ],
)
def test_chebyshev_large_alpha_is_split(dt: float, substeps: int) -> None:
    """
    Steps with alpha = r dt > 2000 are split into m = ceil(alpha / 2000) equal
    sub-steps, each with phase exp(-i dt c / m) and tolerance / m.
    Dyadic inputs make c = 2 and r = 4 exact, so alpha = 4 dt hits the threshold exactly.
    """
    torch.manual_seed(1337)
    omegas = torch.tensor([2.0, 2.0], dtype=torch.float64)  # R = 2
    deltas = torch.zeros(2, dtype=torch.float64)
    phis = torch.zeros(2, dtype=torch.float64)
    interaction = torch.tensor([[0.0, 4.0], [4.0, 0.0]], dtype=torch.float64)
    c, r, tolerance = 2.0, 4.0, 1e-10  # diagonal (0, 0, 0, 4), R = 2
    state_in = get_randn_state(2, device=device)
    h = dense_rydberg_hamiltonian(omegas, deltas, phis, interaction)
    expected = torch.linalg.matrix_exp(-1j * dt * h) @ state_in
    budget = chebyshev_step_budget(h, omegas, dt, tolerance * CHEBYSHEV_TOLERANCE_FACTOR)

    with patch(CHEBYSHEV, autospec=True, side_effect=chebyshev_exp) as cheb:
        state_out, _ = DifferentiableStateVectorEvolution.apply(
            dt, omegas, deltas, phis, interaction, state_in, tolerance, None, True
        )

    assert cheb.call_count == substeps
    # phase argument dt c / m rounded with relative eps, twice: 4 (|dt c / m| + 1) eps
    phase_atol = 4 * (abs(dt * c / substeps) + 1) * EPS
    for call in cheb.call_args_list:
        args = chebyshev_call_args(call)
        assert args["alpha"] == pytest.approx(r * dt / substeps, rel=4 * EPS, abs=0.0)
        assert args["tolerance"] == pytest.approx(
            tolerance * CHEBYSHEV_TOLERANCE_FACTOR / substeps, rel=1e-12, abs=0.0
        )
        assert complex(args["phase"]) == pytest.approx(
            cmath.exp(-1j * dt * c / substeps), rel=0.0, abs=phase_atol
        )
    torch.testing.assert_close(state_out, expected, rtol=0.0, atol=3 * budget)


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("chebyshev_on_gpu", [None, True, False])
@pytest.mark.parametrize("kernel_available", [True, False])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_evolve_state_vector_dispatch(
    chebyshev_on_gpu: bool | None, kernel_available: bool, requires_grad: bool
) -> None:
    """
    Chebyshev iff the option is on (default: on) and the fused kernel is available
    for the state (CUDA, Triton, complex128, N >= 8); the choice is the 9th argument.
    Recording autograd (inputs that require grad) does not change the choice.
    """
    torch.manual_seed(1337)
    ham_params = get_randn_ham_params(3, requires_grad=requires_grad)
    state_in = get_randn_state(3, device=device, requires_grad=requires_grad)
    if chebyshev_on_gpu is None:
        stepper = EvolveStateVector()
    else:
        stepper = EvolveStateVector(chebyshev_on_gpu=chebyshev_on_gpu)

    with (
        patch(
            "emu_sv.hamiltonian_kernels.is_available",
            autospec=True,
            return_value=kernel_available,
        ) as available,
        patch.object(
            DifferentiableStateVectorEvolution,
            "apply",
            autospec=True,
            return_value=(state_in, None),
        ) as evolution,
    ):
        stepper.apply(0.1, *ham_params, state_in, 1e-10, [])

    args = evolution.call_args.args
    assert len(args) == 9
    assert args[8] is ((chebyshev_on_gpu is not False) and kernel_available)
    if chebyshev_on_gpu is not False:
        available.assert_called_once_with(state_in)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.usefixtures("restore_rng_state")
def test_evolve_state_vector_uses_krylov_without_triton() -> None:
    """Default stepper, real availability check: CUDA without Triton -> Lanczos."""
    torch.manual_seed(1337)
    N = 8  # at MIN_QUBITS: only Triton rules out the kernel
    ham_params = get_randn_ham_params(N)
    state_in = get_randn_state(N, device="cuda")

    with (
        patch.object(kernels, "triton", None),
        patch(CHEBYSHEV, autospec=True, side_effect=chebyshev_exp) as cheb,
        patch(KRYLOV, autospec=True, side_effect=krylov_exp) as krylov,
    ):
        EvolveStateVector().apply(0.1, *ham_params, state_in, 1e-10, [])

    assert cheb.call_count == 0
    assert krylov.call_count == 1


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("stepper_type", [EvolveMonteCarlo, EvolveDensityMatrix])
def test_noisy_steppers_use_krylov(
    stepper_type: type[EvolveMonteCarlo] | type[EvolveDensityMatrix],
) -> None:
    """
    Monte Carlo and density-matrix steps keep the non-Hermitian Lanczos, even where the
    noiseless steps would use Chebyshev: the kernel availability is forced to True
    (H products kept on the torch path, as the kernel needs CUDA).
    """
    torch.manual_seed(1337)
    N = 3
    omegas, deltas, phis, interaction = get_randn_ham_params(N, with_phase=True)
    # complex parameters, as the adapter gives them
    ham_params = (omegas.to(dtype), deltas.to(dtype), phis.to(dtype), interaction)
    lindblads = [math.sqrt(0.1) * torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=dtype)]
    psi = get_randn_state(N, device=device)
    state_in = psi if stepper_type is EvolveMonteCarlo else torch.outer(psi, psi.conj())

    with (
        patch(
            "emu_sv.hamiltonian_kernels.is_available", autospec=True, return_value=True
        ),
        patch.object(
            RydbergHamiltonian, "_use_kernel", autospec=True, return_value=False
        ),
        patch(CHEBYSHEV, autospec=True, side_effect=chebyshev_exp) as cheb,
        patch(
            "emu_sv.time_evolution.implementations.krylov_exp",
            autospec=True,
            side_effect=krylov_exp,
        ) as krylov,
    ):
        stepper_type().apply(0.1, *ham_params, state_in, 1e-10, lindblads)

    assert krylov.call_count >= 1
    for call in krylov.call_args_list:
        bound = inspect.signature(krylov_exp).bind(*call.args, **call.kwargs)
        assert bound.arguments["is_hermitian"] is False
    assert cheb.call_count == 0


@pytest.mark.usefixtures("restore_rng_state")
def test_backward_uses_krylov_after_chebyshev_forward() -> None:
    torch.manual_seed(1337)
    N = 3
    ham_params = get_randn_ham_params(N, with_phase=True, requires_grad=True)
    state_in = get_randn_state(N, device=device, requires_grad=True)
    r = torch.randn(2**N, dtype=dtype)

    with (
        patch(CHEBYSHEV, autospec=True, side_effect=chebyshev_exp) as cheb,
        patch(KRYLOV, autospec=True, side_effect=krylov_exp) as krylov,
        patch(
            "emu_sv.time_evolution.torch.double_krylov",
            autospec=True,
            side_effect=double_krylov,
        ) as dkrylov,
    ):
        state_out, _ = DifferentiableStateVectorEvolution.apply(
            1.0, *ham_params, state_in, 1e-10, None, True
        )
        assert (cheb.call_count, krylov.call_count) == (1, 0)
        torch.autograd.grad(torch.vdot(r, state_out).real, (*ham_params, state_in))

    assert cheb.call_count == 1
    assert dkrylov.call_count == 1  # gradients of the parameters
    assert krylov.call_count == 1  # gradient of the input state


@pytest.mark.usefixtures("restore_rng_state")
def test_chebyshev_step_unnormalised_state() -> None:
    """
    The Chebyshev path leaves the input state untouched, so the gradients are taken at
    the state given (the Krylov path normalises it in place; not changed here).
    """
    torch.manual_seed(1337)
    N, dt, tolerance = 3, 1.0, 1e-10
    ham_params = get_randn_ham_params(N, with_phase=True, requires_grad=True)
    state_in = (0.7 * get_randn_state(N, device=device)).requires_grad_()
    state_copy = state_in.detach().clone()
    r = torch.randn(2**N, dtype=dtype)
    r *= 0.71 / r.norm()

    state_out, _ = DifferentiableStateVectorEvolution.apply(
        dt, *ham_params, state_in, tolerance, None, True
    )
    assert torch.equal(state_in.detach(), state_copy)
    grads = torch.autograd.grad(torch.vdot(r, state_out).real, (*ham_params, state_in))

    expected = do_dense_time_step(dt, *ham_params, state_in)
    expected_grads = torch.autograd.grad(
        torch.vdot(r, expected).real, (*ham_params, state_in)
    )
    h = dense_rydberg_hamiltonian(*(p.detach() for p in ham_params))
    budget = 0.7 * chebyshev_step_budget(
        h, ham_params[0].detach(), dt, tolerance * CHEBYSHEV_TOLERANCE_FACTOR
    )
    torch.testing.assert_close(
        state_out.detach(), expected.detach(), rtol=0.0, atol=3 * budget
    )
    # Gradients: same tolerances as test_backward_accuracy for a normalised state; the
    # exact gradients and the Lanczos error are both linear in ||psi|| = 0.7, so they
    # carry over. Gradients divided by ||psi|| (in-place normalisation) are 43 % off.
    for g, eg in zip(grads, expected_grads):
        torch.testing.assert_close(g, eg, rtol=tolerance, atol=1e-8)


@requires_kernel
@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("N", [8, 10])
def test_chebyshev_steps_cuda_match_dense(N: int) -> None:
    """Default stepper on CUDA with Triton: Chebyshev steps through the fused kernel."""
    torch.manual_seed(1337)
    tolerance, dt, nsteps = 1e-10, 0.5, 3
    stepper = EvolveStateVector()
    state = get_randn_state(N, device="cuda")
    expected = state.cpu()
    budget = 0.0
    with (
        patch(CHEBYSHEV, autospec=True, side_effect=chebyshev_exp) as cheb,
        patch.object(
            kernels, "apply", autospec=True, side_effect=kernels.apply
        ) as kernel,
    ):
        for _ in range(nsteps):  # a different Hamiltonian at each step
            ham_params = off_centre_params(N, with_phase=True)
            state, _ = stepper.apply(dt, *ham_params, state, tolerance, [])
            h = dense_rydberg_hamiltonian(*ham_params)
            expected = torch.linalg.matrix_exp(-1j * dt * h) @ expected
            # unitary steps: the errors of the steps add up
            budget += chebyshev_step_budget(
                h, ham_params[0], dt, tolerance * CHEBYSHEV_TOLERANCE_FACTOR
            )

    assert cheb.call_count == nsteps
    assert kernel.call_count == series_applications(cheb)  # every H product
    assert state.is_cuda
    torch.testing.assert_close(state.cpu(), expected, rtol=0.0, atol=3 * budget)


@requires_kernel
def test_chebyshev_loop_does_not_synchronize() -> None:
    """Once op has been applied once (factor and table on the device), no host sync."""
    torch.manual_seed(1337)
    N = 10
    omegas, deltas, phis, interaction = get_randn_ham_params(N, with_phase=True)
    ham = DifferentiableStateVectorEvolution.get_hamiltonian(
        omegas, deltas, phis, interaction, torch.device("cuda")
    )
    # 1/norm bound <= 1/||H||: the series argument stays in [-1, 1]
    norm_bound = (
        deltas.abs().sum() + interaction.abs().sum() / 2 + omegas.abs().sum() / 2
    ).item()
    op = ham.make_op(1.0 / norm_bound)
    state = get_randn_state(N, device="cuda")
    op(state)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        chebyshev_exp(op, state, 30.0, 1.0 + 0.0j, 1e-12)
    finally:
        torch.cuda.set_sync_debug_mode("default")


@requires_kernel
@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("N", [8, 9])
@pytest.mark.parametrize("chebyshev_on_gpu", [False, True])
def test_forward_backward_accuracy_cuda(N: int, chebyshev_on_gpu: bool) -> None:
    """
    On CUDA with Triton, both passes apply H with the fused kernel; the forward pass
    uses Chebyshev steps unless disabled, the backward pass Lanczos.
    """
    torch.manual_seed(1337)
    tolerance = 1e-10
    ham_params = get_randn_ham_params(N, with_phase=True, requires_grad=True)
    state_in = get_randn_state(N, device="cuda", requires_grad=True)
    dt = 1.0
    r = torch.randn(2**N, dtype=state_in.dtype, device=state_in.device)
    r *= 0.71 / r.norm()

    with (
        patch(CHEBYSHEV, autospec=True, side_effect=chebyshev_exp) as cheb,
        patch.object(
            kernels, "apply", autospec=True, side_effect=kernels.apply
        ) as kernel,
    ):
        state_out, _ = EvolveStateVector(chebyshev_on_gpu=chebyshev_on_gpu).apply(
            dt, *ham_params, state_in, tolerance, []
        )
        forward_kernel_calls = kernel.call_count
        grads = torch.autograd.grad(
            torch.vdot(r, state_out).real, (*ham_params, state_in)
        )
    assert cheb.call_count == int(chebyshev_on_gpu)
    if chebyshev_on_gpu:  # every H product of the series
        assert forward_kernel_calls == series_applications(cheb)
    else:
        assert forward_kernel_calls >= 1
    assert kernel.call_count > forward_kernel_calls  # backward

    expected = do_dense_time_step(dt, *ham_params, state_in)
    expected_grads = torch.autograd.grad(
        torch.vdot(r, expected).real, (*ham_params, state_in)
    )
    if chebyshev_on_gpu:
        h = dense_rydberg_hamiltonian(*(p.detach() for p in ham_params))
        atol = 3 * chebyshev_step_budget(
            h, ham_params[0].detach(), dt, tolerance * CHEBYSHEV_TOLERANCE_FACTOR
        )
        torch.testing.assert_close(
            state_out.detach(), expected.detach(), rtol=0.0, atol=atol
        )
    else:
        assert torch.allclose(expected, state_out, atol=tolerance)
    for g, eg in zip(grads, expected_grads):
        assert torch.allclose(g, eg, rtol=tolerance)
