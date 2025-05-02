import torch
import pytest
from test.utils_testing import (
    dense_rydberg_hamiltonian,
    nn_interaction_matrix,
    randn_interaction_matrix,
)
from emu_sv.time_evolution import EvolveStateVector

dtype = torch.complex128
device = "cpu"


@pytest.mark.parametrize(
    ("N", "krylov_tolerance"),
    [(3, 1e-10), (5, 1e-12), (7, 1e-10), (8, 1e-12)],
)
def test_forward_no_phase(N: int, krylov_tolerance: float) -> None:
    torch.manual_seed(1337)
    omegas = torch.randn(N)
    deltas = torch.randn(N)
    phis = torch.zeros_like(omegas)
    interaction = nn_interaction_matrix(N)
    ham_params = (omegas, deltas, phis, interaction)

    state = torch.randn(2**N, dtype=dtype, device=device)
    state /= state.norm()

    H = dense_rydberg_hamiltonian(*ham_params).to(device)
    dt = 1.0  # 1 μs big time step
    ed = torch.linalg.matrix_exp(-1j * dt * H) @ state
    krylov, _ = EvolveStateVector.apply(
        dt,
        *ham_params,
        state,
        krylov_tolerance,
    )
    assert torch.allclose(ed, krylov, atol=krylov_tolerance)


@pytest.mark.parametrize(
    ("N", "krylov_tolerance"),
    [(3, 1e-10), (5, 1e-12), (7, 1e-10), (8, 1e-12)],
)
def test_forward_with_phase(N: int, krylov_tolerance: float) -> None:
    torch.manual_seed(1337)
    omegas, deltas, phis = torch.randn(3, N)  # unpack a 3*N tensor
    interaction = randn_interaction_matrix(N)
    ham_params = (omegas, deltas, phis, interaction)

    state = torch.randn(2**N, dtype=dtype, device=device)
    state /= state.norm()

    H = dense_rydberg_hamiltonian(*ham_params).to(device)
    dt = 1.0  # 1 μs big time step
    ed = torch.linalg.matrix_exp(-1j * dt * H) @ state
    krylov, _ = EvolveStateVector.apply(
        dt,
        *ham_params,
        state,
        krylov_tolerance,
    )
    assert torch.allclose(ed, krylov, atol=krylov_tolerance)


dtype_params = torch.float64


@pytest.mark.parametrize(
    "N, krylov_tolerance",
    [(n, tol) for n in [5, 8] for tol in [1e-8, 1e-10, 1e-12]],
)
def test_backward(N, krylov_tolerance):
    torch.manual_seed(1337)
    omegas = torch.randn(N, dtype=dtype_params, requires_grad=True)
    deltas = torch.randn(N, dtype=dtype_params, requires_grad=True)
    phis = torch.zeros(N, dtype=dtype_params)
    interactions = randn_interaction_matrix(N)

    ham_params = (omegas, deltas, phis, interactions)

    state = torch.randn(2**N, dtype=dtype, requires_grad=True)
    state = state / state.norm()
    r = torch.randn(2**N, dtype=dtype)
    r /= r.norm()

    dt = 1.0  # big timestep 1 μs

    krylov, _ = EvolveStateVector.apply(dt, *ham_params, state, krylov_tolerance)
    scalar = torch.abs(r @ krylov)
    grads = torch.autograd.grad(scalar, (omegas, deltas, state), retain_graph=True)

    h = dense_rydberg_hamiltonian(*ham_params).to(device)
    ed = torch.linalg.matrix_exp(-1j * dt * h) @ state
    ed_scalar = torch.abs(r @ ed)
    ed_grads = torch.autograd.grad(ed_scalar, (omegas, deltas, state), retain_graph=True)

    # expected tolerance for the gradients is bigger
    expected_grad_tolerance = krylov_tolerance
    for grad, ed_grad in zip(grads, ed_grads):
        assert torch.allclose(grad, ed_grad, rtol=expected_grad_tolerance)


@pytest.mark.parametrize(
    "N, krylov_tolerance",
    [(n, tol) for n in [3, 7] for tol in [1e-8, 1e-10, 1e-12]],
)
def test_backward_Op(N, krylov_tolerance):
    torch.manual_seed(1337)
    omegas = torch.randn(N, dtype=dtype_params, requires_grad=True)
    deltas = torch.randn(N, dtype=dtype_params, requires_grad=True)
    phis = torch.zeros(N, dtype=dtype_params)
    interactions = randn_interaction_matrix(N)

    ham_params = (omegas, deltas, phis, interactions)

    state = torch.randn(2**N, dtype=dtype, requires_grad=True)
    state = state / state.norm()

    dt = 1.0  # big timestep 1 μs

    Op = torch.randn(2**N, 2**N, dtype=dtype)
    Op /= Op.norm()
    Op = Op @ Op.mH  # Hermitian observable

    krylov, _ = EvolveStateVector.apply(dt, *ham_params, state, krylov_tolerance)
    scalar = torch.vdot(krylov, Op @ krylov).real
    grads = torch.autograd.grad(scalar, (omegas, deltas, state), retain_graph=True)

    h = dense_rydberg_hamiltonian(*ham_params).to(device)
    ed = torch.linalg.matrix_exp(-1j * dt * h) @ state
    ed_scalar = torch.vdot(ed, Op @ ed).real
    ed_grads = torch.autograd.grad(ed_scalar, (omegas, deltas, state), retain_graph=True)

    for grad, ed_grad in zip(grads, ed_grads):
        assert torch.allclose(grad, ed_grad, rtol=krylov_tolerance)


@pytest.mark.parametrize(
    "N, krylov_tolerance",
    [(n, tol) for n in [4, 6] for tol in [1e-8, 1e-12]],
)
def test_backward_with_phase(N, krylov_tolerance):
    torch.manual_seed(1337)
    omegas = torch.randn(N, dtype=dtype_params, requires_grad=True)
    deltas = torch.randn(N, dtype=dtype_params, requires_grad=True)
    phis = torch.randn(N, dtype=dtype_params, requires_grad=True)
    interactions = randn_interaction_matrix(N)

    ham_params = (omegas, deltas, phis, interactions)

    state = torch.randn(2**N, dtype=dtype, requires_grad=True)
    state = state / state.norm()
    r = torch.randn(2**N, dtype=dtype)
    r /= r.norm()

    dt = 1.0  # big timestep 1 μs

    krylov, _ = EvolveStateVector.apply(dt, *ham_params, state, krylov_tolerance)
    scalar = torch.abs(r @ krylov)
    grads = torch.autograd.grad(scalar, (omegas, deltas, phis, state), retain_graph=True)

    h = dense_rydberg_hamiltonian(*ham_params).to(device)
    ed = torch.linalg.matrix_exp(-1j * dt * h) @ state
    ed_scalar = torch.abs(r @ ed)
    ed_grads = torch.autograd.grad(
        ed_scalar, (omegas, deltas, phis, state), retain_graph=True
    )

    for grad, ed_grad in zip(grads, ed_grads):
        assert torch.allclose(grad, ed_grad, rtol=krylov_tolerance)
