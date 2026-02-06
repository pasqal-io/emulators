import torch
import pytest
from test.utils_testing import (
    dense_rydberg_hamiltonian,
    randn_interaction_matrix,
)
from emu_sv.time_evolution import EvolveStateVector

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
    state_out, _ = EvolveStateVector.apply(0.3, *ham_params, state_in, 1e-5, None)
    assert state_out.requires_grad


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
    state_out, _ = EvolveStateVector.apply(dt, *ham_params, state_in, tolerance, None)
    assert torch.allclose(expected, state_out, atol=tolerance)


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

    state_out, _ = EvolveStateVector.apply(dt, *ham_params, state_in, tolerance, None)
    scalar = torch.vdot(r, state_out).real
    grads = torch.autograd.grad(scalar, (*ham_params, state_in))

    expected = do_dense_time_step(dt, *ham_params, state_in)
    expected_scalar = torch.vdot(r, expected).real
    expected_grads = torch.autograd.grad(expected_scalar, (*ham_params, state_in))

    for g, eg in zip(grads, expected_grads):
        assert torch.allclose(g, eg, rtol=tolerance)
