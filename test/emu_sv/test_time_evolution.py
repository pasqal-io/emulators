import torch
import pytest
from test.utils_testing import (
    dense_rydberg_hamiltonian,
    randn_interaction_matrix,
)
from emu_sv.time_evolution import EvolveStateVector

dtype = torch.complex128
dtype_params = torch.float64
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
    nqubits: int, with_phase: bool = False, **kwargs
) -> tuple[torch.Tensor]:
    omegas = torch.randn(nqubits, **kwargs)
    deltas = torch.randn(nqubits, **kwargs)
    if with_phase:
        phis = torch.randn(nqubits, **kwargs)
    else:
        phis = torch.zeros(nqubits)
    interaction = randn_interaction_matrix(nqubits)
    return omegas, deltas, phis, interaction


def get_randn_state(nqubits: int, **kwargs) -> torch.Tensor:
    state = torch.randn(2**nqubits, **kwargs)
    return state / state.norm()


@pytest.mark.parametrize(
    "N, tolerance, with_phase",
    [
        (n, tol, wp)
        for n in [3, 5, 8]
        for tol in [1e-8, 1e-10, 1e-12]
        for wp in [False, True]
    ],
)
def test_forward(N: int, tolerance: float, with_phase: bool) -> None:
    torch.manual_seed(1337)
    ham_params = get_randn_ham_params(N, with_phase=with_phase, dtype=dtype_params)
    state_in = get_randn_state(N, dtype=dtype, device=device)
    dt = 1.0  # 1 μs big time step

    expected = do_dense_time_step(dt, *ham_params, state_in)
    state_out, _ = EvolveStateVector.apply(
        dt,
        *ham_params,
        state_in,
        tolerance,
    )
    assert torch.allclose(expected, state_out, atol=tolerance)


@pytest.mark.parametrize(
    "N, tolerance",
    [(n, tol) for n in [3, 5, 8] for tol in [1e-8, 1e-10, 1e-12]],
)
def test_backward(N, tolerance):
    torch.manual_seed(1337)
    ham_params = get_randn_ham_params(
        N, with_phase=True, dtype=dtype_params, requires_grad=True
    )
    state_in = get_randn_state(N, dtype=dtype, device=device, requires_grad=True)
    dt = 1.0  # big timestep 1 μs

    # arbitrary vector to construct a scalar
    r = torch.randn(2**N, dtype=dtype)
    r *= torch.randn(1) / r.norm()

    state_out, _ = EvolveStateVector.apply(dt, *ham_params, state_in, tolerance)
    scalar = torch.vdot(r, state_out).real
    omegas, deltas, phis = ham_params[0:3]
    grads = torch.autograd.grad(
        scalar, (omegas, deltas, phis, state_in), retain_graph=True
    )

    expected = do_dense_time_step(dt, *ham_params, state_in)
    expected_scalar = torch.vdot(r, expected).real
    expected_grads = torch.autograd.grad(
        expected_scalar, (omegas, deltas, phis, state_in), retain_graph=True
    )

    for g, eg in zip(grads, expected_grads):
        assert torch.allclose(g, eg, rtol=tolerance)
