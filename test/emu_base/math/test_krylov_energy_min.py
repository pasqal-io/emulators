import torch
import pytest

from emu_base.math.krylov_energy_min import (
    krylov_energy_minimization_impl,
    krylov_energy_minimization,
    KrylovEnergyResult,
)

dtype = torch.complex128


def make_hermitian(m: torch.Tensor) -> torch.Tensor:
    return m @ m.mH


def test_impl_converges_to_true_ground_state():
    dim = 5
    A = torch.randn(dim, dim, dtype=dtype)
    H = make_hermitian(A)

    # Random initial vector
    v = torch.randn(dim, dtype=dtype)
    op = lambda x: H @ x

    result: KrylovEnergyResult = krylov_energy_minimization_impl(
        op=op,
        v=v,
        norm_tolerance=1e-12,
        residual_tolerance=1e-6,
        max_krylov_dim=dim,
    )

    assert result.converged is True
    assert result.happy_breakdown is True
    assert 1 <= result.iteration_count <= dim

    # Compare ground‐energy and state to torch.linalg.eigh
    eigvals, eigvecs = torch.linalg.eigh(H)
    E0 = eigvals[0].real
    assert pytest.approx(E0, rel=1e-5) == result.ground_energy

    psi0 = eigvecs[:, 0]
    psi = result.ground_state
    psi = psi / psi.norm()
    overlap = torch.abs(torch.dot(psi0.conj(), psi))
    assert overlap > 1 - 1e-5


def test_wrapper_returns_state_and_energy():
    dim = 4
    A = torch.randn(dim, dim, dtype=dtype)
    H = make_hermitian(A)

    v = torch.randn(dim, dtype=dtype)
    op = lambda x: H @ x

    psi, E = krylov_energy_minimization(
        op=op,
        v=v,
        norm_tolerance=1e-12,
        residual_tolerance=1e-6,
        max_krylov_dim=dim,
    )

    # Full eigen decomposition vs current Krylov solver
    eigvals, eigvecs = torch.linalg.eigh(H)
    E0 = eigvals[0].real
    psi0 = eigvecs[:, 0]

    assert pytest.approx(E0, rel=1e-5) == E

    # State overlap ~ 1
    psi = psi / psi.norm()
    overlap = torch.abs(torch.dot(psi0.conj(), psi))
    assert overlap > 1 - 1e-5


def test_krylov_on_diagonal():

    # Diagonal Hermitian matrix with known smallest eigenvalue
    H = torch.diag(torch.tensor([5.0, 1.0, 3.0, 4.0, 2.0], dtype=torch.float64))

    def op(v):
        return H @ v

    v = torch.randn(5, dtype=torch.float64)

    state, energy = krylov_energy_minimization(
        op=op,
        v=v,
        norm_tolerance=1e-10,
        residual_tolerance=1e-10,
        max_krylov_dim=5,
    )

    expected_energy = 1.0
    assert torch.isclose(
        torch.tensor(energy), torch.tensor(expected_energy)
    ), f"The Expected energy is {expected_energy}, here it gives {energy}"

    assert torch.isclose(
        state.norm(), torch.tensor(1.0, dtype=state.norm().dtype), atol=1e-5
    ), f"State not normalized. Norm: {state.norm()}"
