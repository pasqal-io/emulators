import torch
import pytest

from emu_base.math.krylov_energy_min import (
    krylov_energy_minimization_impl,
)

dtype = torch.complex128


def build_Ising_hamiltonian(N: int, J: float, h: float) -> torch.Tensor:
    """Build the Ising Hamiltonian and test the Lanczos routine afterwards

    H = -J * sum Z_i Z_{i+1} - h * sum X_i
    Returns: H in matrix form
    """
    Identity = torch.eye(2, dtype=torch.float64)
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.float64)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.float64)

    def construct_tensor_product(list_of_ops):
        res = list_of_ops[0]
        for operator in list_of_ops[1:]:
            res = torch.kron(res, operator)
        return res

    def build_operator(pauli_op, site):
        # Pauli op acting on one site, and Identity on all others
        list_of_op = [Identity] * N
        list_of_op[site] = pauli_op
        return construct_tensor_product(list_of_op)

    H = torch.zeros((2**N, 2**N), dtype=torch.float64)

    # Interaction term
    for i in range(N - 1):
        H += -J * build_operator(Z, i) @ build_operator(Z, i + 1)

    # Transverse field term
    for i in range(N):
        H += -h * build_operator(X, i)

    return H


def check(
    A: torch.Tensor,
    v: torch.Tensor,
    is_hermitian: bool,
    norm_tolerance: float,
    residual_tolerance: float,
    expect_converged: bool,
    expected_iteration_count: int,
    expect_happy_breakdown: bool,
    max_krylov_dim: int = 80,
):

    v = v.clone().to(A.dtype)
    v = v / v.norm()

    def op(x: torch.Tensor) -> torch.Tensor:
        assert x.norm() == pytest.approx(1)
        return A @ x

    result = krylov_energy_minimization_impl(
        op=op,
        v=v,
        norm_tolerance=norm_tolerance,
        residual_tolerance=residual_tolerance,
        is_hermitian=is_hermitian,
        max_krylov_dim=max_krylov_dim,
    )

    assert result.converged == expect_converged
    assert result.happy_breakdown == expect_happy_breakdown
    assert result.iteration_count in {
        expected_iteration_count,
        expected_iteration_count + 1,
    }

    E_approx = result.ground_energy
    psi_approx = result.ground_state

    eigen_energy, eigen_state = torch.linalg.eigh(A)
    E_exact = eigen_energy[0].item()
    psi_exact = eigen_state[:, 0]

    # test residual norm criterion
    if expect_converged and expect_happy_breakdown is False:
        assert torch.allclose(
            torch.tensor(E_approx), torch.tensor(E_exact), atol=norm_tolerance
        )

        res = torch.norm(op(psi_approx) - E_approx * psi_approx).item()
        assert res < residual_tolerance

        overlap = torch.norm(torch.dot(psi_exact.conj(), psi_approx))
        assert torch.allclose(overlap, torch.tensor(1.0, dtype=overlap.dtype), atol=1e-1)

    # test happy breakdown criterion
    if expect_happy_breakdown and expect_converged:
        assert torch.allclose(
            torch.tensor(E_approx), torch.tensor(E_exact), atol=norm_tolerance
        )

        res = torch.norm(op(psi_approx) - E_approx * psi_approx).item()
        assert res < norm_tolerance

        overlap = torch.norm(torch.dot(psi_exact.conj(), psi_approx))
        assert torch.allclose(overlap, torch.tensor(1.0, dtype=overlap.dtype), atol=1e-8)


def test_id():
    dim = 3
    A = torch.eye(dim, dtype=dtype)
    v = torch.zeros(dim, dtype=dtype)
    v[0] = 1.0

    check(
        A,
        v,
        is_hermitian=True,
        norm_tolerance=1e-8,
        residual_tolerance=1e-10,
        expect_converged=True,
        expect_happy_breakdown=True,
        expected_iteration_count=1,
        max_krylov_dim=dim,
    )


def test_id_large():
    dim = 100
    A = torch.eye(dim, dtype=dtype)
    v = torch.zeros(dim, dtype=dtype)
    v[0] = 1.0

    check(
        A,
        v,
        is_hermitian=True,
        norm_tolerance=1e-8,
        residual_tolerance=1e-10,
        expect_converged=True,
        expect_happy_breakdown=True,
        expected_iteration_count=1,
        max_krylov_dim=dim,
    )


def test_exact_case():
    # A has ground-state with eigenvalue 1.0
    A = torch.tensor([[2.0, -1.0], [-1.0, 2.0]], dtype=torch.float64)
    v = torch.tensor([1.0, 1.0], dtype=torch.float64)
    check(
        A,
        v,
        is_hermitian=True,
        norm_tolerance=1e-12,
        residual_tolerance=1e-10,
        expect_converged=True,
        expect_happy_breakdown=True,
        expected_iteration_count=1,
        max_krylov_dim=2,
    )


def test_ising_hamiltonian():
    N = 12
    d = 2
    J, h = 1.0, 0.1

    # build initial state 'v' from the solution of the smaller H
    N_small = 4
    H_small = build_Ising_hamiltonian(N_small, J, h)
    eigvals, eigvecs = torch.linalg.eigh(H_small)
    ground_state_small = eigvecs[:, 0]

    v = torch.ones(d**N)
    v[: d**N_small] = ground_state_small
    v = v / v.norm()

    A = build_Ising_hamiltonian(N, J, h)

    check(
        A,
        v,
        max_krylov_dim=d**N,
        is_hermitian=True,
        norm_tolerance=1e-7,
        residual_tolerance=1e-6,
        expect_converged=True,
        expect_happy_breakdown=False,
        expected_iteration_count=99,
    )
