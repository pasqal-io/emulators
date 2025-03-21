import torch
import pytest

from emu_base.math.krylov_exp import krylov_exp, krylov_exp_impl, krylov_exp_to_matrix

dtype = torch.complex128


def make_hermitian(m: torch.Tensor) -> torch.Tensor:
    m = torch.tensordot(m, m.conj().transpose(0, 1), dims=1)


def uniform_random_tensor(*dims):
    return (
        torch.rand(*dims, dtype=dtype)
        + 1j * torch.rand(*dims, dtype=dtype)
        - (0.5 + 0.5j) * torch.ones(*dims)
    )


def check(
    m: torch.Tensor,
    v: torch.Tensor,
    is_hermitian: bool,
    converged: bool,
    happy_breakdown: bool,
    iteration_count: int,
    norm_tolerance: float,
    exp_tolerance: float,
    max_krylov_dim: int = 80,
    normalize: bool = True,
):
    if normalize:
        v /= v.norm()

    def op(x):
        assert x.norm() == pytest.approx(1)  # Check all Lanczos vectors have norm 1.
        return torch.tensordot(m, x, dims=1)

    result = krylov_exp_impl(
        op,
        v,
        is_hermitian=is_hermitian,
        exp_tolerance=exp_tolerance,
        norm_tolerance=norm_tolerance,
        max_krylov_dim=max_krylov_dim,
    )

    assert result.converged == converged
    assert result.happy_breakdown == happy_breakdown
    assert result.iteration_count == iteration_count

    expected = torch.tensordot(torch.linalg.matrix_exp(m), v, dims=1)

    if result.happy_breakdown:
        assert torch.allclose(
            result.result,
            expected,
            atol=norm_tolerance,
            rtol=0,
        )
    elif result.converged:
        assert torch.allclose(
            result.result,
            expected,
            atol=exp_tolerance,
            rtol=0,
        )

    # TODO: a case when convergence is not reached?


def test_id():
    m = torch.eye(3, dtype=dtype)
    v = torch.arange(3).to(dtype)

    check(
        m,
        v,
        is_hermitian=True,
        converged=True,
        happy_breakdown=True,
        iteration_count=1,
        norm_tolerance=1e-12,
        exp_tolerance=1e-6,
    )


def test_id_large():
    dim = 300
    m = torch.eye(dim, dtype=dtype)
    v = torch.arange(dim).to(dtype)

    check(
        m,
        v,
        is_hermitian=True,
        converged=True,
        happy_breakdown=True,
        iteration_count=1,
        norm_tolerance=1e-12,
        exp_tolerance=1e-6,
    )


def test_happy_breakdown():
    m = torch.diag(torch.tensor([1.0, 2.0, 3.0, 3.0, 3.0], dtype=dtype))
    v = torch.ones(5, 1).to(dtype)

    ###
    # Minimal polynomial of m is P = (X-1).(X-2).(X-3) = X^3 - 6X^2 + 11X - 6
    #
    #         v = (1, 1, 1, 1, 1)
    #     m . v = (1, 2, 3, 3, 3)
    #   m^2 . v = (1, 4, 9, 9, 9)
    #   m^3 . v = (1, 8, 27, 27, 27)
    #
    # Happy breakdown happens at 3rd iteration because
    # P(X) = 0, so m^3.v = 6v - 11 m.v + 6 m^2.v belongs to the Krylov space of dim 3.
    ###

    check(
        m,
        v,
        is_hermitian=True,
        converged=True,
        happy_breakdown=True,
        iteration_count=3,
        norm_tolerance=1e-12,
        exp_tolerance=1e-6,
    )


def test_converged():
    torch.random.manual_seed(1234)

    dim = 1000
    m = 0.001 * uniform_random_tensor(dim, dim)
    v = uniform_random_tensor(dim, 1)
    make_hermitian(m)
    check(
        m,
        v,
        is_hermitian=True,
        converged=True,
        happy_breakdown=False,
        iteration_count=7,
        norm_tolerance=1e-20,
        exp_tolerance=1e-13,
    )


def test_converged_non_hermitian():
    torch.random.manual_seed(1234)

    dim = 1000
    m = 0.001 * uniform_random_tensor(dim, dim)
    v = uniform_random_tensor(dim, 1)

    check(
        m,
        v,
        is_hermitian=False,
        converged=True,
        happy_breakdown=False,
        iteration_count=7,
        norm_tolerance=1e-20,
        exp_tolerance=1e-15,
    )


def test_converged_non_hermitian_non_normalized():
    torch.random.manual_seed(1234)

    dim = 1000
    m = 0.001 * uniform_random_tensor(dim, dim)
    v = uniform_random_tensor(dim, 1)
    check(
        m,
        v,
        is_hermitian=False,
        converged=True,
        happy_breakdown=False,
        iteration_count=6,
        norm_tolerance=1e-20,
        exp_tolerance=1e-13,
        normalize=False,
    )


def test_krylov_with_matrix():

    def op(x):
        # pi/2 sigma_x
        A = 3.14159 / 2 * torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
        return A @ x

    M = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=dtype)

    result = krylov_exp_to_matrix(
        op,
        M,
        exp_tolerance=1e-6,
        norm_tolerance=1e-6,
        is_hermitian=False,
        max_krylov_dim=100,
    )

    A = 3.14159 / 2 * torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
    expected = torch.linalg.matrix_exp(A) @ M

    assert torch.allclose(result, expected)

    Ident = torch.eye(2, 2, dtype=dtype)
    sigma_x = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    def op(x):
        # pi/2 sigma_x
        A = (
            3.14159 / 2 * torch.kron(sigma_x, Ident)
            + 3.14159 / 2
            + torch.kron(Ident, sigma_x)
        )
        return A @ x

    M = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=dtype)

    MxM = torch.kron(M, M)
    result = krylov_exp_to_matrix(
        op,
        MxM,
        exp_tolerance=1e-6,
        norm_tolerance=1e-6,
        is_hermitian=False,
        max_krylov_dim=100,
    )

    A = (
        3.14159 / 2 * torch.kron(sigma_x, Ident)
        + 3.14159 / 2
        + torch.kron(Ident, sigma_x)
    )
    expected = torch.linalg.matrix_exp(A) @ MxM

    assert torch.allclose(result, expected)


def make_random_hermitian_mat_from_params(
    dim: int, params: tuple[torch.Tensor] = ()
) -> torch.Tensor:
    """Returns a random Hermitian matrix with non-trivial dependency on input parameters"""
    A = torch.randn(dim, dim, dtype=dtype)
    for p in params:
        A = A + p * torch.randn(dim, dim, dtype=dtype)
    return A @ A.mH


def make_random_sv_from_params(
    dim: int, params: tuple[torch.Tensor] = ()
) -> torch.Tensor:
    x = torch.randn(dim, dtype=dtype)
    """Returns a random normalized state vector with non-trivial dependency on input parameters"""
    for p in params:
        x = x + p * torch.randn(dim, dtype=dtype)
    return x / x.norm()


def test_grad_accuracy_vs_matrix_exp():
    torch.manual_seed(2024)

    omega = torch.tensor(1.0, requires_grad=True)
    delta = torch.tensor(2.0, requires_grad=True)
    params = (omega, delta)
    dim = 21
    x = make_random_sv_from_params(dim, params=params)
    H = make_random_hermitian_mat_from_params(dim, params=params)
    dt = 7

    # krylov_exp setup
    def Hcall(x: torch.Tensor) -> torch.Tensor:
        return -dt * 1j * H @ x

    y = krylov_exp(Hcall, x, exp_tolerance=1e-12, norm_tolerance=1e-12)
    loss = torch.vdot(x, y).real
    grads = torch.autograd.grad(loss, inputs=params, retain_graph=True)

    # matrix_exp setup
    y_expected = torch.matrix_exp(-dt * 1j * H) @ x
    loss_expected = torch.vdot(x, y_expected).real
    grads_expected = torch.autograd.grad(loss_expected, inputs=params)

    assert torch.allclose(y, y_expected)
    assert torch.allclose(loss, loss_expected)
    for ge, g in zip(grads_expected, grads):
        assert torch.allclose(ge, g)


def test_grad_accuracy_vs_analytical():
    torch.manual_seed(2024)

    alpha = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    dim = 18
    psi_0 = make_random_sv_from_params(dim)
    # H is linear in parameter α
    H = alpha * make_random_hermitian_mat_from_params(dim)
    t = 9.5

    def Hcall(x: torch.Tensor) -> torch.Tensor:
        return -t * 1j * H @ x

    # random observable
    M = make_random_hermitian_mat_from_params(dim)

    # |Ψt〉= U|Ψ0〉, loss = 〈Ψt|M|Ψt〉
    psi_t = krylov_exp(Hcall, psi_0, exp_tolerance=1e-12, norm_tolerance=1e-12)
    loss = torch.vdot(psi_t, M @ psi_t).real

    # ∂loss/∂α with automatic differentiation
    (grad,) = torch.autograd.grad(loss, inputs=alpha)

    # expected ∂U/∂α = -itH/α * U
    # expected ∂loss/∂α = 2Re〈Ψ0|U† M ∂U/∂α|Ψ0〉 = 2Re〈Ψt|M @ (-itH/α)|Ψt〉
    grad_expected = 2 * torch.vdot(psi_t, M @ (-1j * t * H / alpha) @ psi_t).real

    assert torch.allclose(grad_expected, grad)
