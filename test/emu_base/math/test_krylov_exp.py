import weakref
import torch
import pytest

from emu_base.math.krylov_exp import krylov_exp, krylov_exp_impl

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
        v.clone(),
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


def test_happy_breakdown() -> None:
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


@pytest.mark.usefixtures("restore_rng_state")
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


@pytest.mark.usefixtures("restore_rng_state")
def test_converged_non_hermitian() -> None:
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


@pytest.mark.usefixtures("restore_rng_state")
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


@pytest.mark.usefixtures("restore_rng_state")
def test_krylov_with_matrix():

    torch.random.manual_seed(1211)

    n_atoms = 8  # do not change this value and let is_hermitian be False
    random_matrix = torch.rand(2**n_atoms, 2**n_atoms, dtype=dtype)  # random operator
    random_matrix = random_matrix / random_matrix.norm()

    def op(x):
        return random_matrix @ x

    MxM = torch.rand(2**n_atoms, 2**n_atoms, dtype=dtype)  # matrix to be evolved

    result = krylov_exp(
        op,
        MxM.clone(),
        exp_tolerance=1e-6,
        norm_tolerance=1e-6,
        is_hermitian=False,
        max_krylov_dim=100,
    )
    expected = torch.linalg.matrix_exp(random_matrix) @ MxM

    assert torch.allclose(result, expected)


def test_non_convergence_restores_input():
    """A step too large for the cap leaves v as it was, so the caller can retry."""
    torch.manual_seed(1337)
    m = uniform_random_tensor(40, 40)
    m = -1j * 5.0 * (m + m.conj().T)
    v = 3.0 * uniform_random_tensor(40)
    v_in = v.clone()

    result = krylov_exp_impl(
        lambda x: m @ x,
        v,
        is_hermitian=True,
        exp_tolerance=1e-12,
        norm_tolerance=1e-12,
        max_krylov_dim=3,
    )

    assert not result.converged
    assert torch.allclose(v, v_in, atol=1e-14, rtol=0)


def test_oom_restores_input_and_releases_basis():
    """A CUDA out-of-memory error in the iteration propagates with v restored and
    the Krylov basis dropped (the traceback would otherwise keep it alive)."""
    torch.manual_seed(1337)
    m = uniform_random_tensor(40, 40)
    m = -1j * (m + m.conj().T)
    v = 3.0 * uniform_random_tensor(40)
    v_in = v.clone()
    calls = 0
    produced: list[torch.Tensor] = []

    def op(x: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise torch.cuda.OutOfMemoryError("simulated")
        produced.append(m @ x)
        return produced[-1]

    # handled like a caller retrying would: the exception is dropped afterwards
    try:
        krylov_exp_impl(
            op, v, is_hermitian=True, exp_tolerance=1e-12, norm_tolerance=1e-12
        )
    except torch.cuda.OutOfMemoryError:
        oom = True
    assert oom

    assert torch.allclose(v, v_in, atol=1e-14, rtol=0)
    # the normalised op outputs became the basis vectors: once this test's own
    # list lets go of them, nothing else keeps them alive
    assert len(produced) == 3
    basis = [weakref.ref(t) for t in produced]
    produced.clear()
    assert all(ref() is None for ref in basis)
