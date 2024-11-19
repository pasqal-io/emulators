from typing import Callable
import torch

DEFAULT_MAX_KRYLOV_DIM: int = 100


class KrylovExpResult:
    def __init__(
        self,
        result: torch.Tensor,
        converged: bool,
        happy_breakdown: bool,
        iteration_count: int,
    ):
        assert (not happy_breakdown) or converged

        self.converged = converged
        self.happy_breakdown = happy_breakdown
        self.iteration_count = iteration_count
        self.result = result


def krylov_exp_impl(
    op: Callable,
    v: torch.Tensor,
    is_hermitian: bool,  # note: complex-proportional to its adjoint is enough
    exp_tolerance: float,
    norm_tolerance: float,
    max_krylov_dim: int = DEFAULT_MAX_KRYLOV_DIM,
) -> KrylovExpResult:
    """
    Computes exp(op).v using either the Lanczos or Arnoldi algorithm,
    based on the `is_hermitian` flag.
    All inputs must be on the same device.

    Convergence is checked using the exponential of the "extended T matrix", a criterion
    described in "Expokit: A Software Package for Computing Matrix Exponentials"
    (https://www.maths.uq.edu.au/expokit/paper.pdf).
    """

    initial_norm = v.norm()

    lanczos_vectors = [v / initial_norm]
    T = torch.zeros(max_krylov_dim + 2, max_krylov_dim + 2, dtype=v.dtype)

    for j in range(max_krylov_dim):
        w = op(lanczos_vectors[-1])

        n = w.norm()

        k_start = max(0, j - 1) if is_hermitian else 0
        for k in range(k_start, j + 1):
            overlap = torch.tensordot(lanczos_vectors[k].conj(), w, dims=w.dim())
            T[k, j] = overlap
            w = w - overlap * lanczos_vectors[k]

        n2 = w.norm()
        T[j + 1, j] = n2

        if n2 < norm_tolerance:
            # Happy breakdown
            expd = torch.linalg.matrix_exp(T[: j + 1, : j + 1])
            result = initial_norm * sum(
                a * b for a, b in zip(expd[:, 0], lanczos_vectors)
            )
            return KrylovExpResult(
                result=result, converged=True, happy_breakdown=True, iteration_count=j + 1
            )

        lanczos_vectors.append(w / n2)

        # Compute exponential of extended T matrix
        T[j + 2, j + 1] = 1
        expd = torch.linalg.matrix_exp(T[: j + 3, : j + 3])

        # Local truncation error estimation
        err1 = abs(expd[j + 1, 0])
        err2 = abs(expd[j + 2, 0] * n)

        err = err1 if err1 < err2 else (err1 * err2 / (err1 - err2))

        if err < exp_tolerance:
            # Converged
            result = initial_norm * sum(
                a * b for a, b in zip(expd[: len(lanczos_vectors), 0], lanczos_vectors)
            )
            return KrylovExpResult(
                result=result,
                converged=True,
                happy_breakdown=False,
                iteration_count=j + 1,
            )

    result = initial_norm * sum(
        a * b for a, b in zip(expd[: len(lanczos_vectors), 0], lanczos_vectors)
    )
    return KrylovExpResult(
        result=result,
        converged=False,
        happy_breakdown=False,
        iteration_count=max_krylov_dim,
    )


def krylov_exp(
    op: torch.Tensor,
    v: torch.Tensor,
    exp_tolerance: float,
    norm_tolerance: float,
    is_hermitian: bool = True,  # note: complex-proportional to its adjoint is enough
    max_krylov_dim: int = DEFAULT_MAX_KRYLOV_DIM,
) -> torch.Tensor:
    krylov_result = krylov_exp_impl(
        op,
        v,
        is_hermitian=is_hermitian,
        exp_tolerance=exp_tolerance,
        norm_tolerance=norm_tolerance,
        max_krylov_dim=max_krylov_dim,
    )

    if not krylov_result.converged:
        raise RecursionError(
            "exponentiation algorithm did not converge to precision in allotted number of steps."
        )

    return krylov_result.result
