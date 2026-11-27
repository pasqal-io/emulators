from typing import Callable
import torch

DEFAULT_MAX_KRYLOV_DIM: int = 100

# Peak number of state-sized tensors `krylov_exp_impl` holds beyond `max_krylov_dim`
# of them, outside of `op`: the initial vector (the basis grows to max_krylov_dim + 1
# vectors) and the result being accumulated; during the iteration, the latest
# vector is instead the output of `op`, so the peak of a whole call is
# max_krylov_dim + max(KRYLOV_EXTRA_VECTORS, n) vectors, n being the state-sized
# tensors `op` allocates at its own peak, its output included.
KRYLOV_EXTRA_VECTORS: int = 2


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


def _linear_combination(
    coefs: torch.Tensor, vectors: list[torch.Tensor], scale: torch.Tensor
) -> torch.Tensor:
    """scale·∑ₖ coefs[k]·vectors[k], accumulated in place: one state-sized allocation."""
    coefs = coefs.to(vectors[0].device) * scale
    result = vectors[0] * coefs[0]
    for coef, vector in zip(coefs[1:], vectors[1:]):
        result.addcmul_(vector, coef)
    return result


def krylov_exp_impl(
    op: Callable[[torch.Tensor], torch.Tensor],
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

    The input tensor object `v` becomes invalid after calling that function, unless
    the computation fails: on non-convergence or a CUDA out-of-memory error, `v` is
    restored (up to roundoff) so the caller can retry with a smaller step, and the
    Krylov basis is released before the error propagates, since the exception's
    traceback would otherwise keep it alive while the caller handles it.
    """

    initial_norm = v.norm()
    v /= initial_norm

    lanczos_vectors = [v]
    T = torch.zeros(max_krylov_dim + 2, max_krylov_dim + 2, dtype=v.dtype)

    try:
        return _krylov_loop(
            op,
            v,
            initial_norm,
            lanczos_vectors,
            T,
            is_hermitian,
            exp_tolerance,
            norm_tolerance,
            max_krylov_dim,
        )
    except torch.cuda.OutOfMemoryError:
        lanczos_vectors.clear()
        v *= initial_norm
        raise


def _krylov_loop(
    op: Callable[[torch.Tensor], torch.Tensor],
    v: torch.Tensor,
    initial_norm: torch.Tensor,
    lanczos_vectors: list[torch.Tensor],
    T: torch.Tensor,
    is_hermitian: bool,
    exp_tolerance: float,
    norm_tolerance: float,
    max_krylov_dim: int,
) -> KrylovExpResult:
    """The iteration of `krylov_exp_impl`, in its own frame so that an exception
    unwinding it drops the references to the Krylov vectors it holds."""
    for j in range(max_krylov_dim):
        w = op(lanczos_vectors[-1])

        n = w.norm()

        k_start = max(0, j - 1) if is_hermitian else 0
        for k in range(k_start, j + 1):
            overlap = torch.tensordot(lanczos_vectors[k].conj(), w, dims=w.dim())
            T[k, j] = overlap
            w.addcmul_(lanczos_vectors[k], overlap, value=-1)

        n2 = w.norm()
        T[j + 1, j] = n2

        if n2 < norm_tolerance:
            # Happy breakdown
            expd = torch.linalg.matrix_exp(T[: j + 1, : j + 1])
            result = _linear_combination(expd[:, 0], lanczos_vectors, initial_norm)
            return KrylovExpResult(
                result=result, converged=True, happy_breakdown=True, iteration_count=j + 1
            )

        w /= n2
        lanczos_vectors.append(w)

        # Compute exponential of extended T matrix
        T[j + 2, j + 1] = 1  # will be overwritten by line 62 in the next iteration
        expd = torch.linalg.matrix_exp(T[: j + 3, : j + 3])

        # Local truncation error estimation
        err1 = abs(expd[j + 1, 0])
        err2 = abs(expd[j + 2, 0] * n)

        err = err1 if err1 < err2 else (err1 * err2 / (err1 - err2))

        if err < exp_tolerance:
            # Converged
            result = _linear_combination(
                expd[: len(lanczos_vectors), 0], lanczos_vectors, initial_norm
            )
            return KrylovExpResult(
                result=result,
                converged=True,
                happy_breakdown=False,
                iteration_count=j + 1,
            )

    result = _linear_combination(
        expd[: len(lanczos_vectors), 0], lanczos_vectors, initial_norm
    )
    v *= initial_norm  # restore the input for a retry with a smaller step
    return KrylovExpResult(
        result=result,
        converged=False,
        happy_breakdown=False,
        iteration_count=max_krylov_dim,
    )


def krylov_exp(
    op: Callable[[torch.Tensor], torch.Tensor],
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
