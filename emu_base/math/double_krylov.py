import torch
from typing import Callable

from emu_base.math.krylov_exp import DEFAULT_MAX_KRYLOV_DIM

max_krylov_dim = DEFAULT_MAX_KRYLOV_DIM


def double_krylov(
    op: Callable,
    grad: torch.Tensor,
    state: torch.Tensor,
    tolerance: float,
) -> tuple[list, list, torch.Tensor]:
    """
    Decomposition of the Fréchet derivative of the exponential map along the
    direction |state❭❬grad|, such that, with U=exp(op),
        dU(op, |state❭❬grad|) = V_even * H * V_odd^-1

    Args:
        op (Callable): linear map to exponentiate, op(|x❭) = H|x❭.
        grad (torch.Tensor):
        state (torch.Tensor):
        tolerance (float): tolerance of the returned derivative.

    Notes:

            |op |state❭❬grad||
    let h = |0        op     |
    Then this computes
    V = Gram-Scmidt(grad,state,op(grad),op(state),op^2(grad),op^2(state),...)
    and e^T = V^-1 exp(h) V
    """
    lanczos_vectors_state, Ts = lanczos(op, state, tolerance)
    lanczos_vectors_grad, Tg = lanczos(op, grad, tolerance)

    size_Ts = Ts.shape[0]
    size_Tg = Tg.shape[0]

    big_mat = torch.block_diag(Ts, Tg)
    big_mat[0, size_Ts] = state.norm() * grad.norm()
    dU = torch.matrix_exp(big_mat)[:size_Ts, size_Tg:]

    return lanczos_vectors_state, lanczos_vectors_grad, dU


def lanczos(
    op: Callable,
    v: torch.Tensor,
    tolerance: float,
) -> tuple[list, torch.Tensor]:
    lanczos_vectors = [v / v.norm()]
    T = torch.zeros(max_krylov_dim + 2, max_krylov_dim + 2, dtype=v.dtype)

    for j in range(max_krylov_dim):
        w = op(lanczos_vectors[-1])
        n = w.norm()
        for k in range(max(0, j - 1), j + 1):
            overlap = torch.tensordot(lanczos_vectors[k].conj(), w, dims=w.dim())
            T[k, j] = overlap
            w -= overlap * lanczos_vectors[k]

        n2 = w.norm()
        T[j + 1, j] = n2

        if n2 < tolerance:
            break

        lanczos_vectors.append(w / n2)
        # Compute exponential of extended T matrix
        T[j + 2, j + 1] = 1
        expd = torch.linalg.matrix_exp(T[: j + 3, : j + 3])

        # Local truncation error estimation
        err1 = abs(expd[j + 1, 0])
        err2 = abs(expd[j + 2, 0] * n)

        err = err1 if err1 < err2 else (err1 * err2 / (err1 - err2))
        if err < tolerance:
            break

    size = len(lanczos_vectors)
    return lanczos_vectors, T[:size, :size]
