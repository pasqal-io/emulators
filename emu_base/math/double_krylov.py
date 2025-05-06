import torch
from typing import Callable

from emu_base.math.krylov_exp import DEFAULT_MAX_KRYLOV_DIM

max_krylov_dim = DEFAULT_MAX_KRYLOV_DIM


def double_krylov(
    op: Callable,
    state: torch.Tensor,
    grad: torch.Tensor,
    tolerance: float,
) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
    """
    Returns a Lanczos decomposition of the Fréchet derivative of the exponential
    map U=exp(op) along the direction |state❭❬grad|.
    The decomposition is represented by the tuple (Vs, dS, Vg) such that,
        dU(op, |state❭❬grad|) = Vsᵗ @ dS @ Vg*

    Args:
        op (Callable): linear map to exponentiate, e.g. op(|ψ❭) = H|ψ❭.
        state (torch.Tensor):
        grad (torch.Tensor):
        tolerance (float): tolerance of the returned derivative.

    Returns:
        Vstate (list): Lanczos basis of state
        dS (torch.Tensor): matrix representing the derivative in the new basis
        Vgrad (list): Lanczos basis of grad

    Notes:
        Fréchet derivative dU(op,|a❭❬b|) being defined as:
            exp|op |a❭❬b|| = |exp(op) dU(op,|a❭❬b|)|
               |0   op   |   |0       exp(op)      |

        The function computes two Lanczos decomposition
        up to the given tolerance
            Va = Lanczos(|a❭,op(|a❭),op^2(|a❭),...)
            Vb = Lanczos(|b❭,op(|b❭),op^2(|b❭),...)
        such that,
            op = Vaᵗ @ Ta @ Va*
            op = Vbᵗ @ Tb @ Vb*

        In the new basis Va, Vb
            |op |a❭❬b|| -> |Ta  ab|0❭❬0||
            |0   op   |    |0   Tb      |
        where the top-right block only has one nonzero element.
        Exponentiating such matrix and selecting the top-right block
        gives us the desired matrix dS such that
            dU(op, |a❭❬b|) = Vaᵗ @ dS @ Vb*
    """
    Vs, Ts = lanczos(op, state, tolerance)
    Vg, Tg = lanczos(op, grad, tolerance)
    size_s = len(Vs)
    big_mat = torch.block_diag(Ts, Tg)
    # Only one element in the top-right corner
    big_mat[0, size_s] = state.norm() * grad.norm()
    dS = torch.matrix_exp(big_mat)[:size_s, size_s:]
    return Vs, dS, Vg


def lanczos(
    op: Callable,
    v: torch.Tensor,
    tolerance: float,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """
    Copy of the code in krylov_exp to do Laczos iteration
    To decide
     1. refactor this
     2. allow krylov results to store lanczos_vectors and T
    """
    converged = False
    lanczos_vectors = [v / v.norm()]
    T = torch.zeros(
        max_krylov_dim + 2, max_krylov_dim + 2, dtype=v.dtype, device=v.device
    )

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
            converged = True
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
            converged = True
            break

    if not converged:
        raise RecursionError(
            "Lanczos iteration did not converge to precision in allotted number of steps."
        )
    size = len(lanczos_vectors)
    return lanczos_vectors, T[:size, :size]
