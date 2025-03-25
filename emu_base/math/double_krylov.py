import torch
from typing import Callable


from emu_base.math.krylov_exp import DEFAULT_MAX_KRYLOV_DIM

# some number larger than needed
max_krylov_dim = DEFAULT_MAX_KRYLOV_DIM


def double_krylov(
    op: Callable,
    grad: torch.Tensor,
    state: torch.Tensor,
    iteration_count: int,
    tolerance: float,
) -> tuple[list, torch.Tensor, torch.Tensor]:
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

    lanczos_vectors_even = [grad / grad.norm()]
    lanczos_vectors_odd = torch.zeros(
        iteration_count + 1, state.shape[0], dtype=torch.complex128, device=state.device
    )
    lanczos_vectors_odd[0] = state / state.norm()
    Tb = torch.zeros(2 * max_krylov_dim + 4, 2 * max_krylov_dim + 4, dtype=state.dtype)
    Tb[1, 0] = (
        grad.norm() * state.norm()
    )  # the A matrix in the top left overlaps these two vectors, and nothing else
    T = Tb[1::2, 1::2]
    for j_o in range(iteration_count):
        w = op(lanczos_vectors_odd[j_o])

        for k in range(max(0, j_o - 1), j_o + 1):
            tmp = torch.tensordot(lanczos_vectors_odd[k].conj(), w, dims=w.dim())
            T[k, j_o] = tmp
            w -= tmp * lanczos_vectors_odd[k]

        tmp = w.norm()
        T[j_o + 1, j_o] = tmp

        if tmp.real < tolerance:
            # Happy breakdown
            lanczos_vectors_odd = lanczos_vectors_odd[:-1]
            break

        lanczos_vectors_odd[j_o + 1] = w / tmp

        # Compute exponential of extended T matrix
        T[j_o + 2, j_o + 1] = 1
    if T[j_o + 2, j_o + 1]:
        size_odd = j_o + 3
    else:
        size_odd = j_o + 1
    current_vector = 0
    old = 1e12
    new = 1e12
    n = 1e12
    for j_e in range(max_krylov_dim):  # TODO: exit condition
        w = op(lanczos_vectors_even[-1])
        for k in range(max(0, j_e - 1), j_e + 1):
            # overlaps of the even subspace
            Tb[2 * k, 2 * j_e] = torch.tensordot(
                lanczos_vectors_even[k].conj(), w, dims=w.dim()
            )
            w -= Tb[2 * k, 2 * j_e] * lanczos_vectors_even[k]
        # even
        tmp = w.norm()
        Tb[2 * j_e + 2, 2 * j_e] = tmp
        if tmp.real < tolerance:
            # Happy breakdown
            size = max(2 * j_e + 2, 2 * size_odd)
            expd = torch.linalg.matrix_exp(Tb[:size, :size])
            current_vector = j_e
            break
        lanczos_vectors_even.append(w / tmp)
        # Compute exponential of extended T matrix
        Tb[2 * j_e + 4, 2 * j_e + 2] = 1
        size = max(2 * j_e + 6, 2 * size_odd)
        expd = torch.linalg.matrix_exp(Tb[:size, :size])
        old = new
        new = abs(expd[1, 2 * current_vector])

        err = abs(old - new)

        if err < tolerance:
            # check if we have enough vectors for the trace
            tr_err = expd[1:size:2, 2 * current_vector]
            n = tr_err.norm()
            if n < tolerance:
                break
            current_vector += 1
            new = abs(expd[1, 2 * current_vector])
    return lanczos_vectors_even[: current_vector + 1], lanczos_vectors_odd, expd
