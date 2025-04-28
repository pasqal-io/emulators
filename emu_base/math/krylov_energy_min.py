import torch
from typing import Callable, Tuple

DEFAULT_MAX_KRYLOV_DIM: int = 100


def _eigen_pair(
    T_trunc: torch.Tensor,
    guess: torch.Tensor,
    residual_tolerance: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return the smallest eigenpair of T_trunc.
    If T_trunc is too small for LOBPCG (n < 3), do a full eigh;
    otherwise call lobpcg with the given initial guess.
    """
    n = T_trunc.size(0)
    if n < 3:
        eigvals, eigvecs = torch.linalg.eigh(T_trunc)
    else:
        eigvals, eigvecs = torch.lobpcg(
            T_trunc,
            k=1,
            X=guess,
            tol=residual_tolerance,
            largest=False,
        )
    return eigvals, eigvecs


class KrylovEnergyResult:
    def __init__(
        self,
        ground_state: torch.Tensor,
        ground_energy: float,
        converged: bool,
        happy_breakdown: bool,
        iteration_count: int,
    ):
        self.ground_state = ground_state
        self.ground_energy = ground_energy
        self.converged = converged
        self.happy_breakdown = happy_breakdown
        self.iteration_count = iteration_count


def krylov_energy_minimization_impl(
    op: Callable[[torch.Tensor], torch.Tensor],
    v: torch.Tensor,
    residual_tolerance: float,
    norm_tolerance: float,
    max_krylov_dim: int = DEFAULT_MAX_KRYLOV_DIM,
) -> KrylovEnergyResult:

    device = v.device
    dtype = v.dtype

    initial_norm = v.norm()
    lanczos_vectors = [v / initial_norm]
    T = torch.zeros(max_krylov_dim + 2, max_krylov_dim + 2, dtype=dtype, device=device)

    converged = False
    happy_breakdown = False
    prev_eigen_vec: torch.Tensor | None = None

    for j in range(max_krylov_dim):
        w = op(lanczos_vectors[-1])

        for k in range(max(0, j - 1), j + 1):
            alpha = torch.tensordot(lanczos_vectors[k].conj(), w, dims=w.dim())
            T[k, j] = alpha
            w = w - alpha * lanczos_vectors[k]

        beta = w.norm()
        T[j + 1, j] = beta
        T[j, j + 1] = beta

        effective_dim = len(lanczos_vectors)
        size = len(lanczos_vectors) + (0 if beta < norm_tolerance else 1)
        T_truncated = T[:size, :size]

        if prev_eigen_vec is None:
            guessed_state = torch.zeros(
                T_truncated.size(0), 1, dtype=dtype, device=device
            )
            guessed_state[0, 0] = 1.0
        else:

            if beta > norm_tolerance:

                prev_eigen_vec = prev_eigen_vec.view(-1, 1)  # (n-1,1) or smaller
                zero = torch.zeros((1, 1), dtype=dtype, device=device)
                guessed_state = torch.cat([prev_eigen_vec, zero], dim=0)  # now (n,1)

        if beta < norm_tolerance:
            happy_breakdown = True
            eigvals, eigvecs = _eigen_pair(T_truncated, guessed_state, residual_tolerance)
            ground_energy = eigvals[0].real
            ground_eigenvector = eigvecs[:, 0]

            break

        lanczos_vectors.append(w / beta)

        eigvals, eigvecs = _eigen_pair(T_truncated, guessed_state, residual_tolerance)
        ground_energy = eigvals[0].real
        ground_eigenvector = eigvecs[:, 0]
        prev_eigen_vec = ground_eigenvector.clone()

        # Residual convergence check
        residual_norm = torch.norm(
            T_truncated @ ground_eigenvector - ground_energy * ground_eigenvector
        )
        if residual_norm < residual_tolerance:
            converged = True

    # true ground state
    coeffs = ground_eigenvector
    psi = sum(c * vec for c, vec in zip(coeffs, lanczos_vectors))
    psi = psi / psi.norm()

    return KrylovEnergyResult(
        ground_state=psi,
        ground_energy=ground_energy.item(),
        converged=converged,
        happy_breakdown=happy_breakdown,
        iteration_count=effective_dim,
    )


def krylov_energy_minimization(
    op: Callable[[torch.Tensor], torch.Tensor],
    v: torch.Tensor,
    norm_tolerance: float,
    residual_tolerance: float,
    max_krylov_dim: int = DEFAULT_MAX_KRYLOV_DIM,
) -> Tuple[torch.Tensor, float]:

    result = krylov_energy_minimization_impl(
        op=op,
        v=v,
        norm_tolerance=norm_tolerance,
        residual_tolerance=residual_tolerance,
        max_krylov_dim=max_krylov_dim,
    )

    if not result.converged and not result.happy_breakdown:
        raise RecursionError(
            "Krylov ground state solver did not converge within allotted iterations."
        )

    return result.ground_state, result.ground_energy
