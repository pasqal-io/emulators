import torch
from typing import Callable, Tuple

DEFAULT_MAX_KRYLOV_DIM: int = 100


def _eigen_pair(
    T_trunc: torch.Tensor,
    guessed_state: torch.Tensor,
    residual_tolerance: float,
    is_hermitian: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return the smallest eigenpair of T_trunc.
    If T_trunc is small (n < 3) or Hermitian, do a full eigh;
    otherwise use lobpcg with the given initial guessed state.
    """
    n = T_trunc.size(0)
    if is_hermitian:
        if n > 2:
            return torch.lobpcg(
                T_trunc,
                k=1,
                X=guessed_state,
                tol=residual_tolerance,
                largest=False,
            )
        else:
            return tuple(torch.linalg.eigh(T_trunc))

    # eig does not guarantee that eigenvalues are sorted
    # use argmin to force extract the smallest eigen-pair
    eigvals, eigvecs = torch.linalg.eig(T_trunc)
    idx = torch.argmin(eigvals.abs())
    energy = eigvals[idx : idx + 1]
    state = eigvecs[:, idx : idx + 1]
    return energy, state


def extend_initial_state(
    prev_eigen_vec: torch.Tensor, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """
    Extend the previous eigenvector guess by one zero element to match the
    increased size of T_truncated .
    """
    if prev_eigen_vec is None:
        raise ValueError("prev_eigen_vec cannot be None when extending the guess state.")
    prev_eigen_vec = prev_eigen_vec.view(-1, 1)
    zero = torch.zeros((1, 1), dtype=dtype, device=device)
    return torch.cat([prev_eigen_vec, zero], dim=0)


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
    psi_local: torch.Tensor,
    residual_tolerance: float,
    norm_tolerance: float,
    is_hermitian: bool,
    max_krylov_dim: int = DEFAULT_MAX_KRYLOV_DIM,
) -> KrylovEnergyResult:
    """
    Computes the ground state of a Hermitian operator using Lanczos algorithm.
    The Rayleigh quotient ⟨ψ|H|ψ⟩ is minimized over the Krylov subspace.

    The convergence of the results is determined by a residual norm criterion or a happy breakdown.
    """

    device = psi_local.device
    dtype = psi_local.dtype

    initial_norm = psi_local.norm()
    lanczos_vectors = [psi_local / initial_norm]
    T = torch.zeros(max_krylov_dim + 2, max_krylov_dim + 2, dtype=dtype, device=device)

    converged = False
    happy_breakdown = False
    iteration_count = 0
    prev_eigen_vec: torch.Tensor | None = None

    for j in range(max_krylov_dim):
        w = op(lanczos_vectors[-1])

        k_start = max(0, j - 1) if is_hermitian else 0
        for k in range(k_start, j + 1):
            alpha = torch.tensordot(lanczos_vectors[k].conj(), w, dims=w.dim())
            T[k, j] = alpha
            w = w - alpha * lanczos_vectors[k]

        beta = w.norm()
        T[j + 1, j] = beta

        effective_dim = len(lanczos_vectors)
        size = effective_dim + (0 if beta < norm_tolerance else 1)
        T_truncated = T[:size, :size]

        # Initial guess state for LOBPCG solver
        if prev_eigen_vec is None:
            guessed_state = torch.ones(size, 1, dtype=dtype, device=device)

        eigvals, eigvecs = _eigen_pair(
            T_truncated, guessed_state, residual_tolerance, is_hermitian
        )
        ground_energy = eigvals[0]
        ground_eigenvector = eigvecs[:, 0]  # in Krylov subspace
        iteration_count = j + 1

        if beta < norm_tolerance:
            final_state = sum(
                c * vec for c, vec in zip(ground_eigenvector, lanczos_vectors)
            )
            final_state = final_state / final_state.norm()
            happy_breakdown = True
            converged = True
            break

        lanczos_vectors.append(w / beta)
        # build the new guessed state from the eigenvector of the previous iteration
        prev_eigen_vec = ground_eigenvector.clone()
        guessed_state = extend_initial_state(prev_eigen_vec, dtype, device)

        # Reconstruct final state in original Hilbert space
        final_state = sum(c * vec for c, vec in zip(ground_eigenvector, lanczos_vectors))
        final_state = final_state / final_state.norm()

        # residual norm convergence check
        residual_norm = torch.norm(op(final_state) - ground_energy * final_state)
        if residual_norm < residual_tolerance:
            happy_breakdown = False
            converged = True
            break

    return KrylovEnergyResult(
        ground_state=final_state,
        ground_energy=ground_energy.item(),
        converged=converged,
        happy_breakdown=happy_breakdown,
        iteration_count=iteration_count,
    )


def krylov_energy_minimization(
    op: Callable[[torch.Tensor], torch.Tensor],
    v: torch.Tensor,
    norm_tolerance: float,
    residual_tolerance: float,
    is_hermitian: bool = True,
    max_krylov_dim: int = DEFAULT_MAX_KRYLOV_DIM,
) -> Tuple[torch.Tensor, float]:

    result = krylov_energy_minimization_impl(
        op=op,
        psi_local=v,
        norm_tolerance=norm_tolerance,
        residual_tolerance=residual_tolerance,
        is_hermitian=is_hermitian,
        max_krylov_dim=max_krylov_dim,
    )

    if not result.converged and not result.happy_breakdown:
        raise RecursionError(
            "Krylov ground state solver did not converge within allotted iterations."
        )

    return result.ground_state, result.ground_energy
