import torch
from typing import Callable, Tuple

DEFAULT_MAX_KRYLOV_DIM: int = 200


def _lowest_eigen_pair(
    T_trunc: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return the lowest eigenpair of the hermitian matrix T_trunc.
    """
    eig_energy, eig_state = torch.linalg.eigh(T_trunc)
    return eig_energy[0], eig_state[:, 0]


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

    for j in range(max_krylov_dim):
        w = op(lanczos_vectors[-1])

        for k in range(max(0, j - 1), j + 1):
            alpha = torch.tensordot(lanczos_vectors[k].conj(), w, dims=w.dim())
            T[k, j] = alpha
            w = w - alpha * lanczos_vectors[k]

        beta = w.norm()
        T[j + 1, j] = beta

        effective_dim = len(lanczos_vectors)
        size = effective_dim + (0 if beta < norm_tolerance else 1)
        T_truncated = T[:size, :size]

        ground_energy, ground_eigenvector = _lowest_eigen_pair(
            T_truncated
        )  # in Krylov subspace
        iteration_count = j + 1

        # happy breakdown check
        if beta < norm_tolerance:
            final_state = sum(
                c * vec for c, vec in zip(ground_eigenvector, lanczos_vectors)
            )
            final_state = final_state / final_state.norm()
            happy_breakdown = True
            converged = True
            break

        # Reconstruct final state in original Hilbert space
        lanczos_vectors.append(w / beta)
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
    max_krylov_dim: int = DEFAULT_MAX_KRYLOV_DIM,
) -> Tuple[torch.Tensor, float]:

    result = krylov_energy_minimization_impl(
        op=op,
        psi_local=v,
        norm_tolerance=norm_tolerance,
        residual_tolerance=residual_tolerance,
        max_krylov_dim=max_krylov_dim,
    )

    if not result.converged and not result.happy_breakdown:
        raise RecursionError(
            "Krylov ground state solver did not converge within allotted iterations."
        )

    return result.ground_state, result.ground_energy
