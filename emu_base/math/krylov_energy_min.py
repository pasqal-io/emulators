"""
To implement the thick-restart Lanczos algorithm, I followed:
(0) Thick-Restart Lanczos Method for Large Symmetric
Eigenvalue Problems, K. Wu, H. Simon
https://doi.org/10.1137/S0895479898334605
(1) Numerical Methods for Large Eigenvalue Problems, Y. Saad.
https://epubs.siam.org/doi/book/10.1137/1.9781611970739
Standard SIAM book on Krylov methods
(2) Applied numerical linear algebra, J. Demmel
https://www.stat.uchicago.edu/~lekheng/courses/302/demmel/
Chapter 7 shows cases on misconvergence and residual growth.
(3) Numerical Methods for Solving Large Scale Eigenvalue Problems, P. Arbenz
https://people.inf.ethz.ch/arbenz/ewp/lnotes.html
Chapter 11. Explain the original paper and
Restarting Arnodli and Lanczos algorithms, algo 11.4
"""

import torch
from dataclasses import dataclass, replace
from typing import Callable, Tuple, cast

DEFAULT_MAX_KRYLOV_DIM: int = 100
DEFAULT_MAX_RESTARTS: int = 100
NUMERICAL_TOLERANCE: float = 1e-12


@dataclass(slots=True)
class KrylovEnergyResult:
    ground_state: torch.Tensor
    ground_energy: torch.Tensor
    residual_norm: torch.Tensor
    converged: bool
    happy_breakdown: bool
    iteration_count: int
    restart_count: int


def krylov_energy_minimization(
    op: Callable[[torch.Tensor], torch.Tensor],
    psi: torch.Tensor,
    norm_tolerance: float,
    residual_tolerance: float,
    max_krylov_dim: int = DEFAULT_MAX_KRYLOV_DIM,
) -> Tuple[torch.Tensor, float]:

    result = krylov_energy_minimization_impl(
        op=op,
        psi=psi,
        norm_tolerance=norm_tolerance,
        residual_tolerance=residual_tolerance,
        max_krylov_dim=max_krylov_dim,
    )

    if not result.converged and not result.happy_breakdown:
        raise RecursionError(
            "Krylov ground state solver did not converge "
            f"within allotted {max_krylov_dim} iterations."
        )

    return result.ground_state, result.ground_energy.item()


def krylov_energy_minimization_impl(
    op: Callable[[torch.Tensor], torch.Tensor],
    psi: torch.Tensor,
    residual_tolerance: float,
    norm_tolerance: float,
    max_krylov_dim: int = DEFAULT_MAX_KRYLOV_DIM,
    *,
    max_restarts: int = DEFAULT_MAX_RESTARTS,
) -> KrylovEnergyResult:
    """
    Restarted Lanczos ground state search for a Hermitian operator.

    Restart strategy (explicit):
      - Run a Lanczos cycle up to `max_krylov_dim`.
      - If not converged, restart with the best Ritz vector (lowest residual) from that cycle.

    Convergence:
      - residual norm ||Hψ - Eψ|| < residual_tolerance
      - happy breakdown: beta < norm_tolerance
    """
    result = KrylovEnergyResult(
        ground_state=psi,
        ground_energy=torch.tensor(float("inf"), device=psi.device),
        residual_norm=torch.tensor(float("inf"), device=psi.device),
        converged=False,
        happy_breakdown=False,
        iteration_count=0,
        restart_count=0,
    )

    total_iters = 0
    for r in range(max_restarts + 1):
        result = _lowest_eigenvector_krylov_method(
            op=op,
            v_init=result.ground_state,
            residual_tolerance=residual_tolerance,
            norm_tolerance=norm_tolerance,
            max_krylov_dim=max_krylov_dim,
        )

        total_iters += result.iteration_count
        result = replace(result, restart_count=r, iteration_count=total_iters)

        if result.happy_breakdown or result.converged:
            break

    return result


def _lowest_eigenvector_krylov_method(
    op: Callable[[torch.Tensor], torch.Tensor],
    v_init: torch.Tensor,
    residual_tolerance: float,
    norm_tolerance: float,
    max_krylov_dim: int = DEFAULT_MAX_KRYLOV_DIM,
) -> KrylovEnergyResult:
    """
    Approximate the lowest eigenpair of a Hermitian linear operator via Lanczos/Krylov.

    Builds a Lanczos basis from `v_init`, solves the projected eigenproblem each iteration
    to obtain a Ritz pair, tracks the best (lowest) Ritz value, and stops when the Ritz
    residual norm meets `residual_tolerance` or a happy breakdown
    (norm of the new vector is 0) occurs.
    """

    device = v_init.device
    real_dtype = v_init.real.dtype
    v_init_norm = cast(torch.Tensor, v_init.norm())
    if v_init_norm.item() < norm_tolerance:
        raise ValueError("Starting vector has zero norm")

    q_0 = v_init / v_init_norm
    lanczos_vectors: list[torch.Tensor] = [q_0]
    alphas = torch.zeros(max_krylov_dim, dtype=real_dtype, device=device)
    betas = torch.zeros(max_krylov_dim, dtype=real_dtype, device=device)

    best_state: torch.Tensor = q_0
    best_energy = torch.tensor(float("inf"), device=device)
    best_resid = torch.tensor(float("inf"), device=device)
    converged = False
    happy_breakdown = False
    n_iteration = 0

    for j in range(max_krylov_dim):
        n_iteration += 1
        w = _next_lanczos_iteration(op, lanczos_vectors, alphas, betas)

        m = len(lanczos_vectors)
        ritz_value, y = _lowest_ritz_pair_tridiagonal(alphas[:m], betas[: m - 1])
        ritz_vec = _ritz_vector(y, lanczos_vectors)
        # Residual equivalence: see Saad (1), Prop. 6.8 (Ch. 6, p. 131).
        resid = (betas[j] * y[j]).abs()  # == norm(op(ritz_vec) - ritz_value * ritz_vec)

        if resid < best_resid:
            best_state, best_energy = ritz_vec, ritz_value
            best_resid = resid

        if betas[j] < norm_tolerance:
            # Happy breakdown: A*lanczos_vectors doesn't produce new direction
            converged, happy_breakdown = True, True
            break

        if resid.item() < residual_tolerance:
            converged = True
            break

        lanczos_vectors.append(w / betas[j])

    return KrylovEnergyResult(
        ground_state=best_state,
        ground_energy=best_energy,
        residual_norm=best_resid,
        converged=converged,
        happy_breakdown=happy_breakdown,
        iteration_count=n_iteration,
        restart_count=0,
    )


def _next_lanczos_iteration(
    op: Callable[[torch.Tensor], torch.Tensor],
    lanczos_vectors: list[torch.Tensor],
    alphas: torch.Tensor,
    betas: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the next Lanczos vector (aka residual)`w` and the next
    coefficients `alpha[i]`, `beta[i]`.

    Applies `op` to the most recent Lanczos vector `q_i`, orthogonalizes the result
    against `q_i` and `q_{i-1}` and returns `(w, alpha, beta)`
    where
    `w = op(q_i) - alpha[i] * q_i - beta_{i-1} * q_{i-1}`,
    `alpha[i] = <q_i, op(q_i)>`,
    `beta[i] = ||w||`,
    `w` is the unnormalized candidate to form `q_{i+1} = w / ||w||`.
    """
    i = len(lanczos_vectors) - 1
    w = op(lanczos_vectors[i])

    alphas[i] = torch.vdot(lanczos_vectors[i].reshape(-1), w.reshape(-1))
    w -= alphas[i] * lanczos_vectors[i]
    if i > 0:
        w -= betas[i - 1] * lanczos_vectors[i - 1]
    betas[i] = cast(torch.Tensor, w.norm())
    return w


def _ritz_vector(
    coefficients: torch.Tensor,
    basis: list[torch.Tensor],
) -> torch.Tensor:
    """
    Return the normalized Ritz vector from Lanczos basis vectors.
    """
    ritz_v = sum(c * vec for c, vec in zip(coefficients, basis))

    norm = ritz_v.norm()
    if norm.item() <= NUMERICAL_TOLERANCE:
        raise ValueError("Ritz vector has zero norm")
    return cast(torch.Tensor, ritz_v / norm)  # mypy requires explicit type


def _lowest_ritz_pair_tridiagonal(
    alphas: torch.Tensor,
    betas: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the lowest Ritz pair - smallest eigenvalue and eigenvector of a
    symmetric/Hermitian tridiagonal matrix defined by its diagonal `alphas`
    and subdiagonal `betas`.

    Note on implementation:
        `torch.linalg.eigh` by default requires the lower triangular part of
        the input symmetric/Hermitian matrix via UPLO='L'.
    """
    n = alphas.numel()
    h = alphas.new_zeros((n, n))
    h.diagonal(0).copy_(alphas)
    h.diagonal(-1).copy_(betas)

    eig_energy, eig_state = torch.linalg.eigh(h, UPLO="L")
    return eig_energy[0], eig_state[:, 0]
