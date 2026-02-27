import torch
from dataclasses import dataclass, replace
from typing import Callable, Tuple, cast

DEFAULT_MAX_KRYLOV_DIM: int = 100
DEFAULT_MAX_RESTARTS: int = 100
NUMERICAL_TOLERANCE: float = 1e-12


@dataclass(slots=True)
class KrylovEnergyResult:
    ground_state: torch.Tensor
    ground_energy: float
    residual_norm: float
    converged: bool
    happy_breakdown: bool
    iteration_count: int
    restart_count: int


def _dotc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.vdot(a.reshape(-1), b.reshape(-1))


def _lowest_eigen_pair(h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return the lowest eigenpair of the hermitian matrix h.
    """
    err = (h - h.mH).norm()
    if err.item() > NUMERICAL_TOLERANCE:
        raise ValueError(
            f"Matrix is not Hermitian within tolerance: "
            f"rel_err={err.item():.3e} > tol={NUMERICAL_TOLERANCE:.3e}"
        )
    eig_energy, eig_state = torch.linalg.eigh(h)
    return eig_energy[0], eig_state[:, 0]


def _ritz_vector(coefficients: torch.Tensor, basis: list[torch.Tensor]) -> torch.Tensor:
    """
    Return the normalized Ritz vector from Lanczos basis vectors.
    """
    if len(coefficients) != len(basis):
        raise ValueError(
            f"Expected len(coefficients) == len(basis), got {len(coefficients)} != {len(basis)}"
        )

    Q = torch.stack(basis, dim=1)
    c = coefficients.to(device=Q.device, dtype=Q.dtype)
    v = Q @ c

    norm = cast(torch.Tensor, v.norm())  # mypy requires explicit type
    if norm.item() <= NUMERICAL_TOLERANCE:
        raise ValueError("Ritz vector has zero norm")
    return v / norm


def build_next_lanczos_vector(
    op: Callable[[torch.Tensor], torch.Tensor],
    lanczos_vectors: list[torch.Tensor],
    T_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Perform one Lanczos iteration step.

    Applies `op` to the latest Lanczos vector, updates the tridiagonal entries
    `T_matrix[i,i]=alpha` and `T_matrix[i,i±1]=beta`, and returns the unnormalized
    vector for forming the next Lanczos basis vector.
    """
    i = len(lanczos_vectors) - 1
    w = op(lanczos_vectors[i])

    alpha = _dotc(lanczos_vectors[i], w).real
    T_matrix[i, i] = alpha

    w -= alpha * lanczos_vectors[i]
    if i > 0:
        beta_prev = T_matrix[i, i - 1]
        w -= beta_prev * lanczos_vectors[i - 1]

    beta = w.norm()
    T_matrix[i + 1, i] = beta
    T_matrix[i, i + 1] = beta
    return w


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
    v_init_norm = v_init.norm()
    if v_init_norm < norm_tolerance:
        raise ValueError("Starting vector has zero norm")

    q_0 = v_init / v_init_norm
    best_state: torch.Tensor = q_0
    best_energy = (_dotc(q_0, op(q_0))).real
    best_resid = float("inf")

    lanczos_vectors: list[torch.Tensor] = [q_0]

    T = torch.zeros(
        (max_krylov_dim + 1, max_krylov_dim + 1),
        dtype=real_dtype,
        device=device,
    )

    converged = False
    happy_breakdown = False
    n_iteration = 0

    for j in range(max_krylov_dim):
        n_iteration += 1
        w = build_next_lanczos_vector(op, lanczos_vectors, T)

        m = len(lanczos_vectors)
        ritz_value, y = _lowest_eigen_pair(T[:m, :m])
        ritz_vec = _ritz_vector(y, lanczos_vectors)

        beta = T[j, j + 1]
        if beta < norm_tolerance:
            # Happy breakdown: A*lanczos_vectors doesn't produce new direction
            best_state, best_energy = ritz_vec, ritz_value
            best_resid = 0.0
            converged, happy_breakdown = True, True
            break

        resid = (beta * y[-1]).abs()  # == norm(op(ritz_vec) - ritz_value * ritz_vec)
        if resid < best_resid:
            best_state, best_energy = ritz_vec, ritz_value
            best_resid = resid.item()

        if resid < residual_tolerance:
            converged = True
            break

        lanczos_vectors.append(w / beta)

    return KrylovEnergyResult(
        ground_state=best_state,
        ground_energy=best_energy.item(),
        residual_norm=best_resid,
        converged=converged,
        happy_breakdown=happy_breakdown,
        iteration_count=n_iteration,
        restart_count=0,
    )


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

    best_resid_glob = float("inf")

    result = KrylovEnergyResult(
        ground_state=psi,
        ground_energy=_dotc(psi, op(psi)).item(),
        residual_norm=best_resid_glob,
        converged=False,
        happy_breakdown=False,
        iteration_count=0,
        restart_count=0,
    )

    for r in range(max_restarts + 1):
        result = _lowest_eigenvector_krylov_method(
            op=op,
            v_init=result.ground_state,
            residual_tolerance=residual_tolerance,
            norm_tolerance=norm_tolerance,
            max_krylov_dim=max_krylov_dim,
        )

        result = replace(result, restart_count=r)
        if result.happy_breakdown or result.converged:
            break

    total_iterations = result.iteration_count + max_krylov_dim * result.restart_count
    result = replace(result, iteration_count=total_iterations)

    return result


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
            "Krylov ground state solver did not converge within allotted iterations."
        )

    return result.ground_state, result.ground_energy
