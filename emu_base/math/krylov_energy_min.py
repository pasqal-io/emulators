import torch
from typing import Callable, Tuple

DEFAULT_MAX_KRYLOV_DIM: int = 200


def _lowest_eigen_pair(T_trunc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    eig_energy, eig_state = torch.linalg.eigh(T_trunc)  # Hermitian
    return eig_energy[0], eig_state[:, 0]


def _ritz_vector(
    coefficients: torch.Tensor, lanczos_vectors: list[torch.Tensor]
) -> torch.Tensor:
    psi = torch.zeros_like(lanczos_vectors[0])
    for c, vec in zip(coefficients, lanczos_vectors):
        psi += c * vec

    norm = torch.linalg.norm(psi)
    assert isinstance(norm, torch.Tensor)
    return psi / norm


def _mgs_reorthogonalize(v: torch.Tensor, basis: list[torch.Tensor]) -> torch.Tensor:
    for q in basis:
        v = v - torch.vdot(q, v) * q
    return v


class KrylovEnergyResult:
    def __init__(
        self,
        ground_state: torch.Tensor,
        ground_energy: float,
        residual_norm: float,
        converged: bool,
        happy_breakdown: bool,
        iteration_count: int,
        restart_count: int,
    ):
        self.ground_state = ground_state
        self.ground_energy = ground_energy
        self.residual_norm = residual_norm
        self.converged = converged
        self.happy_breakdown = happy_breakdown
        self.iteration_count = iteration_count
        self.restart_count = restart_count


@torch.no_grad()
def krylov_energy_minimization_impl(
    op: Callable[[torch.Tensor], torch.Tensor],
    psi_local: torch.Tensor,
    residual_tolerance: float,
    norm_tolerance: float,
    max_krylov_dim: int = DEFAULT_MAX_KRYLOV_DIM,
    *,
    max_restarts: int = 20,
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
    if max_krylov_dim < 1:
        raise ValueError("max_krylov_dim must be >= 1")
    if max_restarts < 0:
        raise ValueError("max_restarts must be >= 0")
    if residual_tolerance <= 0.0 or norm_tolerance <= 0.0:
        raise ValueError("tolerances must be positive")

    device = psi_local.device
    real_dtype = psi_local.real.dtype

    total_iters = 0
    restart_count = 0

    # Global best (across restarts)
    best_state_global = None
    best_energy_global = None
    best_resid_global = torch.tensor(float("inf"), device=device, dtype=real_dtype)

    converged = False
    happy_breakdown = False

    # Start vector for current cycle
    psi_start = psi_local

    for r in range(max_restarts + 1):
        restart_count = r

        psi_norm = psi_start.norm()
        if psi_norm.item() == 0.0:
            raise ValueError("psi_local (or restart vector) has zero norm")

        lanczos_vectors: list[torch.Tensor] = [psi_start / psi_norm]

        T = torch.zeros(
            (max_krylov_dim + 2, max_krylov_dim + 2),
            dtype=real_dtype,
            device=device,
        )

        # Best within this cycle
        best_state = lanczos_vectors[0]
        best_energy = (
            torch.vdot(best_state, op(best_state)) / torch.vdot(best_state, best_state)
        ).real.to(real_dtype)
        best_resid = torch.tensor(float("inf"), device=device, dtype=real_dtype)

        cycle_converged = False
        cycle_happy = False

        for j in range(max_krylov_dim):
            qj = lanczos_vectors[-1]
            w = op(qj)

            # 3-term recurrence fill (real symmetric T)
            for k in range(max(0, j - 1), j + 1):
                alpha = torch.vdot(lanczos_vectors[k], w).real
                T[k, j] = alpha
                w = w - alpha * lanczos_vectors[k]

            w = _mgs_reorthogonalize(w, lanczos_vectors)

            beta = w.norm().to(real_dtype)
            T[j + 1, j] = beta
            T[j, j + 1] = beta

            total_iters += 1

            m = len(lanczos_vectors)
            E, y = _lowest_eigen_pair(T[:m, :m])
            psi = _ritz_vector(y, lanczos_vectors)

            # Happy breakdown: no new direction
            if beta < norm_tolerance:
                best_state, best_energy = psi, E
                best_resid = torch.zeros((), device=device, dtype=real_dtype)
                cycle_converged, cycle_happy = True, True
                break

            resid = (op(psi) - E * psi).norm().to(real_dtype)

            if resid < best_resid:
                best_state, best_energy, best_resid = psi, E, resid

            if resid < residual_tolerance:
                cycle_converged = True
                break

            lanczos_vectors.append(w / beta)

        # Remove in pref to local.Update global best
        if best_resid < best_resid_global:
            best_state_global = best_state
            best_energy_global = best_energy
            best_resid_global = best_resid

        if cycle_converged:
            converged = True
            happy_breakdown = cycle_happy
            break

        # Not converged: explicit restart with best vector from this cycle
        psi_start = best_state

    assert best_state_global is not None and best_energy_global is not None

    return KrylovEnergyResult(
        ground_state=best_state_global,
        ground_energy=float(best_energy_global.item()),
        residual_norm=float(best_resid_global.item()),
        converged=converged,
        happy_breakdown=happy_breakdown,
        iteration_count=int(total_iters),
        restart_count=int(restart_count),
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
