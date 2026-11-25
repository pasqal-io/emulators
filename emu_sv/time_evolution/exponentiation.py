import cmath
import math
import torch


from emu_base.math.chebyshev_exp import chebyshev_exp
from emu_base.math.krylov_exp import DEFAULT_MAX_KRYLOV_DIM, krylov_exp
from emu_sv.hamiltonians import RydbergHamiltonian, RydbergLindbladian
from emu_sv.time_evolution.interface import Exponentiation

# Steps with a larger alpha = r dt are split into equal sub-steps (rounding control).
CHEBYSHEV_MAX_ALPHA = 2000.0
# With jump noise the operator is non-Hermitian and ‖Tₖ‖ can reach ρᵏ, amplifying
# roundoff by up to ρᴷ ≈ exp(|α| ln ρ): split the step to cap that growth at e⁶ ≈ 400,
# keeping the amplified roundoff ≲ 1e-13.
CHEBYSHEV_MAX_LOG_GROWTH = 6.0


class Lanczos(Exponentiation):
    """Krylov subspace exponentiation (Lanczos, or Arnoldi for non-Hermitian H)."""

    def __init__(self, max_krylov_dim: int = DEFAULT_MAX_KRYLOV_DIM) -> None:
        self.max_krylov_dim = max_krylov_dim

    def __call__(
        self,
        ham: RydbergHamiltonian | RydbergLindbladian,
        dt: float,
        state: torch.Tensor,
        tolerance: float,
    ) -> torch.Tensor:
        is_hermitian = isinstance(ham, RydbergHamiltonian) and ham.noise is None
        return krylov_exp(
            ham.make_op(-1j * dt),
            state,
            norm_tolerance=tolerance,
            exp_tolerance=tolerance,
            is_hermitian=is_hermitian,
            max_krylov_dim=self.max_krylov_dim,
        )


class Chebyshev(Exponentiation):
    """
    Chebyshev expansion of exp(-i dt H), see `chebyshev_exp`, for a
    `RydbergHamiltonian` — Hermitian, or with Lindblad jump noise H = H₀ - iΓ as in
    the Monte Carlo stepper. Keeps three vectors in addition to the in- and output,
    whatever the step size.

    The numerical range of H lies in the rectangle [lower, upper] × i[im_lo, im_up]
    of `spectral_bounds`. Shifting by the complex centre c and scaling by the real
    half-width r maps it into [-1, 1] × [-ε, ε], which fits in the Bernstein ellipse
    E_ρ with ρ = √(1+s) + √s, s = ε(ε + √(ε²+4))/2 (the smallest such ellipse, from
    the corner condition 1/cosh²t + ε²/sinh²t = 1 with ρ = eᵗ). Without noise ε = 0
    and ρ = 1; so too for dephasing and depolarizing, whose Γ is a multiple of the
    identity, absorbed entirely by the imaginary shift. If r = 0 (no drive and a
    constant real diagonal, but noise off-diagonals), r is the imaginary half-width.
    """

    def __call__(
        self,
        ham: RydbergHamiltonian | RydbergLindbladian,
        dt: float,
        state: torch.Tensor,
        tolerance: float,
    ) -> torch.Tensor:
        if not isinstance(ham, RydbergHamiltonian):
            raise ValueError(
                "Chebyshev exponentiation needs a RydbergHamiltonian; use Lanczos "
                "for density-matrix (Lindbladian) evolution."
            )
        d, lower, upper, offdiag_bound, im_lower, im_upper = ham.spectral_bounds()
        noise = ham.noise
        if offdiag_bound == 0.0 and (
            noise is None or bool((noise[0, 1] == 0) and (noise[1, 0] == 0))
        ):  # H is diagonal (d includes the imaginary noise part)
            return torch.exp(-1j * dt * d) * state
        # exp(-i dt H) = exp(-i dt c) exp(-i α (H - c)/r), with c the complex centre
        # of the rectangle bounding the numerical range, r its real half-width and
        # α = r dt: the numerical range of (H - c)/r lies in [-1, 1] × [-ε, ε]
        shift = complex((upper + lower) / 2, (im_upper + im_lower) / 2)
        radius = (upper - lower) / 2
        if radius == 0.0:  # no drive and a constant real diagonal: ε = 1
            radius = (im_upper - im_lower) / 2
        epsilon = (im_upper - im_lower) / (2 * radius)
        if epsilon > 0.0:
            s = epsilon * (epsilon + math.sqrt(epsilon * epsilon + 4.0)) / 2
            rho = math.sqrt(1.0 + s) + math.sqrt(s)
            tolerance /= 1.0 + math.sqrt(2.0)  # Crouzeix–Palencia constant
        else:
            rho = 1.0
        alpha = radius * dt
        substeps = max(
            1,
            math.ceil(abs(alpha) / CHEBYSHEV_MAX_ALPHA),
            math.ceil(abs(alpha) * math.log(rho) / CHEBYSHEV_MAX_LOG_GROWTH),
        )
        op = ham.shifted(shift, d).make_op(2 / radius)
        del d
        phase = cmath.exp(-1j * dt * shift / substeps)
        tolerance = tolerance / substeps
        res = state
        for _ in range(substeps):
            res = chebyshev_exp(op, res, alpha / substeps, phase, tolerance, rho)
        return res
