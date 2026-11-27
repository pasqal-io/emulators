import cmath
import logging
import math
import typing
import torch


from emu_base.math.chebyshev_exp import chebyshev_exp, exp_chebyshev_coefficients
from emu_base.math.krylov_exp import (
    DEFAULT_MAX_KRYLOV_DIM,
    KRYLOV_EXTRA_VECTORS,
    krylov_exp_impl,
)
from emu_base.utils import available_memory
from emu_sv.hamiltonians import RydbergHamiltonian, RydbergLindbladian
from emu_sv.time_evolution.interface import Exponentiation

# Steps with a larger alpha = r dt are split into equal sub-steps (rounding control).
CHEBYSHEV_MAX_ALPHA = 2000.0
# With jump noise the operator is non-Hermitian and ‖Tₖ‖ can reach ρᵏ, amplifying
# roundoff by up to ρᴷ ≈ exp(|α| ln ρ): split the step to cap that growth at e⁶ ≈ 400,
# keeping the amplified roundoff ≲ 1e-13.
CHEBYSHEV_MAX_LOG_GROWTH = 6.0

# The a priori Lanczos iteration count (see `Lanczos`) may use this fraction of the
# Krylov cap, the rest being headroom for the finite-precision delay of convergence.
LANCZOS_ITERATION_MARGIN = 0.8
# Fraction of the available memory a Lanczos call may fill, the rest is headroom
# for fragmentation and for the workspace (a few MB, whatever the state size)
# cuBLAS takes from the allocator during each dot product on recent torch.
LANCZOS_MEMORY_FRACTION = 0.8
# A step is halved at most this many times before giving up.
LANCZOS_MAX_SPLITS = 20


def bernstein_frame(
    lower: float, upper: float, im_lower: float, im_upper: float
) -> tuple[complex, float, float]:
    """
    (shift, radius, ρ) mapping the rectangle [lower, upper] × i[im_lower, im_upper]
    bounding the numerical range of H (see `RydbergHamiltonian.spectral_bounds`)
    into the Bernstein ellipse E_ρ: (H - shift)/radius has its numerical range in
    [-1, 1] × [-ε, ε] ⊆ E_ρ with ρ = √(1+s) + √s, s = ε(ε + √(ε²+4))/2 (the
    smallest such ellipse, from the corner condition 1/cosh²t + ε²/sinh²t = 1 with
    ρ = eᵗ). Without noise ε = 0 and ρ = 1; so too for dephasing and depolarizing,
    whose Γ is a multiple of the identity, absorbed entirely by the imaginary shift.
    If radius would be 0 (no drive and a constant real diagonal, but noise
    off-diagonals), the imaginary half-width is used instead.
    """
    shift = complex((upper + lower) / 2, (im_upper + im_lower) / 2)
    radius = (upper - lower) / 2
    if radius == 0.0:  # no drive and a constant real diagonal: ε = 1
        radius = (im_upper - im_lower) / 2
    epsilon = (im_upper - im_lower) / (2 * radius) if radius > 0.0 else 0.0
    if epsilon > 0.0:
        s = epsilon * (epsilon + math.sqrt(epsilon * epsilon + 4.0)) / 2
        rho = math.sqrt(1.0 + s) + math.sqrt(s)
    else:
        rho = 1.0
    return shift, radius, rho


class Lanczos(Exponentiation):
    """
    Krylov subspace exponentiation (Lanczos, or Arnoldi for non-Hermitian H).

    exp(-i dt H) = exp(-i (dt/s) H)ˢ exactly, so a step is split into s equal
    sub-steps, each a Krylov call within the cap on the Krylov dimension, which is
    `max_krylov_dim` or fewer if the basis vectors would not fit in memory (each
    iteration stores one state-sized vector, see `krylov_cap`). The split is chosen
    a priori for a `RydbergHamiltonian`: by Saad's bound, the Lanczos error after m
    iterations is at most twice the best uniform polynomial approximation of exp
    of degree m - 1 on the spectrum, and the truncated Chebyshev series (see
    `exp_chebyshev_coefficients`) is such a polynomial, so its term count bounds
    the iteration count — for non-Hermitian H up to the Crouzeix–Palencia constant
    1 + √2, on the Bernstein ellipse enclosing the numerical range. The bound uses
    the Gershgorin width of the spectrum, so it is conservative. A sub-step that
    still does not converge, or runs out of GPU memory, is halved and retried: its
    input is restored by `krylov_exp_impl` and the tolerance stays proportional to
    the sub-step, so the total error stays at the tolerance. The Lindbladian has
    no spectral bounds, so density matrices rely on that retry alone.
    """

    def __init__(self, max_krylov_dim: int = DEFAULT_MAX_KRYLOV_DIM) -> None:
        self.max_krylov_dim = max_krylov_dim
        self._memory_capped = False  # whether the memory cap has been logged

    def krylov_cap(
        self, ham: RydbergHamiltonian | RydbergLindbladian, state: torch.Tensor
    ) -> int:
        """
        The Krylov dimension a call on `state` may build: `max_krylov_dim`, or, if
        smaller, what fits in `LANCZOS_MEMORY_FRACTION` of the memory available on
        the state's device. A call with a cap of m holds at most m + max(k, n)
        state-sized tensors, k being `KRYLOV_EXTRA_VECTORS` and n the
        `working_vectors` of the Hamiltonian's operator (see `krylov_exp_impl`).
        """
        available = available_memory(state.device)
        if available is None:
            return self.max_krylov_dim
        bytes_per_vector = state.numel() * state.element_size()
        fits = int(LANCZOS_MEMORY_FRACTION * available / bytes_per_vector) - max(
            KRYLOV_EXTRA_VECTORS, ham.working_vectors
        )
        if fits < 2:
            raise MemoryError(
                f"Not enough memory on {state.device} for Lanczos: a state takes "
                f"{bytes_per_vector / 1e6:.0f} MB and {available / 1e6:.0f} MB are "
                "available."
            )
        if fits < self.max_krylov_dim and not self._memory_capped:
            self._memory_capped = True
            logging.getLogger("emulators").info(
                f"Memory caps the Krylov dimension at {fits} instead of "
                f"{self.max_krylov_dim}; time steps are split accordingly."
            )
        return min(self.max_krylov_dim, fits)

    def a_priori_substeps(
        self,
        ham: RydbergHamiltonian | RydbergLindbladian,
        dt: float,
        tolerance: float,
        cap: int,
    ) -> int:
        """
        The number of equal sub-steps of `dt` whose bound on the Krylov iteration
        count (the Chebyshev term count, see the class docstring) fits in
        `LANCZOS_ITERATION_MARGIN` × `cap`. 1 for a Lindbladian.
        """
        if not isinstance(ham, RydbergHamiltonian):
            return 1
        d, lower, upper, offdiag_bound, im_lower, im_upper = ham.spectral_bounds()
        del d
        if offdiag_bound == 0.0 and im_lower == im_upper:
            return 1  # H is diagonal up to the noise: a happy breakdown in ≤ 2 steps
        _, radius, rho = bernstein_frame(lower, upper, im_lower, im_upper)
        alpha = radius * abs(dt)
        budget = max(2, int(LANCZOS_ITERATION_MARGIN * cap))
        tolerance /= 2.0  # Saad's factor
        if rho > 1.0:
            tolerance /= 1.0 + math.sqrt(2.0)  # Crouzeix–Palencia constant
        substeps = max(1, math.ceil(alpha / budget))  # the term count is ≥ α
        while (
            len(exp_chebyshev_coefficients(alpha / substeps, tolerance / substeps, rho))
            > budget
        ):
            substeps += 1
        return substeps

    def __call__(
        self,
        ham: RydbergHamiltonian | RydbergLindbladian,
        dt: float,
        state: torch.Tensor,
        tolerance: float,
    ) -> torch.Tensor:
        is_hermitian = isinstance(ham, RydbergHamiltonian) and ham.noise is None
        cap = self.krylov_cap(ham, state)
        remaining = self.a_priori_substeps(ham, dt, tolerance, cap)
        substep = dt / remaining
        substep_tolerance = tolerance / remaining
        op = ham.make_op(-1j * substep)
        splits = 0
        while remaining > 0:
            result = self._try_substep(op, state, is_hermitian, substep_tolerance, cap)
            if result is not None:
                state = result
                remaining -= 1
                continue
            if splits == LANCZOS_MAX_SPLITS:
                raise RecursionError(
                    "exponentiation algorithm did not converge to precision in "
                    f"allotted number of steps, even after halving the time step "
                    f"{splits} times."
                )
            splits += 1
            remaining *= 2
            substep /= 2
            substep_tolerance /= 2
            op = ham.make_op(-1j * substep)
            logging.getLogger("emulators").info(
                f"Lanczos step halved to {substep:.3e} (Krylov cap {cap})."
            )
        return state

    @staticmethod
    def _try_substep(
        op: typing.Callable[[torch.Tensor], torch.Tensor],
        state: torch.Tensor,
        is_hermitian: bool,
        tolerance: float,
        cap: int,
    ) -> torch.Tensor | None:
        """
        exp(op) state, or None if the Krylov iteration did not converge within `cap`
        iterations or ran out of GPU memory, `state` being restored either way (see
        `krylov_exp_impl`).
        """
        try:
            result = krylov_exp_impl(
                op,
                state,
                is_hermitian=is_hermitian,
                exp_tolerance=tolerance,
                norm_tolerance=tolerance,
                max_krylov_dim=cap,
            )
        except torch.cuda.OutOfMemoryError:
            result = None
        if result is None:
            # out of the except clause, the exception and its traceback are gone, so
            # the Krylov basis is released: hand its memory back to the allocator
            torch.cuda.empty_cache()
            return None
        return result.result if result.converged else None


class Chebyshev(Exponentiation):
    """
    Chebyshev expansion of exp(-i dt H), see `chebyshev_exp`, for a
    `RydbergHamiltonian` — Hermitian, or with Lindblad jump noise H = H₀ - iΓ as in
    the Monte Carlo stepper. Keeps three vectors in addition to the in- and output,
    whatever the step size.

    The numerical range of H lies in the rectangle [lower, upper] × i[im_lo, im_up]
    of `spectral_bounds`; `bernstein_frame` maps it into the Bernstein ellipse E_ρ.
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
        shift, radius, rho = bernstein_frame(lower, upper, im_lower, im_upper)
        if rho > 1.0:
            tolerance /= 1.0 + math.sqrt(2.0)  # Crouzeix–Palencia constant
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
