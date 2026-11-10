import cmath
import math
from abc import ABC, abstractmethod

import torch

from emu_base.math.chebyshev_exp import chebyshev_exp
from emu_base.math.krylov_exp import DEFAULT_MAX_KRYLOV_DIM, krylov_exp
from emu_sv.hamiltonians import RydbergHamiltonian, RydbergLindbladian

# The Chebyshev series is truncated where its error bound reaches this fraction of
# the tolerance: the Lanczos stopping test overestimates the error (see
# docs/emu_sv/benchmarks/accuracy.md), whereas the Chebyshev bound is nearly attained.
CHEBYSHEV_TOLERANCE_FACTOR = 1e-2
# Steps with a larger alpha = r dt are split into equal sub-steps (rounding control).
CHEBYSHEV_MAX_ALPHA = 2000.0


class Exponentiation(ABC):
    """Algorithm computing exp(-i dt H)|ψ❭ for a time step where H is constant."""

    @abstractmethod
    def __call__(
        self,
        ham: RydbergHamiltonian | RydbergLindbladian,
        dt: float,
        state: torch.Tensor,
        tolerance: float,
    ) -> torch.Tensor:
        pass


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
    Chebyshev expansion of exp(-i dt H), see `chebyshev_exp`. Only for a Hermitian
    `RydbergHamiltonian` (no noise). Keeps three vectors, whatever the step size.
    """

    def __call__(
        self,
        ham: RydbergHamiltonian | RydbergLindbladian,
        dt: float,
        state: torch.Tensor,
        tolerance: float,
    ) -> torch.Tensor:
        if not isinstance(ham, RydbergHamiltonian) or ham.noise is not None:
            raise ValueError(
                "Chebyshev exponentiation needs a Hermitian RydbergHamiltonian."
            )
        d, lower, upper = ham.spectral_bounds()
        if not ham.omegas.any():  # H is diagonal
            return torch.exp(-1j * dt * d) * state
        del d
        # exp(-i dt H) = exp(-i dt c) exp(-i α (H - c)/r), with c the shift, r the
        # radius and α = r dt: the spectrum of (H - c)/r lies in [-1, 1]
        shift, radius = (upper + lower) / 2, (upper - lower) / 2
        alpha = radius * dt
        substeps = max(1, math.ceil(abs(alpha) / CHEBYSHEV_MAX_ALPHA))
        op = ham.shifted(shift).make_op(2 / radius)
        phase = cmath.exp(-1j * dt * shift / substeps)
        tolerance = tolerance * CHEBYSHEV_TOLERANCE_FACTOR / substeps
        res = state
        for _ in range(substeps):
            res = chebyshev_exp(op, res, alpha / substeps, phase, tolerance)
        return res
