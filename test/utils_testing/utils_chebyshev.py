"""
Independent oracles and error budgets for the Chebyshev expansion of exp(-i alpha x).

exp(-i alpha x) = sum_k c_k T_k(x) on [-1, 1], with c_0 = J_0(alpha) and
c_k = 2 (-i)^k J_k(alpha) (Jacobi-Anger). Neither oracle uses the Bessel recurrence of
the code under test:
- `chebyshev_coefficients_exact`: power series of J_k in decimal arithmetic, accurate far
  below double precision (coefficient tests, |alpha| <= a few hundred);
- `chebyshev_coefficients_oracle`: FFT of exp(-i alpha cos(theta)), cheap for any alpha,
  accurate to ~|alpha| eps (error budgets only).
"""

import math
from decimal import Decimal, localcontext
from functools import lru_cache

import torch

from emu_sv.time_evolution.torch import CHEBYSHEV_MAX_ALPHA

EPS = torch.finfo(torch.float64).eps


def _significant_length(alpha: float) -> int:
    """Number of c_k above 1e-30: J_k(alpha) decays super-exponentially past the
    turning point k = alpha (Airy regime of width alpha**(1/3))."""
    a = abs(alpha)
    return int(a + 12 * a ** (1 / 3) + 60) + 1


def chebyshev_coefficients_oracle(alpha: float) -> torch.Tensor:
    """
    c_0, c_1, ... of exp(-i alpha x), complex128, from the FFT of
    f(theta) = exp(-i alpha cos theta) = sum_k (-i)^k J_k(alpha) e^{i k theta}.
    The FFT on M points returns sum_{m = k mod M} (-i)^m J_m(alpha); with
    M >= 4 * _significant_length(alpha), the aliased terms are below 1e-30.
    """
    length = _significant_length(alpha)
    size = 2 ** math.ceil(math.log2(4 * length))
    theta = 2 * math.pi * torch.arange(size, dtype=torch.float64) / size
    a = torch.fft.fft(torch.exp(-1j * alpha * torch.cos(theta))) / size
    coefs = a[:length].clone()
    coefs[1:] *= 2
    return coefs


@lru_cache
def _bessel_exact(a: float) -> tuple[Decimal, ...]:
    """
    J_0(a), ..., J_{L-1}(a) for a >= 0, L = _significant_length(a), from
    J_k(a) = sum_m (-1)^m (a/2)^(2m+k) / (m! (m+k)!) in decimal arithmetic (a is exact).
    The terms sum in absolute value to I_k(a) <= e^a, so a / ln(10) + 50 digits leave
    50 correct digits after cancellation; the series is cut once its terms decrease and
    are below 1e-60 (alternating: the remainder is smaller).
    """
    with localcontext() as ctx:
        ctx.prec = int(a / math.log(10)) + 50
        x = Decimal(a) / 2
        x2 = x * x
        tiny = Decimal(10) ** -60
        values = []
        first = Decimal(1)  # (a/2)^k / k!
        for k in range(_significant_length(a)):
            term, total, m = first, first, 0
            while not (m * (m + k) > x2 and abs(term) < tiny):
                m += 1
                term = -term * x2 / (m * (m + k))
                total += term
            values.append(+total)
            first = first * x / (k + 1)
    return tuple(values)


def chebyshev_coefficients_exact(alpha: float) -> tuple[torch.Tensor, list[Decimal]]:
    """
    c_0, c_1, ... of exp(-i alpha x) rounded to complex128 (error <= eps |c_k| / 2), and
    the tails sum_{j > k} |c_j| in decimal arithmetic (for the truncation index).
    J_k(-a) = (-1)^k J_k(a); terms beyond _significant_length are below 1e-30.
    """
    bessel = _bessel_exact(abs(alpha))
    power = (1, -1j, -1, 1j) if alpha >= 0 else (1, 1j, -1, -1j)  # (-i)^k (sign alpha)^k
    coefs = torch.tensor(
        [complex(float(bessel[0]))]
        + [2 * power[k % 4] * float(bessel[k]) for k in range(1, len(bessel))],
        dtype=torch.complex128,
    )
    tails = [Decimal(0)] * len(bessel)
    for k in range(len(bessel) - 2, -1, -1):
        tails[k] = tails[k + 1] + 2 * abs(bessel[k + 1])
    return coefs, tails


def tails(coefs: torch.Tensor) -> torch.Tensor:
    """tails[k] = sum_{j > k} |c_j|"""
    a = coefs.abs()
    suffix = torch.flip(torch.cumsum(torch.flip(a, [0]), 0), [0])
    return torch.cat((suffix[1:], torch.zeros(1, dtype=a.dtype)))


def truncation_index(coefs: torch.Tensor, tolerance: float) -> int:
    """Smallest K with sum_{k > K} |c_k| <= tolerance."""
    return int(torch.nonzero(tails(coefs) <= tolerance)[0])


def chebyshev_rounding_bound(coefs: torch.Tensor, local_error: float) -> float:
    """
    First-order bound on the rounding error of phase * sum_k c_k T_k(Ht) v, computed with
    the recurrence T_k = 2 Ht T_{k-1} - T_{k-2}, for ||v|| = 1 and the spectrum of Ht
    in [-1, 1]:
    - a local error delta_j at term j propagates as e_k = sum_{j<=k} U_{k-j}(Ht) delta_j,
      with ||U_m(Ht)|| <= m + 1 (Chebyshev polynomials of the second kind), so the
      error of the sum is <= sum_j ||delta_j|| sum_{k>=j} |c_k| (k - j + 1);
    - accumulation into the sum: (K + 1) eps sum_k |c_k|.
    `local_error` bounds ||delta_j|| (one application of the operator, one subtraction).
    This is a worst-case bound, growing as alpha**2.5 eps.
    """
    a = coefs.abs()
    k = torch.arange(len(a), dtype=torch.float64)
    s0 = torch.flip(torch.cumsum(torch.flip(a, [0]), 0), [0])
    s1 = torch.flip(torch.cumsum(torch.flip(a * (k + 1), [0]), 0), [0])
    beta = s1 - k * s0  # sum_{k' >= j} |c_k'| (k' - j + 1)
    return float(local_error * beta.sum() + len(a) * EPS * a.sum())


def matrix_exp_error(h: torch.Tensor, dt: float) -> float:
    """
    Estimate (not a proven bound) of ||matrix_exp(-i dt h) v - exp(-i dt h) v|| for a
    unit vector v and a Hermitian h: scaling and squaring has a backward error of order
    eps (Higham 2005), amplified by the norm of the argument and summed over dim terms.
    """
    norm = torch.linalg.eigvalsh(h).abs().max().item() * abs(dt)
    return h.shape[0] * EPS * max(1.0, norm)


def chebyshev_step_budget(
    h: torch.Tensor, omegas: torch.Tensor, dt: float, tolerance: float
) -> float:
    """
    Budget on ||chebyshev step - matrix_exp(-i dt h) v|| for a unit vector v, where h is
    the dense Rydberg Hamiltonian built from `omegas` (raw Rabi frequencies) and
    `tolerance` is the truncation tolerance of the whole step.
    Spectral bounds as in the code under test, from independent inputs: the dense
    diagonal and R = sum_j |Omega_j| / 2 (Gershgorin). Sub-steps of
    alpha <= CHEBYSHEV_MAX_ALPHA, each with tolerance / m. The operator 2 (H - c) / r
    applied by the torch path sums N + 1 terms per row with absolute row sums <= 2:
    local error <= (2 (N + 1) + 3) eps.
    """
    nqubits = len(omegas)
    d = h.diagonal().real
    radius = omegas.abs().sum().item() / 2
    r = (d.max() - d.min()).item() / 2 + radius
    alpha = r * dt
    m = max(1, math.ceil(abs(alpha) / CHEBYSHEV_MAX_ALPHA))
    coefs = chebyshev_coefficients_oracle(alpha / m)
    K = truncation_index(coefs, tolerance / m)
    local = (2 * (nqubits + 1) + 3) * EPS
    per_substep = tolerance / m + chebyshev_rounding_bound(coefs[: K + 1], local)
    return m * per_substep + matrix_exp_error(h, dt)
