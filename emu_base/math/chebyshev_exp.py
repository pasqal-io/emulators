import math
from typing import Callable

import torch


def exp_chebyshev_coefficients(
    alpha: float, tolerance: float, rho: float = 1.0
) -> list[complex]:
    """
    Coefficients cₖ of the Chebyshev expansion of exp(-iαz),

        exp(-iαz) = ∑ₖ cₖ Tₖ(z),   c₀ = J₀(α),   cₖ = 2(-i)ᵏ Jₖ(α),

    truncated after the first K+1 terms, K being the smallest index such that
    ∑ₖ﹥K |cₖ| ρᵏ ≤ tolerance. exp is entire, so the series converges on all of ℂ,
    and |Tₖ(z)| ≤ ρᵏ on the Bernstein ellipse E_ρ (foci ±1, ρ the sum of the semi-
    axes): the truncation bounds the error of the series on E_ρ. The default ρ = 1
    degenerates to [-1, 1] and |Tₖ(x)| ≤ 1, for an operator with spectrum there.
    `chebyshev_exp` picks ρ > 1 for a non-Hermitian operator, whose numerical range
    sticks out of the real axis.

    The Bessel functions Jₖ(|α|) are computed with Miller's backward recurrence,
    normalised by J₀ + 2∑ₖJ₂ₖ = 1. For α < 0, cₖ(α) is the conjugate of cₖ(|α|).
    """
    if not math.isfinite(alpha):
        raise ValueError(f"alpha must be finite, got {alpha}")
    assert rho >= 1.0
    a = abs(alpha)
    if a < 1e-50:
        # c₀ = 1 and c₁ = -iα in double precision, |c₂|ρ² ≈ (αρ/2)² is below any
        # useful tolerance, and the recurrence below would overflow.
        return [1.0 + 0.0j] if a * rho <= tolerance else [1.0 + 0.0j, -1.0j * alpha]
    # Jₖ(a) decays faster than exponentially once k > a + O(a^(1/3)). Iterate backwards,
    # so we find Jₖ(a), and not Yₖ(a), which has the same recurrence but grows.
    # The 12 α^{1/3} term puts J_start ≲ 1e-17 (below double-precision roundoff),
    # in the regime where J_k(α) ≈ (2/α)^{1/3} Ai(t).
    # The Airy estimate is only accurate for large α; for α ≲ 10, more terms are needed.
    # For the extra 60 do J_k(α) ≈ (α/2)^k/k! from the power series and take a margin.
    # In that decay regime ρᵏJₖ(α) ≈ Jₖ(αρ), so inflating the argument by ρ puts the
    # ρ-weighted tail below roundoff at the start index as well.
    start = int(a * rho + 12.0 * (a * rho) ** (1.0 / 3.0) + 60.0)
    values = [0.0] * (start + 1)
    values[start] = 1.0  # find a scaled solution by starting the recurrence at 1
    j_next, j_cur = 0.0, 1.0
    for k in range(start, 0, -1):
        # the recursion relation j_(n-1)(x) = 2n j_n(x)/x - j_(n+1)(x)
        j_next, j_cur = j_cur, 2.0 * k / a * j_cur - j_next
        values[k - 1] = j_cur
        if abs(j_cur) > 1e200:  # rescale to avoid overflow
            for i in range(k - 1, start + 1):
                values[i] *= 1e-200
            j_next *= 1e-200
            j_cur *= 1e-200
    norm = values[0] + 2.0 * sum(values[2::2])  # the normalization condition
    bessel = [v / norm for v in values]  # the normalized solution

    tail = 0.0
    rho_pow = rho**start  # inf is fine: the loop then keeps every coefficient
    last = start  # find K by summing the ρ-weighted tail until we hit tolerance
    while last > 0 and tail + 2.0 * abs(bessel[last]) * rho_pow <= tolerance:
        tail += 2.0 * abs(bessel[last]) * rho_pow
        rho_pow /= rho
        last -= 1

    powers = (1.0 + 0.0j, -1.0j, -1.0 + 0.0j, 1.0j)  # (-i)ᵏ
    coefs = [complex(bessel[0])] + [  # put in the prefactor for k>0
        2.0 * powers[k % 4] * bessel[k] for k in range(1, last + 1)
    ]
    if alpha < 0.0:
        coefs = [c.conjugate() for c in coefs]
    return coefs


def chebyshev_exp(
    op: Callable[[torch.Tensor], torch.Tensor],
    v: torch.Tensor,
    alpha: float,
    phase: complex,
    tolerance: float,
    rho: float = 1.0,
) -> torch.Tensor:
    """
    Computes phase·exp(-iαA)v by its Chebyshev series, where op(x) = 2Ax and the
    numerical range of A lies in the Bernstein ellipse E_ρ (see
    `exp_chebyshev_coefficients`). With the default ρ = 1, A is Hermitian with
    spectrum in [-1, 1] and the truncation error is ≤ tolerance·‖v‖ in exact
    arithmetic. With ρ > 1, A may be non-Hermitian (numerical range in a strip
    [-1, 1] × [-ε, ε] ⊆ E_ρ); the error is then ≤ (1+√2)·tolerance·‖v‖ by the
    Crouzeix–Palencia theorem ‖p(A)‖ ≤ (1+√2)·sup over the numerical range of |p|.

    Calls `op` K times, K + 1 being the number of coefficients. `v` is not modified.

    Unlike the power series of exp, this is numerically stable since |Tₖ(x)| ≤ 1 on
    [-1, 1]. The number of iterations is ~alpha, which is good since Tₖ(x) has the
    same number of oscillations as exp(ikx), so it needs at least alpha. For ρ > 1,
    ‖Tₖ(A)‖ can reach ρᵏ and amplifies roundoff by up to ρᴷ: the caller must keep
    K·ln ρ moderate by splitting the step (see `CHEBYSHEV_MAX_LOG_GROWTH` in emu-sv).
    """
    coefs = [phase * c for c in exp_chebyshev_coefficients(alpha, tolerance, rho)]
    acc = coefs[0] * v
    if len(coefs) == 1:
        return acc
    cur = op(v)
    cur.mul_(0.5)  # T₁ = Av
    acc.add_(cur, alpha=coefs[1])
    prev = v
    for coef in coefs[2:]:
        # op = 2A to avoid a scalar multiplication here.
        # Saves a kernel launch when using a fused gpu kernel.
        out = op(cur)
        out.sub_(prev)  # Tₖ = 2ATₖ₋₁ - Tₖ₋₂
        acc.add_(out, alpha=coef)
        prev, cur = cur, out
    return acc
