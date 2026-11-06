import math
from typing import Callable

import torch


def exp_chebyshev_coefficients(alpha: float, tolerance: float) -> list[complex]:
    """
    Coefficients cₖ of the Chebyshev expansion of exp(-iαx) on [-1, 1],

        exp(-iαx) = ∑ₖ cₖ Tₖ(x),   c₀ = J₀(α),   cₖ = 2(-i)ᵏ Jₖ(α),

    truncated after the first K+1 terms, K being the smallest index such that
    ∑ₖ﹥K |cₖ| ≤ tolerance. Since |Tₖ(x)| ≤ 1 on [-1, 1], this bounds the error of the
    truncated series applied to a unit vector, for any operator with spectrum in [-1, 1].

    The Bessel functions Jₖ(|α|) are computed with Miller's backward recurrence,
    normalised by J₀ + 2∑ₖJ₂ₖ = 1. For α < 0, cₖ(α) is the conjugate of cₖ(|α|).
    """
    if not math.isfinite(alpha):
        raise ValueError(f"alpha must be finite, got {alpha}")
    a = abs(alpha)
    if a < 1e-50:
        # c₀ = 1 and c₁ = -iα in double precision, |c₂| ≈ α²/4 is below any useful
        # tolerance, and the recurrence below would overflow.
        return [1.0 + 0.0j] if a <= tolerance else [1.0 + 0.0j, -1.0j * alpha]
    # Jₖ(a) decays faster than exponentially once k > a + O(a^(1/3)).
    start = int(a + 12.0 * a ** (1.0 / 3.0) + 60.0)
    values = [0.0] * (start + 1)
    values[start] = 1.0
    j_next, j_cur = 0.0, 1.0
    for k in range(start, 0, -1):
        j_next, j_cur = j_cur, 2.0 * k / a * j_cur - j_next
        values[k - 1] = j_cur
        if abs(j_cur) > 1e200:  # rescale to avoid overflow
            for i in range(k - 1, start + 1):
                values[i] *= 1e-200
            j_next *= 1e-200
            j_cur *= 1e-200
    norm = values[0] + 2.0 * sum(values[2::2])
    bessel = [v / norm for v in values]

    tail = 0.0
    last = start
    while last > 0 and tail + 2.0 * abs(bessel[last]) <= tolerance:
        tail += 2.0 * abs(bessel[last])
        last -= 1
    powers = (1.0 + 0.0j, -1.0j, -1.0 + 0.0j, 1.0j)  # (-i)ᵏ
    coefs = [complex(bessel[0])] + [
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
) -> torch.Tensor:
    """
    Computes phase·exp(-iαA)v by its Chebyshev series, where op(x) = 2Ax and the
    spectrum of the Hermitian operator A lies in [-1, 1]. The series is truncated
    where its tail is ≤ tolerance (see `exp_chebyshev_coefficients`), so the
    truncation error is ≤ tolerance·‖v‖ in exact arithmetic.

    Calls `op` K times, K + 1 being the number of coefficients. `v` is not modified.
    """
    coefs = [phase * c for c in exp_chebyshev_coefficients(alpha, tolerance)]
    acc = coefs[0] * v
    if len(coefs) == 1:
        return acc
    cur = op(v)
    cur.mul_(0.5)  # T₁ = Av
    acc.add_(cur, alpha=coefs[1])
    prev = v
    for coef in coefs[2:]:
        out = op(cur)
        out.sub_(prev)  # Tₖ = 2ATₖ₋₁ - Tₖ₋₂
        acc.add_(out, alpha=coef)
        prev, cur = cur, out
    return acc
