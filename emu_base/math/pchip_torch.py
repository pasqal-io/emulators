"""
Reference:
  See page 98 (a few pages are sufficient) in:
  C. Moler, Numerical Computing with MATLAB, 2004.

PCHIP - Piecewise Cubic Hermite Interpolating Polynomial is a 1D, C¹-continuous
interpolation method. It builds a separate cubic polynomial P_i(x) on each
interval [x[i], x[i+1]] and is designed to preserve the shape of the data
(literally means visually pleasing) e.g., avoid overshoot near sharp changes.

The cubic polynomials P_i(x) are expressed in Hermite form, which means:
(1) they match the data values
    P(x[i])     == y[i],
    P(x[i+1])   == y[i+1].
(2) they match derivatives
    P(x[i])'    == d[i],
    P(x[i+1])'  == d[i+1],
where d[i] is a derivative better then the first order finite difference
Δ[i] = (y[i+1]-y[i])/(x[i+1]-x[i]). How to find d[i] and what better means
is explained below.

What distinguishes PCHIP from generic cubic spline interpolators is how PCHIP
selects the knot derivatives d[i]. PCHIP computes:
(1) Secant slopes (finite differences) on each interval:
    Δ[i] = (y[i+1] - y[i]) / (x[i+1] - x[i]).

(2) The knot derivatives d[i] as a harmonic mean of the neighboring slopes:
        1/d[i] = 0.5 * (1/Δ[i-1] + 1/Δ[i]),
    which interpolates visually better compare to the
    - arithmetic mean d[i] = 0.5 * (Δ[i-1] + Δ[i]), or
    - finite difference d[i] = Δ[i].

    If Δ[i-1] and Δ[i] have opposite signs (or either is zero), x[i]
    is a discrete local minimum, maximum or valley. In this case PCHIP sets
        d[i] = 0
    so the curve flattens at x[i], which helps prevent overshoot and
    preserves the shape of the data.
    (See images https://matthodges.com/posts/2024-08-08-spline-pchip/)

If the grid is non-uniform (h[i-1] = x[i] - x[i-1] and h[i] = x[i+1] - x[i]
are different), the neighboring secant slopes Δ[i-1] and Δ[i] should not be
treated equally. PCHIP therefore combines them using a weighted harmonic mean:
    (w1+w2)/d[i] = w1/Δ[i-1] + w2/Δ[i].
w1 = 2h[i] + h[i-1], w2 = h[i] + 2h[i-1],
For uniform spacing this reduces to the simple harmonic mean w1 = w2 = 0.5

Algorithm outline:
  1) Compute interval widths h[i] and secant slopes delta Δ[i].
  2) Compute knot derivatives d[i] using PCHIP.
  3) For each interval [x[i], x[i+1]],
    form the cubic Hermite polynomial P(x) using y[i], and d[i].
  4) Evaluate P(x) at desired query points xq.

Note: end points x[0] and x[T] are treated a bit differently compare to bulk values x[i]
"""

import torch


def _weighted_harmonic_mean(
    s_l: torch.Tensor,
    s_r: torch.Tensor,
    h_l: torch.Tensor,
    h_r: torch.Tensor,
) -> torch.Tensor:

    w_l = h_l + 2.0 * h_r
    w_r = 2.0 * h_l + h_r
    return (w_l + w_r) / (w_l / s_l + w_r / s_r)


def _endpoint_slope(
    s0: torch.Tensor,
    s1: torch.Tensor,
    h0: torch.Tensor,
    h1: torch.Tensor,
) -> torch.Tensor:
    # One-sided 3-point estimate at the boundary
    w1 = 2.0 * h0 + h1
    return (w1 * s0 - h0 * s1) / (h0 + h1)


def _limit_endpoint(
    d_end: torch.Tensor,
    s0: torch.Tensor,
    s1: torch.Tensor,
) -> torch.Tensor:
    # 1) If derivative points opposite to the first secant, zero it
    mask_neq_sign = torch.sign(d_end) != torch.sign(s0)
    d_end = torch.where(mask_neq_sign, torch.zeros_like(d_end), d_end)

    # 2) If secants switch sign, cap magnitude to 3*|s0|
    mask_cap = (torch.sign(s0) != torch.sign(s1)) & (
        torch.abs(d_end) > 3.0 * torch.abs(s0)
    )
    return torch.where(mask_cap, 3.0 * s0, d_end)


def _pchip_derivatives(
    h: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    """
    Compute PCHIP knot derivatives d[i] from interval widths h and secant slopes delta.
    """
    n = h.numel() + 1
    d = torch.zeros((n,), dtype=delta.dtype, device=delta.device)

    # Two points: straight line
    if n == 2:
        d.fill_(delta[0])
        return d

    # Interior points
    s_l, s_r = delta[:-1], delta[1:]
    h_l, h_r = h[:-1], h[1:]

    mask_same_sign = (s_l * s_r) > 0  # excludes zeros + sign changes
    dh = _weighted_harmonic_mean(s_l, s_r, h_l, h_r)
    d[1:-1] = torch.where(mask_same_sign, dh, torch.zeros_like(dh))

    # Endpoints (one-sided + limiter)
    d0 = _endpoint_slope(delta[0], delta[1], h[0], h[1])
    dn = _endpoint_slope(delta[-1], delta[-2], h[-1], h[-2])

    d[0] = _limit_endpoint(d0, delta[0], delta[1])
    d[-1] = _limit_endpoint(dn, delta[-1], delta[-2])

    return d


def _interval_coeffs(
    y: torch.Tensor,
    h: torch.Tensor,
    delta: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    """
    For each interval i, build cubic in local coordinate t(x) = x - x[i]:

        P(t) = p0 + p1 t + p2 t^2 + p3 t^3,   0 <= t <= h[i]
    """
    p0 = y[:-1]
    p1 = d[:-1]
    p2 = (3.0 * delta - 2.0 * d[:-1] - d[1:]) / h
    p3 = (d[:-1] + d[1:] - 2.0 * delta) / (h * h)
    return torch.stack([p0, p1, p2, p3], dim=-1)  # (N-1, 4)


class PCHIP1D:
    """
    1D PCHIP interpolator (PyTorch).

    - Shape-preserving, C¹ piecewise-cubic Hermite interpolant.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        x, y = self._validate_xy(x, y)
        self.x, self.y = x, y

        h = x[1:] - x[:-1]
        delta = (y[1:] - y[:-1]) / h

        d = _pchip_derivatives(h, delta)
        self._coeffs = _interval_coeffs(y, h, delta, d)

    @staticmethod
    def _validate_xy(
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.as_tensor(x)
        if not x.is_floating_point():
            raise TypeError("x must be a floating point tensor")

        y = torch.as_tensor(y, dtype=x.dtype, device=x.device)
        if not y.is_floating_point():
            raise TypeError("y must be a floating point tensor")

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be 1D tensors")
        if x.numel() != y.numel():
            raise ValueError("x and y must have the same length")
        if x.numel() < 2:
            raise ValueError("Need at least 2 points")
        if not torch.all(x[1:] > x[:-1]):
            raise ValueError("x must be strictly increasing")

        return x, y

    def _interval_index(self, xq: torch.Tensor) -> torch.Tensor:
        i = torch.searchsorted(self.x, xq, right=True) - 1
        return i.clamp(0, self.x.numel() - 2)

    def __call__(self, xq: torch.Tensor) -> torch.Tensor:
        xq = torch.as_tensor(xq, dtype=self.x.dtype, device=self.x.device)

        i = self._interval_index(xq)
        t = xq - self.x[i]

        p0, p1, p2, p3 = self._coeffs[i].unbind(-1)
        return p0 + t * (p1 + t * (p2 + t * p3))
