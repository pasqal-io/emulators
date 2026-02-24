import pytest
import torch

from emu_base.math.pchip_torch import (
    _weighted_harmonic_mean,
    _endpoint_slope,
    _limit_endpoint,
    _pchip_derivatives,
    PCHIP1D,
)


def test_weighted_harmonic_mean() -> None:
    s_l = torch.tensor([2.0, 1.0, 3.0])
    s_r = torch.tensor([4.0, 2.0, 6.0])
    h_l = torch.tensor([1.0, 2.0, 1.0])
    h_r = torch.tensor([3.0, 1.0, 2.0])

    w_l = h_l + 2.0 * h_r
    w_r = 2.0 * h_l + h_r
    expected = (w_l + w_r) / (w_l / s_l + w_r / s_r)

    got = _weighted_harmonic_mean(s_l, s_r, h_l, h_r)
    assert torch.allclose(got, expected)


def test_endpoint_slope_matches_formula() -> None:
    s0 = torch.tensor([2.0, -2.0])
    s1 = torch.tensor([4.0, -1.0])
    h0 = torch.tensor([1.0, 2.0])
    h1 = torch.tensor([3.0, 1.0])

    w1 = 2.0 * h0 + h1
    expected = (w1 * s0 - h0 * s1) / (h0 + h1)

    got = _endpoint_slope(s0, s1, h0, h1)
    assert torch.allclose(got, expected)


def test_limit_endpoint_zeros_when_opposite_sign() -> None:
    d_end = torch.tensor([1.0, -2.0, 3.0])
    s0 = torch.tensor([-1.0, 2.0, 3.0])  # opposite sign for first two, same for third
    s1 = torch.tensor([-1.0, 2.0, 4.0])

    got = _limit_endpoint(d_end, s0, s1)
    expected = torch.tensor([0.0, 0.0, 3.0])
    assert torch.allclose(got, expected)


def test_limit_endpoint_caps_when_secants_switch_sign() -> None:
    # secants switch sign => cap magnitude to 3*s0 when |d_end| > 3*|s0|
    d_end = torch.tensor([10.0, -10.0, 2.0])
    s0 = torch.tensor([2.0, -2.0, 2.0])
    s1 = torch.tensor([-1.0, 1.0, 3.0])  # sign switch for first two, not for third

    got = _limit_endpoint(d_end, s0, s1)
    expected = torch.tensor([6.0, -6.0, 2.0])
    assert torch.allclose(got, expected)


def test_pchip_derivatives_n_equals_2_returns_delta() -> None:
    h = torch.tensor([2.0])
    delta = torch.tensor([3.0])

    d = _pchip_derivatives(h, delta)
    assert torch.allclose(d, torch.tensor([3.0, 3.0]))


def test_pchip_derivatives_interior_zero_when_sign_change_or_zero() -> None:
    # n=4 => two interior derivatives d[1], d[2]
    h = torch.tensor([1.0, 1.0, 1.0])
    delta = torch.tensor([1.0, -2.0, 3.0])  # sign change between 1 and -2, and -2 to 3

    d = _pchip_derivatives(h, delta)
    assert d.shape == (4,)
    assert torch.allclose(d[1:-1], torch.zeros(2))


def test_pchip_derivatives_interior_positive_same_sign_weighted_harmonic() -> None:
    h = torch.tensor([1.0, 2.0, 1.0])
    delta = torch.tensor([1.0, 2.0, 4.0])  # same sign throughout

    d = _pchip_derivatives(h, delta)

    # Interior i=1 uses delta[0],delta[1] with h[0],h[1]
    dh0 = _weighted_harmonic_mean(delta[0], delta[1], h[0], h[1])
    # Interior i=2 uses delta[1],delta[2] with h[1],h[2]
    dh1 = _weighted_harmonic_mean(delta[1], delta[2], h[1], h[2])

    assert torch.allclose(d[1], dh0)
    assert torch.allclose(d[2], dh1)


def test_pchip1d_interpolates_knots_exactly() -> None:
    x = torch.tensor([0.0, 1.0, 2.5, 4.0])
    y = torch.tensor([0.0, 2.0, 1.0, 3.0])

    f = PCHIP1D(x, y)
    yq = f(x)

    assert torch.allclose(yq, y)


def test_pchip1d_monotone_data_stays_in_range_on_each_interval() -> None:
    # For monotone increasing data, PCHIP should not overshoot:
    x = torch.tensor([0.0, 1.0, 2.0, 3.0])
    y = torch.tensor([0.0, 1.0, 1.5, 2.0])

    f = PCHIP1D(x, y)

    # sample points densely; verify within [y_i, y_{i+1}] interval-wise
    xq = torch.linspace(x[0].item(), x[-1].item(), 301)
    yq = f(xq)

    # overall range check (weaker but simple + robust)
    assert torch.all(yq >= y.min() - 1e-6)
    assert torch.all(yq <= y.max() + 1e-6)

    # stronger: interval-wise bounds
    for i in range(x.numel() - 1):
        mask = (xq >= x[i]) & (xq <= x[i + 1])
        lo = min(y[i].item(), y[i + 1].item())
        hi = max(y[i].item(), y[i + 1].item())
        assert torch.all(yq[mask] >= lo - 1e-6)
        assert torch.all(yq[mask] <= hi + 1e-6)


@pytest.mark.parametrize(
    ("x", "y", "err_type", "match"),
    [
        pytest.param(
            torch.tensor([0, 1, 2]),
            torch.tensor([0.0, 1.0, 2.0]),
            TypeError,
            "x must be a floating point tensor",
            id="x-not-float",
        ),
        pytest.param(
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.0, 1.0, 2.0]),
            ValueError,
            "x and y must have the same length",
            id="length-mismatch",
        ),
        pytest.param(
            torch.tensor([0.0, 1.0, 1.0]),
            torch.tensor([0.0, 1.0, 2.0]),
            ValueError,
            "x must be strictly increasing",  # i.e. no division by zero
            id="x-not-strictly-increasing",
        ),
        pytest.param(
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            ValueError,
            "Need at least 2 points",
            id="too-few-points",
        ),
    ],
)
def test_pchip1d_validate_xy_errors(
    x: torch.Tensor,
    y: torch.Tensor,
    err_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(err_type, match=match):
        PCHIP1D(x, y)


def test_pchip_derivatives_handles_zero_slopes_no_nan_inf() -> None:
    # include zeros in delta to trigger the "no harmonic mean" path
    h = torch.tensor([1.0, 1.0, 1.0, 1.0])
    delta = torch.tensor([1.0, 0.0, -2.0, 3.0])  # div 0 if not guarded

    d = _pchip_derivatives(h, delta)

    assert torch.isfinite(d).all()
    # PCHIP sets d[i]=0 unless delta[i-1] and delta[i] have the same nonzero sign
    assert d[1].item() == 0.0  # between 1.0 and 0.0
    assert d[2].item() == 0.0  # between 0.0 and -2.0
    assert d[3].item() == 0.0  # between -2.0 and 3.0


def test_pchip_small_values() -> None:
    # straight line with a slope 0.1
    x = torch.tensor([1.0, 2.0, 3.0, 4.0]) * 1e-8
    y = 0.1 * x

    h = x[1:] - x[:-1]
    delta = (y[1:] - y[:-1]) / h

    d = _pchip_derivatives(h, delta)  # expected derivative d = 0.1

    assert torch.allclose(d, torch.tensor(0.1))
