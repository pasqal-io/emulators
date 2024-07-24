from typing import Callable


def find_root_brents(
    f: Callable[[float], float],
    a: float,
    b: float,
    fa: float | None = None,
    fb: float | None = None,
    /,
    *,
    tolerance: float = 1e-6,
    epsilon: float = 1e-6,
) -> float:
    """
    Approximates and returns the zero of a scalar function using Brent's method.
    """
    fa = fa or f(a)
    fb = fb or f(b)

    assert fa * fb < 0, "Function root needs to be between a and b"

    # b has to be the better guess
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    d = c
    fc = fa

    bisection = True
    while abs(b - a) > tolerance:
        if abs(fc - fa) < epsilon or abs(fc - fb) < epsilon:
            # Secant method
            dx = fb * (b - a) / (fa - fb)
        else:
            # Inverse quadratic interpolation
            s = fb / fa
            r = fb / fc
            t = fa / fc
            q = (t - 1) * (s - 1) * (r - 1)
            p = s * (t * (r - t) * (c - b) + (r - 1) * (b - a))
            dx = p / q

        # Use bisection instead of interpolation
        # if the interpolation is not within bounds.
        delta = abs(2 * epsilon * b)
        adx = abs(dx)
        delta_bc = abs(b - c)
        delta_cd = abs(c - d)
        delta_ab = a - b
        if (
            (adx >= abs(3 * delta_ab / 4) or dx * delta_ab < 0)
            or (bisection and adx >= delta_bc / 2)
            or (not bisection and adx >= delta_cd / 2)
            or (bisection and delta_bc < delta)
            or (not bisection and delta_cd < delta)
        ):
            dx = (a - b) / 2
            bisection = True
        else:
            bisection = False

        x = b + dx
        fx = f(x)
        d = c
        c, fc = b, fb

        # Update interval
        if fa * fx < 0:
            b, fb = x, fx
            fb = fx
        else:
            a, fa = x, fx
            fa = fx

        # b has to be the better guess
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

    return b
