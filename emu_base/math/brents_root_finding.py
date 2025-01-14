from typing import Callable, Optional


class BrentsRootFinder:
    def __init__(
        self,
        *,
        start: float,
        end: float,
        f_start: float,
        f_end: float,
        epsilon: float = 1e-6,
    ):
        self.epsilon = epsilon

        assert start <= end
        self.a, self.b = start, end
        self.fa = f_start
        self.fb = f_end

        assert self.fa * self.fb < 0, "Function root needs to be between a and b"

        # b has to be the better guess
        if abs(self.fa) < abs(self.fb):
            self.a, self.b = self.b, self.a
            self.fa, self.fb = self.fb, self.fa

        self.c = self.a
        self.d = self.c
        self.fc = self.fa

        self.bisection = True
        self.current_guess = self.b
        self.next_abscissa: Optional[float] = None

    def get_next_abscissa(self) -> float:
        if abs(self.fc - self.fa) < self.epsilon or abs(self.fc - self.fb) < self.epsilon:
            # Secant method
            dx = self.fb * (self.b - self.a) / (self.fa - self.fb)
        else:
            # Inverse quadratic interpolation
            s = self.fb / self.fa
            r = self.fb / self.fc
            t = self.fa / self.fc
            q = (t - 1) * (s - 1) * (r - 1)
            p = s * (t * (r - t) * (self.c - self.b) + (r - 1) * (self.b - self.a))
            dx = p / q

        # Use bisection instead of interpolation
        # if the interpolation is not within bounds.
        delta = abs(2 * self.epsilon * self.b)
        adx = abs(dx)
        delta_bc = abs(self.b - self.c)
        delta_cd = abs(self.c - self.d)
        delta_ab = self.a - self.b
        if (
            (adx >= abs(3 * delta_ab / 4) or dx * delta_ab < 0)
            or (self.bisection and adx >= delta_bc / 2)
            or (not self.bisection and adx >= delta_cd / 2)
            or (self.bisection and delta_bc < delta)
            or (not self.bisection and delta_cd < delta)
        ):
            dx = (self.a - self.b) / 2
            self.bisection = True
        else:
            self.bisection = False

        self.next_abscissa = self.b + dx
        self.d = self.c
        self.c, self.fc = self.b, self.fb

        return self.next_abscissa

    def provide_ordinate(self, abscissa: float, ordinate: float) -> None:
        # First argument is just a safety
        assert (
            self.next_abscissa is not None and abscissa == self.next_abscissa
        ), "Something went wrong"

        # Update interval
        if self.fa * ordinate < 0:
            self.b, self.fb = abscissa, ordinate
        else:
            self.a, self.fa = abscissa, ordinate

        # b has to be the better guess
        if abs(self.fa) < abs(self.fb):
            self.a, self.b = self.b, self.a
            self.fa, self.fb = self.fb, self.fa

        self.current_guess = self.b

    def is_converged(self, tolerance: float) -> bool:
        return abs(self.b - self.a) < tolerance


def find_root_brents(
    f: Callable[[float], float],
    *,
    start: float,
    end: float,
    f_start: Optional[float] = None,
    f_end: Optional[float] = None,
    tolerance: float = 1e-6,
    epsilon: float = 1e-6,
) -> float:
    """
    Approximates and returns the zero of a scalar function using Brent's method.
    """
    f_start = f_start if f_start is not None else f(start)
    f_end = f_end if f_end is not None else f(end)

    root_finder = BrentsRootFinder(
        start=start, end=end, f_start=f_start, f_end=f_end, epsilon=epsilon
    )

    while not root_finder.is_converged(tolerance):
        x = root_finder.get_next_abscissa()
        root_finder.provide_ordinate(x, f(x))

    return root_finder.current_guess
