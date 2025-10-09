import random
from unittest.mock import MagicMock

import torch

from emu_base.math.brents_root_finding import find_root_brents


test_tolerance = 1e-10

"""
Returns the result of `find_root_brents` but also the number of times `f` was called.
"""


def find_root_brents_instrumented(
    f,
    *,
    start,
    end,
    f_start: float | None = None,
    f_end: float | None = None,
    tolerance: float = 1e-6,
    epsilon: float = 1e-6,
) -> (float, int):
    mock_f = MagicMock()
    mock_f.side_effect = f

    result_root = find_root_brents(
        mock_f,
        start=start,
        end=end,
        f_start=f_start,
        f_end=f_end,
        tolerance=tolerance,
        epsilon=epsilon,
    )
    result_call_count = mock_f.call_count

    return result_root, result_call_count


def test_find_root_polynomial():
    def P(x):
        return (x + 3) * (x - 1) ** 2

    actual_root, call_count = find_root_brents_instrumented(P, start=-4, end=4 / 3.0)

    expected_root = -3

    assert abs(actual_root - expected_root) < test_tolerance
    assert call_count == 12


def test_find_root_segments():
    def f(x: float) -> float:
        if x < 0:
            return -1
        return -1 + x * 10

    actual_root, call_count = find_root_brents_instrumented(f, start=-10, end=10)

    expected_root = 0.1

    assert abs(actual_root - expected_root) < test_tolerance
    assert call_count == 13


def test_find_root_integral():
    # Create a random continuous monotonously increasing function
    # by integrating a random >= 0 stairs function.

    random.seed(0xDEADBEEF)
    n = 100  # Number of stair steps
    diff_values = [random.random() for _ in range(n)]  # Steps heights
    slice_size = 1 / n  # Width of stair step

    m = (
        sum(diff_values) * slice_size
    )  # Integral of the whole stair function between 0 and 1
    init = -m / 2

    def f(x: float) -> float:
        # Integral between 0 and x of the stairs.

        assert 0 <= x <= 1

        if x >= 1:
            return m

        index = int(x * n)

        return (
            init
            + sum(diff_values[:index]) * slice_size
            + x % slice_size * diff_values[index]
        )

    # Compute the expected root.
    acc = init
    i = 0
    while acc + slice_size * diff_values[i] <= 0:
        acc += slice_size * diff_values[i]
        i += 1

    expected_root = i * slice_size - acc / diff_values[i]

    actual_root, call_count = find_root_brents_instrumented(f, start=0, end=1)

    assert abs(actual_root - expected_root) < test_tolerance
    assert call_count == 18


def test_find_root_store_intermediate_result():
    # For noisy mcwf time steps, we need the scalar function
    # to store intermediate results: the evolved state.
    # This test gives and example of this and makes sure that the last function evaluation
    # corresponds to the found root.
    n = 10

    torch.manual_seed(0xDEADBEEF)
    lindblad = torch.rand(n, n, dtype=torch.complex128)
    rate = 0.1
    hami = torch.rand(n, n, dtype=torch.complex128)
    hami += hami.T.conj()
    noisy_hami = hami - 1j / 2.0 * rate * lindblad.T.conj() @ lindblad
    psi = torch.rand(n, dtype=torch.complex128)
    psi /= psi.norm()

    target_norm = 0.5

    psi_evolved = None

    def f(t: float) -> float:
        nonlocal psi_evolved
        psi_evolved = torch.matmul(torch.linalg.matrix_exp(-1j * t * noisy_hami), psi)
        assert (
            psi_evolved.norm() <= psi.norm()
        ), "Norm of the state should decrease with time"

        return psi_evolved.norm() - target_norm

    found_root, call_count = find_root_brents_instrumented(f, start=0, end=20)

    assert abs(f(found_root)) < test_tolerance
    assert (psi_evolved.norm() - target_norm) < test_tolerance

    assert call_count == 12
