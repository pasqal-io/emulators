from emu_ct import MPS
import cupy as cp
import numpy as np


def test_init():
    factor1 = cp.array([[[0, 1, 0, 0], [0, 0, 0, 0]]], dtype=np.complex128)
    factor2 = cp.array(
        [
            [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 1, 0], [0, 0, 0, 0, 0]],
            [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
        ],
        dtype=np.complex128,
    )
    factor3 = cp.array(
        [[[0], [0]], [[0], [0]], [[0], [0]], [[1], [0]], [[0], [0]]], dtype=np.complex128
    )
    state = MPS([factor1, factor2, factor3])
    for factor in state.factors:
        assert np.allclose(factor.get(), np.array([[[1], [0]]], dtype=np.complex128))
