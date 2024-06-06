from emu_ct import MPS, inner
import torch
from .utils_testing import ghz_state_factors


def test_init():
    factor1 = torch.tensor([[[0, 1, 0, 0], [0, 0, 0, 0]]], dtype=torch.complex128)
    factor2 = torch.tensor(
        [
            [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 1, 0], [0, 0, 0, 0, 0]],
            [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
        ],
        dtype=torch.complex128,
    )
    factor3 = torch.tensor(
        [[[0], [0]], [[0], [0]], [[0], [0]], [[1], [0]], [[0], [0]]],
        dtype=torch.complex128,
    )
    state = MPS([factor1, factor2, factor3], truncate=True)
    for factor in state.factors:
        assert factor.shape == (1, 2, 1)
        # this determines the factors up to a global phase, which is implementation dependent
        # due to svd returning different phases on cpu and gpu
        assert (
            abs(
                torch.tensordot(
                    factor,
                    torch.tensor(
                        [[[1], [0]]], dtype=torch.complex128, device=factor.device
                    ),
                    dims=3,
                )
            )
            - 1
            < 1e-8
        )
        assert abs(torch.tensordot(factor, factor, dims=3)) - 1 < 1e-8


def test_inner():
    n_qubits = 3
    l_factor1 = torch.tensor([[[1, 0], [0, 1j]]], dtype=torch.complex128)
    l_factor2 = torch.tensor(
        [[[1, 0], [0, 0]], [[0, 0], [0, 1]]],
        dtype=torch.complex128,
    )
    l_factor3 = torch.tensor(
        [[[1], [0]], [[0], [1]]],
        dtype=torch.complex128,
    )
    r_factor = torch.tensor(
        [[[0], [1]]],
        dtype=torch.complex128,
    )

    ones = MPS([r_factor] * n_qubits)  # 111
    bell = MPS([l_factor1, l_factor2, l_factor3])  # 000 + i111
    assert abs(inner(bell, ones) + 1j) < 1e-10
    assert abs(inner(ones, bell) - 1j) < 1e-10
    assert abs(inner(ones, ones) - 1) < 1e-10
    assert abs(inner(bell, bell) - 2) < 1e-10


def test_maxbondim():
    bell_state = MPS(ghz_state_factors(3))

    assert 2 == bell_state.get_max_bond_dim()
