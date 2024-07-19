from emu_ct import MPS, inner
import torch
import pytest
import math
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

    # Check that no copy or move is performed when keep_devices=True
    state_keep_devices = MPS([factor1, factor2, factor3], keep_devices=True)
    assert state_keep_devices.factors[0] is factor1
    assert state_keep_devices.factors[1] is factor2
    assert state_keep_devices.factors[2] is factor3


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


dtype = torch.complex128
tol = 1e-12
down = torch.tensor([[[1], [0]]], dtype=dtype)
up = torch.tensor([[[0], [1]]], dtype=dtype)


def test_wrong_external_links():
    factor1 = torch.rand(3, 2, 5)
    factor2 = torch.rand(5, 2, 5)
    good_left_factor = torch.rand(1, 2, 3)
    wrong_left_factor = torch.rand(3, 2, 3)
    good_right_factor = torch.rand(5, 2, 1)
    wrong_right_factor = torch.rand(5, 2, 5)

    MPS([good_left_factor, factor1, factor2, good_right_factor])

    with pytest.raises(ValueError) as ve:
        MPS([wrong_left_factor, factor1, factor2, good_right_factor])
    msg = "The dimension of the left (right) link of the first (last) tensor should be 1"
    assert str(ve.value) == msg

    with pytest.raises(ValueError) as ve:
        MPS([good_left_factor, factor1, factor2, wrong_right_factor])
    msg = "The dimension of the left (right) link of the first (last) tensor should be 1"
    assert str(ve.value) == msg


def get_bit(i: int, bit_index: int) -> int:
    return int(bool(i & (1 << bit_index)))


def get_bistring_coeff(mps: MPS, bitstring: int) -> float | complex:
    """
    Returns the value of the MPS at the physical indexes specified by the bistring.
    Useful way to check single components of the MPS that does not depend on inner.
    """
    N = mps.num_sites
    assert bitstring < 2**N
    bit = get_bit(bitstring, N - 1)
    acc = mps.factors[0][0, bit, :]
    for i in range(1, N):
        bit = get_bit(bitstring, N - 1 - i)
        acc @= mps.factors[i][:, bit, :]
    return acc.item()


def test_add_to_make_ghz_state():
    num_sites = 5  # number of sites
    mps_down = MPS([down for _ in range(num_sites)])
    mps_up = MPS([up for _ in range(num_sites)])

    # make a |000〉+ |111〉state
    mps_sum = mps_down + mps_up

    # test with inner
    assert inner(mps_sum, mps_down) == pytest.approx(1.0, tol)
    assert inner(mps_sum, mps_up) == pytest.approx(1.0, tol)
    assert inner(mps_sum, mps_sum) == pytest.approx(2.0, tol)

    # test with bitstring
    assert get_bistring_coeff(mps_sum, 0) == pytest.approx(1.0, tol)
    assert get_bistring_coeff(mps_sum, 2**num_sites - 1) == pytest.approx(1.0, tol)

    # sum state is orthogonal
    norm = torch.norm(mps_sum.factors[0]).item()
    assert norm == pytest.approx(math.sqrt(2), tol)


def test_add_to_make_w_state():
    num_sites = 7
    states = []
    for i in range(num_sites):
        factors = [up if (j == i) else down for j in range(num_sites)]
        states.append(MPS(factors))

    # make a |100...〉+ |010...〉+ ... + |...001〉 state
    mps_sum = sum(states[1:], start=states[0])

    # test with inner
    for state in states:
        assert inner(mps_sum, state) == pytest.approx(1.0, tol)
    assert inner(mps_sum, mps_sum) == pytest.approx(num_sites, tol)

    # test with bitstring
    for i in range(num_sites):
        bitstring = 1 << (num_sites - 1 - i)
        assert get_bistring_coeff(mps_sum, bitstring) == pytest.approx(1.0, tol)
    norm = torch.norm(mps_sum.factors[0]).item()
    assert norm == pytest.approx(math.sqrt(num_sites), tol)


def test_rmul():
    num_sites = 5
    # this test should work for all states
    mps = MPS([up for _ in range(num_sites)])
    for scale in [3.0, 2j, -1 / 4]:
        scaled_mps = scale * mps
        assert inner(mps, scaled_mps) == pytest.approx(scale, tol)
        assert inner(scaled_mps, mps) == pytest.approx(scale.conjugate(), tol)
        assert inner(scaled_mps, scaled_mps) == pytest.approx(abs(scale) ** 2, tol)


def test_catch_err_when_lmul():
    num_sites = 3
    mps = MPS([down for _ in range(num_sites)])
    with pytest.raises(TypeError) as ve:
        mps * 3.0
    msg = "unsupported operand type(s) for *: 'MPS' and 'float'"
    assert str(ve.value) == msg


def test_mps_algebra():
    num_sites = 5
    # this test should work for all states
    mps = MPS([up for _ in range(num_sites)])
    mps_sum = mps + mps + 0.5 * mps + (1 / 3) * mps
    mps_rmul = (1 + 1 + 0.5 + 1 / 3) * mps

    for a, b in zip(mps_sum.factors, mps_rmul.factors):
        assert torch.equal(a, b)
