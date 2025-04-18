import io
import math
from unittest.mock import patch

import pytest
import torch

from emu_mps import MPS, inner, MPO

from test.utils_testing import ghz_state_factors


def check_orthogonality_center(state: MPS, expected_ortho_center: int):
    assert state.orthogonality_center == expected_ortho_center

    for qubit_index in range(expected_ortho_center):
        f = state.factors[qubit_index]
        contracted = torch.tensordot(f.conj(), f, ([0, 1], [0, 1]))

        assert torch.allclose(
            contracted,
            torch.eye(f.shape[2], dtype=torch.complex128, device=contracted.device),
        )

    for qubit_index in range(expected_ortho_center + 1, state.num_sites):
        f = state.factors[qubit_index]
        contracted = torch.tensordot(f.conj(), f, ([1, 2], [1, 2]))

        assert torch.allclose(
            contracted,
            torch.eye(f.shape[0], dtype=torch.complex128, device=contracted.device),
        )


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
    state = MPS(
        [factor1, factor2, factor3],
        eigenstates=("0", "1"),
    )
    state.truncate()
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

    # Check that no copy or move is performed when num_gpus_to_use=None
    no_device_reassignment = MPS(
        [factor1, factor2, factor3],
        num_gpus_to_use=None,
        eigenstates=("0", "1"),
    )
    assert no_device_reassignment.factors[0] is factor1
    assert no_device_reassignment.factors[1] is factor2
    assert no_device_reassignment.factors[2] is factor3


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

    ones = MPS(
        [r_factor] * n_qubits,
        eigenstates=("0", "1"),
    )  # 111
    bell = MPS(
        [l_factor1, l_factor2, l_factor3],
        eigenstates=("0", "1"),
    )  # 000 + i111
    assert abs(inner(bell, ones) + 1j) < 1e-10
    assert abs(inner(ones, bell) - 1j) < 1e-10
    assert abs(inner(ones, ones) - 1) < 1e-10
    assert abs(inner(bell, bell) - 2) < 1e-10


def test_maxbondim():
    bell_state = MPS(
        ghz_state_factors(3),
        eigenstates=("0", "1"),
    )
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

    factors = [good_left_factor, factor1, factor2, good_right_factor]

    MPS(
        factors,
        eigenstates=("0", "1"),
    )

    with pytest.raises(AssertionError) as ve:
        factors[0] = wrong_left_factor
        MPS(
            factors,
            eigenstates=("0", "1"),
        )
    msg = "The dimension of the left (right) link of the first (last) tensor should be 1"
    assert str(ve.value) == msg

    with pytest.raises(AssertionError) as ve:
        factors[0] = good_left_factor
        factors[-1] = wrong_right_factor
        MPS(
            factors,
            eigenstates=("0", "1"),
        )
    msg = "The dimension of the left (right) link of the first (last) tensor should be 1"
    assert str(ve.value) == msg


def get_bit(i: int, bit_index: int) -> int:
    return int(bool(i & (1 << bit_index)))


def get_bitstring_coeff(mps: MPS, bitstring: int) -> float | complex:
    """
    Returns the value of the MPS at the physical indexes specified by the bitstring.
    Useful way to check single components of the MPS that does not depend on inner.
    """
    N = mps.num_sites
    assert bitstring < 2**N
    bit = get_bit(bitstring, N - 1)
    acc = mps.factors[0][0, bit, :]
    for i in range(1, N):
        bit = get_bit(bitstring, N - 1 - i)
        acc @= mps.factors[i][:, bit, :].to(mps.factors[0].device)
    return acc.item()


def test_add_to_make_ghz_state():
    num_sites = 5  # number of sites
    mps_down = MPS(
        [down for _ in range(num_sites)],
        eigenstates=("0", "1"),
    )
    mps_up = MPS(
        [up for _ in range(num_sites)],
        eigenstates=("0", "1"),
    )

    # make a |000〉+ |111〉state
    mps_sum = mps_down + mps_up

    # test with inner
    assert inner(mps_sum, mps_down) == pytest.approx(1.0, tol)
    assert inner(mps_sum, mps_up) == pytest.approx(1.0, tol)
    assert inner(mps_sum, mps_sum) == pytest.approx(2.0, tol)

    # test with bitstring
    assert get_bitstring_coeff(mps_sum, 0) == pytest.approx(1.0, tol)
    assert get_bitstring_coeff(mps_sum, 2**num_sites - 1) == pytest.approx(1.0, tol)

    # sum state is orthogonal
    norm = torch.linalg.norm(mps_sum.factors[0]).item()
    assert norm == pytest.approx(math.sqrt(2), tol)


def test_add_to_make_w_state():
    num_sites = 7
    states = []
    for i in range(num_sites):
        factors = [up if (j == i) else down for j in range(num_sites)]
        states.append(
            MPS(
                factors,
                eigenstates=("0", "1"),
            )
        )

    # make a |100...〉+ |010...〉+ ... + |...001〉 state
    mps_sum = sum(states[1:], start=states[0])

    # test with inner
    for state in states:
        assert inner(mps_sum, state) == pytest.approx(1.0, tol)
    assert inner(mps_sum, mps_sum) == pytest.approx(num_sites, tol)

    # test with bitstring
    for i in range(num_sites):
        bitstring = 1 << (num_sites - 1 - i)
        assert get_bitstring_coeff(mps_sum, bitstring) == pytest.approx(1.0, tol)
    norm = torch.linalg.norm(mps_sum.factors[0]).item()
    assert norm == pytest.approx(math.sqrt(num_sites), tol)


def test_rmul():
    num_sites = 5
    # this test should work for all states
    mps = MPS(
        [up for _ in range(num_sites)],
        eigenstates=("0", "1"),
    )
    for scale in [3.0, 2j, -1 / 4]:
        scaled_mps = scale * mps
        assert inner(mps, scaled_mps) == pytest.approx(scale, tol)
        assert inner(scaled_mps, mps) == pytest.approx(scale.conjugate(), tol)
        assert inner(scaled_mps, scaled_mps) == pytest.approx(abs(scale) ** 2, tol)

        # Scaling doesn't orthogonalize.
        assert scaled_mps.orthogonality_center is mps.orthogonality_center is None


def test_catch_err_when_lmul():
    num_sites = 3
    mps = MPS(
        [down for _ in range(num_sites)],
        eigenstates=("0", "1"),
    )
    with pytest.raises(TypeError) as ve:
        mps * 3.0
    msg = "unsupported operand type(s) for *: 'MPS' and 'float'"
    assert str(ve.value) == msg


def test_mps_algebra():
    num_sites = 5
    # this test should work for all states
    mps = MPS(
        [up for _ in range(num_sites)],
        orthogonality_center=0,
        num_gpus_to_use=0,
        eigenstates=("0", "1"),
    )
    mps_sum = mps + mps + 0.5 * mps + (1 / 3) * mps
    mps_rmul = (1 + 1 + 0.5 + 1 / 3) * mps
    mps_rmul.orthogonalize(0)

    for a, b in zip(mps_sum.factors, mps_rmul.factors):
        assert torch.equal(a, b)


def test_from_string_bell_state():
    afm_string_state = {"rrr": 1.0 / math.sqrt(2), "ggg": 1.0 / math.sqrt(2)}
    afm_mps_state = MPS.from_state_amplitudes(
        eigenstates=("r", "g"), amplitudes=afm_string_state
    )

    assert get_bitstring_coeff(afm_mps_state, 0b000) == pytest.approx(1.0 / math.sqrt(2))
    assert get_bitstring_coeff(afm_mps_state, 0b111) == pytest.approx(1.0 / math.sqrt(2))


@pytest.mark.parametrize(
    ("zero", "one"),
    (("g", "r"), ("0", "1")),
)
@patch("sys.stdout", new_callable=io.StringIO)
def test_from_string_not_normalized_state(mock_print, zero, one):
    afm_not_normalized = {one * 3: 1 / math.sqrt(2), zero * 3: 0.1 / math.sqrt(2)}

    afm_mps_state_normalized = MPS.from_state_amplitudes(
        eigenstates=(one, zero), amplitudes=afm_not_normalized
    )

    assert "The state is not normalized, normalizing it for you" in mock_print.getvalue()

    assert torch.allclose(
        afm_mps_state_normalized.factors[0],
        torch.tensor(
            [[0.0995, 0.0], [0.0, 0.9950]],
            dtype=torch.complex128,
            device=afm_mps_state_normalized.factors[0].device,
        ),
        rtol=0,
        atol=1e-4,
    )
    assert torch.allclose(
        afm_mps_state_normalized.factors[1],
        torch.tensor(
            [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]],
            dtype=torch.complex128,
            device=afm_mps_state_normalized.factors[1].device,
        ),
    )
    assert torch.allclose(
        afm_mps_state_normalized.factors[2],
        torch.tensor(
            [[[1.0], [0.0]], [[0.0], [1.0]]],
            dtype=torch.complex128,
            device=afm_mps_state_normalized.factors[2].device,
        ),
    )


def test_wrong_basis_string_state():
    afm_string_state = {"rrr": 1.0 / math.sqrt(2), "ggg": 1.0 / math.sqrt(2)}

    with pytest.raises(ValueError) as ve:
        MPS.from_state_amplitudes(eigenstates=("0", "r"), amplitudes=afm_string_state)
    msg = "Unsupported basis provided"
    assert str(ve.value) == msg


def test_orthogonalize():
    f1 = torch.rand(1, 2, 2, dtype=torch.complex128)
    f2 = torch.rand(2, 2, 3, dtype=torch.complex128)
    f3 = torch.rand(3, 2, 1, dtype=torch.complex128)

    state = MPS(
        [f1, f2, f3],
        eigenstates=("0", "1"),
    )

    state.orthogonalize(1)
    check_orthogonality_center(state, 1)

    state.orthogonalize(2)
    check_orthogonality_center(state, 2)

    state.orthogonalize(0)
    check_orthogonality_center(state, 0)


def test_norm():
    f1 = torch.rand(1, 2, 2, dtype=torch.complex128)
    f2 = torch.rand(2, 2, 3, dtype=torch.complex128)
    f3 = torch.rand(3, 2, 1, dtype=torch.complex128)

    state = MPS(
        [f1, f2, f3],
        eigenstates=("0", "1"),
    )

    assert state.norm() == pytest.approx(math.sqrt(inner(state, state).real))

    check_orthogonality_center(state, 0)


def test_expect_batch():
    f1 = torch.rand(1, 2, 2, dtype=torch.complex128)
    f2 = torch.rand(2, 2, 3, dtype=torch.complex128)
    f3 = torch.rand(3, 2, 1, dtype=torch.complex128)

    state = MPS(
        [f1, f2, f3],
        eigenstates=("0", "1"),
    )

    op0 = torch.rand(2, 2, dtype=torch.complex128)
    op1 = torch.rand(2, 2, dtype=torch.complex128)

    ops = torch.stack([op0, op1])

    actual = state.expect_batch(ops)

    expected_00 = torch.einsum(
        "abc,cde,efg,bh,ihj,jdk,kfl",
        f1.conj(),
        f2.conj(),
        f3.conj(),
        op0,
        f1,
        f2,
        f3,
    ).item()
    assert expected_00 == pytest.approx(actual[0][0].item())

    expected_11 = torch.einsum(
        "abc,cde,efg,dh,ibj,jhk,kfl",
        f1.conj(),
        f2.conj(),
        f3.conj(),
        op1,
        f1,
        f2,
        f3,
    )
    assert expected_11.item() == pytest.approx(actual[1][1].item())


def test_apply():
    state = MPS.make(4)
    hadamard = (
        1.0 / math.sqrt(2) * torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128)
    )
    state.apply(1, hadamard)
    state.apply(2, hadamard)

    assert get_bitstring_coeff(state, 0b0000) == pytest.approx(0.5)
    assert get_bitstring_coeff(state, 0b0100) == pytest.approx(0.5)
    assert get_bitstring_coeff(state, 0b0010) == pytest.approx(0.5)
    assert get_bitstring_coeff(state, 0b0110) == pytest.approx(0.5)

    check_orthogonality_center(state, 2)


def test_apply_random_orthogonality_center():
    state = MPS.make(4)
    r1 = torch.rand(2, 2, dtype=torch.complex128)
    r2 = torch.rand(2, 2, dtype=torch.complex128)
    state.apply(1, r1)
    state.apply(2, r2)

    check_orthogonality_center(state, 2)


def test_correlation_matrix_random():
    qubit_count = 5
    state = MPS(
        [
            torch.rand(1, 2, 3, dtype=torch.complex128),
            torch.rand(3, 2, 5, dtype=torch.complex128),
            torch.rand(5, 2, 12, dtype=torch.complex128),
            torch.rand(12, 2, 2, dtype=torch.complex128),
            torch.rand(2, 2, 1, dtype=torch.complex128),
        ],
        eigenstates=("r", "g"),
    )

    correlation_matrix_nn = state.get_correlation_matrix()

    correlation_matrix_zz = state.get_correlation_matrix(
        operator=torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
    )

    def nn(index1, index2):
        return MPO.from_operator_repr(
            eigenstates=("r", "g"),
            n_qudits=qubit_count,
            operations=[(1.0, [({"rr": 1.0}, list({index1, index2}))])],
        )

    def zz(index1, index2):
        return MPO.from_operator_repr(
            eigenstates=("r", "g"),
            n_qudits=qubit_count,
            operations=[(1.0, [({"gg": 1.0, "rr": -1.0}, list({index1, index2}))])],
        )

    assert len(correlation_matrix_nn) == len(correlation_matrix_zz) == qubit_count
    for i in range(qubit_count):
        assert (
            len(correlation_matrix_nn[i]) == len(correlation_matrix_zz[i]) == qubit_count
        )
        for j in range(qubit_count):
            assert correlation_matrix_nn[i][j].item() == pytest.approx(
                nn(i, j).expect(state)
            )
            assert correlation_matrix_zz[i][j].item() == pytest.approx(
                zz(i, j).expect(state)
            )


@pytest.mark.parametrize(
    "eig, ampl",
    [
        (("0", "1"), {4 * "1": 1.0}),
        (("0", "1"), {4 * "0": 1.0}),
        (("r", "g"), {4 * "r": 1.0}),
        (("r", "g"), {4 * "g": 1.0}),
    ],
)
def test_to_abstr_repr(eig, ampl) -> None:
    initial_state = MPS.from_state_amplitudes(eigenstates=eig, amplitudes=ampl)
    abstr = initial_state._to_abstract_repr()

    assert ampl == abstr["amplitudes"]
    assert eig == abstr["eigenstates"]


@pytest.mark.parametrize(
    "eig, ampl",
    [
        (("0", "1"), {4 * "r": 1.0}),
        (("r", "g"), {4 * "1": 1.0}),
    ],
)
def test_constructor(eig, ampl) -> None:
    with pytest.raises(ValueError):
        MPS.from_state_amplitudes(eigenstates=eig, amplitudes=ampl)
