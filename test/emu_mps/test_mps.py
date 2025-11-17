import math

import pytest
import torch

from emu_mps import MPS, inner, MPO, MPSConfig

from test.utils_testing import ghz_state_factors

dtype = torch.complex128

tol = 1e-12
down_2level = torch.tensor([[[1.0], [0.0]]], dtype=dtype)
up_2level = torch.tensor([[[0.0], [1.0]]], dtype=dtype)

down_3level = torch.tensor([[[1.0], [0.0], [0.0]]], dtype=dtype)
up_3level = torch.tensor([[[0.0], [1.0], [0.0]]], dtype=dtype)


def check_orthogonality_center(state: MPS, expected_ortho_center: int):
    assert state.orthogonality_center == expected_ortho_center

    for qubit_index in range(expected_ortho_center):
        f = state.factors[qubit_index]
        contracted = torch.tensordot(f.conj(), f, ([0, 1], [0, 1]))

        assert torch.allclose(
            contracted,
            torch.eye(f.shape[2], dtype=dtype, device=contracted.device),
        )

    for qubit_index in range(expected_ortho_center + 1, state.num_sites):
        f = state.factors[qubit_index]
        contracted = torch.tensordot(f.conj(), f, ([1, 2], [1, 2]))

        assert torch.allclose(
            contracted,
            torch.eye(f.shape[0], dtype=dtype, device=contracted.device),
        )


def test_init():
    factor1 = torch.tensor([[[0, 1, 0, 0], [0, 0, 0, 0]]], dtype=dtype)
    factor2 = torch.tensor(
        [
            [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 1, 0], [0, 0, 0, 0, 0]],
            [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
        ],
        dtype=dtype,
    )
    factor3 = torch.tensor(
        [[[0], [0]], [[0], [0]], [[0], [0]], [[1], [0]], [[0], [0]]],
        dtype=dtype,
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
                    torch.tensor([[[1], [0]]], dtype=dtype, device=factor.device),
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
    l_factor1 = torch.tensor([[[1, 0], [0, 1j]]], dtype=dtype)
    l_factor2 = torch.tensor(
        [[[1, 0], [0, 0]], [[0, 0], [0, 1]]],
        dtype=dtype,
    )
    l_factor3 = torch.tensor(
        [[[1], [0]], [[0], [1]]],
        dtype=dtype,
    )
    r_factor = torch.tensor(
        [[[0], [1]]],
        dtype=dtype,
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


@pytest.mark.parametrize("basis", (("0", "1"), ("g", "r", "x")))
def test_maxbondim(basis):
    bell_state = MPS(
        ghz_state_factors(3, dim=len(basis)),
        eigenstates=basis,
    )
    assert 2 == bell_state.get_max_bond_dim()


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
    """Return 1 if the bit at bit_index in i is set, else 0.
    bit_index 0 is the least significant bit."""
    if not isinstance(i, int) or not isinstance(bit_index, int):
        raise TypeError("i and bit_index must be integers")
    if bit_index < 0:
        raise ValueError("bit_index must be >= 0")
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


@pytest.mark.parametrize(
    "basis",
    (
        ("0", "1"),
        ("r", "g", "x"),
    ),
)
def test_add_to_make_ghz_state(basis):
    num_sites = 5  # number of sites

    down = down_3level if len(basis) == 3 else down_2level
    up = up_3level if len(basis) == 3 else up_2level
    mps_down = MPS(
        [down for _ in range(num_sites)],
        eigenstates=basis,
    )
    mps_up = MPS(
        [up for _ in range(num_sites)],
        eigenstates=basis,
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


@pytest.mark.parametrize("basis", (("0", "1"), ("r", "g", "x")))
def test_add_to_make_w_state(basis):
    num_sites = 7
    states = []
    down = down_3level if len(basis) == 3 else down_2level
    up = up_3level if len(basis) == 3 else up_2level
    for i in range(num_sites):
        factors = [up if (j == i) else down for j in range(num_sites)]
        states.append(
            MPS(
                factors,
                eigenstates=basis,
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


@pytest.mark.parametrize("basis", (("0", "1"), ("r", "g", "x")))
def test_rmul(basis):
    num_sites = 5
    # this test should work for all states
    up = up_3level if len(basis) == 3 else up_2level
    mps = MPS(
        [up for _ in range(num_sites)],
        eigenstates=basis,
    )
    for scale in [3.0, 2j, -1 / 4]:
        scaled_mps = scale * mps
        assert inner(mps, scaled_mps) == pytest.approx(scale, tol)
        assert inner(scaled_mps, mps) == pytest.approx(scale.conjugate(), tol)
        assert inner(scaled_mps, scaled_mps) == pytest.approx(abs(scale) ** 2, tol)

        # Scaling doesn't orthogonalize.
        assert scaled_mps.orthogonality_center is mps.orthogonality_center is None

        num_sites = 5


@pytest.mark.parametrize(
    "basis",
    (
        ("0", "1"),
        ("r", "g", "x"),
    ),
)
def test_catch_err_when_lmul(basis):
    num_sites = 3
    down = down_3level if len(basis) == 3 else down_2level
    mps = MPS(
        [down for _ in range(num_sites)],
        eigenstates=basis,
    )
    with pytest.raises(TypeError) as ve:
        mps * 3.0
    msg = "unsupported operand type(s) for *: 'MPS' and 'float'"
    assert str(ve.value) == msg


@pytest.mark.parametrize(
    "basis",
    (
        ("0", "1"),
        ("r", "g", "x"),
    ),
)
def test_mps_algebra(basis):
    num_sites = 5
    # this test should work for all states
    up = up_3level if len(basis) == 3 else up_2level
    mps = MPS(
        [up for _ in range(num_sites)],
        orthogonality_center=0,
        num_gpus_to_use=0,
        eigenstates=basis,
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


def test_from_string_afm_state_3level_system():
    afm_string_state = {"rrr": 1.0 / math.sqrt(2), "ggg": 1.0 / math.sqrt(2)}
    afm_mps_state = MPS.from_state_amplitudes(
        eigenstates=("r", "g", "x"), amplitudes=afm_string_state
    )

    assert get_bitstring_coeff(afm_mps_state, 0b000) == pytest.approx(1.0 / math.sqrt(2))
    assert get_bitstring_coeff(afm_mps_state, 0b111) == pytest.approx(1.0 / math.sqrt(2))


def test_from_string_afm_state_3level_system_leak():
    afm_string_state = {
        "rr": 1.0 / math.sqrt(3),
        "gg": 1.0 / math.sqrt(3),
        "xx": 1.0 / math.sqrt(3),
    }
    afm_mps_state = MPS.from_state_amplitudes(
        eigenstates=("r", "g", "x"), amplitudes=afm_string_state
    )

    assert get_bitstring_coeff(afm_mps_state, 0b11) == pytest.approx(1.0 / math.sqrt(3))
    assert get_bitstring_coeff(afm_mps_state, 0b00) == pytest.approx(1.0 / math.sqrt(3))
    # assert get_bitstring_coeff(afm_mps_state, 0b22) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("zero", "one"),
    (("g", "r"), ("0", "1")),
)
def test_from_string_not_normalized_state(capfd, zero, one):
    MPSConfig()
    afm_not_normalized = {one * 3: 1 / math.sqrt(2), zero * 3: 0.1 / math.sqrt(2)}

    afm_mps_state_normalized = MPS.from_state_amplitudes(
        eigenstates=(one, zero), amplitudes=afm_not_normalized
    )

    out, _ = capfd.readouterr()
    assert "The state is not normalized, normalizing it for you" in out

    assert torch.allclose(
        afm_mps_state_normalized.factors[0],
        torch.tensor(
            [[0.0995, 0.0], [0.0, 0.9950]],
            dtype=dtype,
            device=afm_mps_state_normalized.factors[0].device,
        ),
        rtol=0,
        atol=1e-4,
    )
    assert torch.allclose(
        afm_mps_state_normalized.factors[1],
        torch.tensor(
            [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]],
            dtype=dtype,
            device=afm_mps_state_normalized.factors[1].device,
        ),
    )
    assert torch.allclose(
        afm_mps_state_normalized.factors[2],
        torch.tensor(
            [[[1.0], [0.0]], [[0.0], [1.0]]],
            dtype=dtype,
            device=afm_mps_state_normalized.factors[2].device,
        ),
    )


def test_wrong_basis_string_state():
    N = 3
    afm_string_state = {"r" * N: 1.0 / math.sqrt(2), "g" * N: 1.0 / math.sqrt(2)}

    with pytest.raises(ValueError) as ve:
        MPS.from_state_amplitudes(eigenstates=("0", "r"), amplitudes=afm_string_state)
    msg = (
        "All basis states must be combinations of eigenstates with the same length. "
        f"Expected combinations of ('0', 'r'), each with {N} elements."
    )
    assert str(ve.value) == msg


@pytest.mark.parametrize("basis", (("0", "1"), ("r", "g", "x")))
def test_orthogonalize(basis):
    dim = len(basis)

    f1 = torch.rand(1, dim, 2, dtype=dtype)
    f2 = torch.rand(2, dim, 3, dtype=dtype)
    f3 = torch.rand(3, dim, 1, dtype=dtype)
    state = MPS(
        [f1, f2, f3],
        eigenstates=basis,
    )

    state.orthogonalize(1)
    check_orthogonality_center(state, 1)

    state.orthogonalize(2)
    check_orthogonality_center(state, 2)

    state.orthogonalize(0)
    check_orthogonality_center(state, 0)


@pytest.mark.parametrize("basis", (("0", "1"), ("r", "g", "x")))
def test_norm(basis):
    dim = len(basis)

    f1 = torch.rand(1, dim, 2, dtype=dtype)
    f2 = torch.rand(2, dim, 3, dtype=dtype)
    f3 = torch.rand(3, dim, 1, dtype=dtype)
    state = MPS(
        [f1, f2, f3],
        eigenstates=basis,
    )

    assert state.norm() == pytest.approx(math.sqrt(inner(state, state).real))

    check_orthogonality_center(state, 0)


@pytest.mark.parametrize("basis", (("0", "1"), ("r", "g", "x")))
def test_expect_batch(basis):
    dim = len(basis)
    f1 = torch.rand(1, dim, 2, dtype=dtype)
    f2 = torch.rand(2, dim, 3, dtype=dtype)
    f3 = torch.rand(3, dim, 1, dtype=dtype)
    state = MPS(
        [f1, f2, f3],
        eigenstates=basis,
    )
    op0 = torch.rand(dim, dim, dtype=dtype)
    op1 = torch.rand(dim, dim, dtype=dtype)

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


@pytest.mark.parametrize("basis", (("0", "1"), ("r", "g", "x")))
def test_apply(basis):

    state = MPS.make(4, eigenstates=basis)

    if len(basis) == 2:
        hadamard = 1.0 / math.sqrt(2) * torch.tensor([[1, 1], [1, -1]], dtype=dtype)
    if len(basis) == 3:
        hadamard = (
            1.0
            / math.sqrt(2)
            * torch.tensor([[1, 1, 0], [1, -1, 0], [0, 0, 0]], dtype=dtype)
        )
    state.apply(1, hadamard)
    state.apply(2, hadamard)

    assert get_bitstring_coeff(state, 0b0000) == pytest.approx(0.5)
    assert get_bitstring_coeff(state, 0b0100) == pytest.approx(0.5)
    assert get_bitstring_coeff(state, 0b0010) == pytest.approx(0.5)
    assert get_bitstring_coeff(state, 0b0110) == pytest.approx(0.5)

    check_orthogonality_center(state, 2)


@pytest.mark.parametrize("basis", (("0", "1"), ("r", "g", "x")))
def test_apply_random_orthogonality_center(basis):
    dim = len(basis)
    state = MPS.make(4, eigenstates=basis)
    r1 = torch.rand(dim, dim, dtype=dtype)
    r2 = torch.rand(dim, dim, dtype=dtype)
    state.apply(1, r1)
    state.apply(2, r2)

    check_orthogonality_center(state, 2)


@pytest.mark.parametrize("basis", (("0", "1"), ("g", "r"), ("g", "r", "x")))
def test_correlation_matrix_random(basis):
    qubit_count = 5
    dim = len(basis)
    state = MPS(
        [
            torch.rand(1, dim, 3, dtype=dtype),
            torch.rand(3, dim, 5, dtype=dtype),
            torch.rand(5, dim, 12, dtype=dtype),
            torch.rand(12, dim, 2, dtype=dtype),
            torch.rand(2, dim, 1, dtype=dtype),
        ],
        eigenstates=basis,
    )

    correlation_matrix_nn = state.get_correlation_matrix()

    z_op = torch.zeros(dim, dim, dtype=dtype)
    z_op[0, 0] = 1.0
    z_op[1, 1] = -1.0

    correlation_matrix_zz = state.get_correlation_matrix(z_op)
    excited_excited = basis[1] + basis[1]

    def nn(index1, index2):
        return MPO.from_operator_repr(
            eigenstates=basis,
            n_qudits=qubit_count,
            operations=[(1.0, [({excited_excited: 1.0}, list({index1, index2}))])],
        )

    ground_ground = basis[0] + basis[0]

    def zz(index1, index2):
        return MPO.from_operator_repr(
            eigenstates=basis,
            n_qudits=qubit_count,
            operations=[
                (
                    1.0,
                    [
                        (
                            {ground_ground: 1.0, excited_excited: -1.0},
                            list({index1, index2}),
                        )
                    ],
                )
            ],
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
        (("r", "g", "x"), {4 * "x": 1.0}),
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
        (("r", "g", "x"), {4 * "1": 1.0}),
    ],
)
def test_constructor(eig, ampl) -> None:
    with pytest.raises(ValueError):
        MPS.from_state_amplitudes(eigenstates=eig, amplitudes=ampl)
