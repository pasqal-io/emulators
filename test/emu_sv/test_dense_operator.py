import torch
import pytest


from emu_sv.dense_operator import DenseOperator
from emu_sv.state_vector import StateVector


@pytest.mark.parametrize(
    ("zero", "one"),
    (("g", "r"), ("0", "1")),
)
def test_algebra_dense_op(zero: str, one: str) -> None:

    N = 3
    # define first operator
    x = {zero + one: 1.0, one + zero: 1.0}
    z = {zero + zero: 1.0, one + one: -1.0}

    operators_a = {"X": x, "Z": z}
    operations_a = [
        (
            1.0,
            [
                ({"X": 2.0}, [0, 2]),
                ({"Z": 3.0}, [1]),
            ],
        )
    ]

    oper_a = DenseOperator.from_operator_string({one, zero}, N, operations_a, operators_a)

    expected_a = torch.zeros(2**N, 2**N, dtype=torch.complex128)
    expected_a[0, 5] = 12.0
    expected_a[1, 4] = 12.0
    expected_a[2, 7] = -12.0
    expected_a[3, 6] = -12.0
    expected_a = expected_a + expected_a.T

    assert torch.allclose(oper_a.matrix.cpu(), expected_a)

    # define second operator

    y = {zero + one: -1.0j, one + zero: 1.0j}
    #
    operators_b = {"X": x, "Y": y}
    #
    operations_b = [
        (
            2.0,
            [
                ({"X": 2.0}, [0, 2]),
                ({"Y": 3.0}, [1]),
            ],
        )
    ]
    oper_b = DenseOperator.from_operator_string({one, zero}, N, operations_b, operators_b)

    expected_b = torch.zeros(2**N, 2**N, dtype=torch.complex128)
    expected_b[0, 7] = 24.0j
    expected_b[1, 6] = 24.0j
    expected_b[2, 5] = -24.0j
    expected_b[3, 4] = -24.0j
    expected_b = expected_b + torch.conj(expected_b).T

    assert torch.allclose(oper_b.matrix.cpu(), expected_b)

    # summing 2 operators

    result_sum = oper_a + oper_b
    #
    expected_sum = torch.zeros(2**N, 2**N, dtype=torch.complex128)
    expected_sum[0, 5] = 12.0
    expected_sum[1, 4] = 12.0
    expected_sum[2, 7] = -12.0
    expected_sum[3, 6] = -12.0
    expected_sum[0, 7] = 24.0j
    expected_sum[1, 6] = 24.0j
    expected_sum[2, 5] = -24.0j
    expected_sum[3, 4] = -24.0j
    expected_sum = expected_sum + torch.conj(expected_sum).T

    assert torch.allclose(expected_sum, result_sum.matrix.cpu())

    # multiplication by scalar

    result_mul_r = 5.0 * oper_a

    expected_mult_r = torch.zeros(2**N, 2**N, dtype=torch.complex128)
    expected_mult_r[0, 5] = 12.0 * 5
    expected_mult_r[1, 4] = 12.0 * 5
    expected_mult_r[2, 7] = -12.0 * 5
    expected_mult_r[3, 6] = -12.0 * 5

    expected_mult_r = expected_mult_r + expected_mult_r.T

    assert torch.allclose(result_mul_r.matrix.cpu(), expected_mult_r)

    # application to a StateVector

    state = {one + one + one: 1.0, zero + zero + zero: 1.0}
    nqubits = 3
    from_string = StateVector.from_state_string(
        basis={one, zero}, nqubits=nqubits, strings=state
    )

    result_mult_state = oper_a * from_string

    expected_mult_state = torch.tensor(
        [
            0.0000 + 0.0j,
            0.0000 + 0.0j,
            -8.4853 + 0.0j,
            0.0000 + 0.0j,
            0.0000 + 0.0j,
            8.4853 + 0.0j,
            0.0000 + 0.0j,
            0.0000 + 0.0j,
        ],
        dtype=torch.complex128,
    )

    assert torch.allclose(expected_mult_state, result_mult_state.vector.cpu())

    # expectation value

    result_exp_a = oper_a.expect(from_string)
    result_exp_b = oper_b.expect(from_string)

    assert result_exp_a == result_exp_b == 0.0j

    # multiplication of 2 operators

    result_op_mult_op = oper_a @ oper_b

    expected_mult_op_op = torch.zeros(2**N, 2**N, dtype=torch.complex128)
    expected_mult_op_op[0, 2] = 12 * 24j
    expected_mult_op_op[2, 0] = 12 * 24j

    expected_mult_op_op[1, 3] = 12 * 24j
    expected_mult_op_op[3, 1] = 12 * 24j

    expected_mult_op_op[4, 6] = 12 * 24j
    expected_mult_op_op[6, 4] = 12 * 24j

    expected_mult_op_op[5, 7] = 12 * 24j
    expected_mult_op_op[7, 5] = 12 * 24j

    assert torch.allclose(result_op_mult_op.matrix.cpu(), expected_mult_op_op)


@pytest.mark.parametrize(
    ("zero", "one"),
    (("g", "r"), ("0", "1")),
)
def test_single_dense_operator(zero: str, one: str) -> None:

    N = 3
    # define first operator
    x = {zero + one: 1.0, one + zero: 1.0}
    z = {zero + zero: 1.0, one + one: -1.0}

    operators_a = {"X": x, "Z": z}
    operations_a = [
        (
            1.7,
            [
                ({"X": -2.0}, [0, 2]),
                ({"Z": 0.3}, [1]),
            ],
        )
    ]

    oper_a = DenseOperator.from_operator_string({one, zero}, N, operations_a, operators_a)

    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
    expected = 1.7 * torch.kron(torch.kron(-2 * X, 0.3 * Z), -2 * X)

    assert torch.allclose(oper_a.matrix.cpu(), expected)
