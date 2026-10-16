import torch
import pytest
from functools import reduce

from emu_sv import StateVector, DenseOperator

dtype = torch.complex128
X = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
Y = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype)
Z = torch.tensor([[1, 0], [0, -1]], dtype=dtype)
Id = torch.tensor([[1, 0], [0, 1]], dtype=dtype)
zero_state_torch = torch.tensor([[1, 0]], dtype=dtype)
one_state_torch = torch.tensor([[0, 1]], dtype=dtype)


def test_from_operator_repr_and_rmul_DenseOperator() -> None:
    # creation 2 qubit operator X_0Z_1
    N = 2
    operations = [
        (
            1.0,
            [
                ({"gr": 1.0, "rg": -1.0}, {0}),  # iY
                ({"gg": 1.0, "rr": -1.0}, {1}),  # Z
            ],
        )
    ]

    operator = DenseOperator.from_operator_repr(
        eigenstates=("r", "g"),
        n_qudits=N,
        operations=operations,
    )
    expected = torch.kron(1j * Y, Z)
    assert torch.allclose(operator.data.cpu(), expected)

    # multiplying an operator by a number
    operator = 3 * operator  # testing __rmul__
    expected = 3 * expected
    assert torch.allclose(operator.data.cpu(), expected)


def test_matmul_and_add_DenseOperator() -> None:
    N = 2

    ops_1 = [
        (
            2.0,
            [
                ({"gr": 1.0, "rg": 1.0}, [0]),  # X
            ],
        )
    ]

    ops_2 = [
        (
            1.0,
            [
                ({"gg": 1.0, "rr": -1.0}, [1]),  # Z
            ],
        )
    ]

    operator_1 = DenseOperator.from_operator_repr(
        eigenstates=("r", "g"),
        n_qudits=N,
        operations=ops_1,
    )

    operator_2 = DenseOperator.from_operator_repr(
        eigenstates=("r", "g"),
        n_qudits=N,
        operations=ops_2,
    )

    # sum of 2 operators
    op = operator_1 + operator_2  # testing __add__
    expected = torch.kron(2 * X, Id) + torch.kron(Id, Z)
    assert torch.allclose(op.data.cpu(), expected)

    # product of 2 operators
    op = operator_1 @ operator_2  # testing __matmul__
    expected = torch.kron(2 * X, Id) @ torch.kron(Id, Z)
    assert torch.allclose(op.data.cpu(), expected)


def test_applyto_expect_DenseOperator() -> None:
    # creation 2 qubit operator X_0Z_2
    N = 3
    operations = [
        (
            1.0,
            [
                ({"rg": 1.0, "gr": 1.0}, {0}),  # X
                ({"gg": 1.0, "rr": -1.0}, {2}),  # Z
            ],
        )
    ]

    operator = DenseOperator.from_operator_repr(
        eigenstates=("r", "g"),
        n_qudits=N,
        operations=operations,
    )

    state = {"rrr": -1.0, "ggg": 1.0}
    state_from_string = StateVector.from_state_amplitudes(
        eigenstates=("r", "g"), amplitudes=state
    )
    result = operator.apply_to(state_from_string)  # testing apply_to

    state_torch = reduce(torch.kron, [zero_state_torch] * N)
    state_torch -= reduce(torch.kron, [one_state_torch] * N)
    state_torch /= torch.norm(state_torch, p=2)
    state_torch = state_torch.T

    mult_state = reduce(torch.kron, [X, Id, Z]) @ state_torch
    mult_state = mult_state.T

    assert torch.allclose(mult_state, result.data.cpu())

    # expectation value
    res = operator.expect(state_from_string)
    expected = state_torch.mH @ mult_state.T
    assert torch.isclose(res, expected)


def test_wrong_basis_string_state():
    operations = [
        (
            1.0,
            [
                ({"X": 2.0}, [0, 2]),
                ({"Z": 3.0}, [1]),
            ],
        )
    ]

    with pytest.raises(ValueError) as ve:
        DenseOperator.from_operator_repr(
            eigenstates=("g", "1"), n_qudits=3, operations=operations
        )
    msg = "Every QuditOp key must be made up of two eigenstates among ('g', '1'); instead, got 'X'."
    assert str(ve.value) == msg
