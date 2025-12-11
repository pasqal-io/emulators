import torch
import pytest
from functools import reduce

from emu_sv import StateVector, SparseOperator

dtype = torch.complex128
X = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
Y = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype)
Z = torch.tensor([[1, 0], [0, -1]], dtype=dtype)
Id = torch.tensor([[1, 0], [0, 1]], dtype=dtype)
zero_state_torch = torch.tensor([[1, 0]], dtype=dtype)
one_state_torch = torch.tensor([[0, 1]], dtype=dtype)


@pytest.mark.parametrize(("zero", "one"), [("g", "r")])
def test_from_operator_repr_and_rmul(zero: str, one: str) -> None:
    # creation 2 qubit operator X_0Z_1
    N = 2
    operations = [
        (
            1.0,
            [
                ({zero + one: 1.0, one + zero: -1.0}, {0}),  # iY
                ({zero + zero: 1.0, one + one: -1.0}, {1}),  # Z
            ],
        )
    ]

    operator = SparseOperator.from_operator_repr(
        eigenstates=(one, zero),
        n_qudits=N,
        operations=operations,
    )
    expected = torch.kron(1j * Y, Z)
    assert torch.allclose(operator.matrix.to_dense().cpu(), expected)

    # multiplying an operator by a number
    operator = 3 * operator  # testing __rmul__
    expected = 3 * expected
    assert torch.allclose(operator.matrix.to_dense().cpu(), expected)


@pytest.mark.parametrize(("zero", "one"), [("g", "r")])
def test_add(zero: str, one: str) -> None:
    N = 2

    ops_1 = [
        (
            2.0,
            [
                ({zero + one: 1.0, one + zero: 1.0}, [0]),  # X
            ],
        )
    ]

    ops_2 = [
        (
            1.0,
            [
                ({zero + zero: 1.0, one + one: -1.0}, [1]),  # Z
            ],
        )
    ]

    operator_1 = SparseOperator.from_operator_repr(
        eigenstates=(one, zero),
        n_qudits=N,
        operations=ops_1,
    )

    operator_2 = SparseOperator.from_operator_repr(
        eigenstates=(one, zero),
        n_qudits=N,
        operations=ops_2,
    )

    # sum of 2 operators
    op = operator_1 + operator_2  # testing __add__
    expected = torch.kron(2 * X, Id) + torch.kron(Id, Z)
    assert torch.allclose(op.matrix.to_dense().cpu(), expected)


@pytest.mark.parametrize(("zero", "one"), [("g", "r")])
def test_applyto_expect(zero: str, one: str) -> None:
    # creation 2 qubit operator X_0Z_2
    N = 3
    operations = [
        (
            1.0,
            [
                ({zero + one: 1.0, one + zero: 1.0}, {0}),  # X
                ({zero + zero: 1.0, one + one: -1.0}, {2}),  # Z
            ],
        )
    ]

    operator = SparseOperator.from_operator_repr(
        eigenstates=(one, zero),
        n_qudits=N,
        operations=operations,
    )

    state = {one + one + one: -1.0, zero + zero + zero: 1.0}
    state_from_string = StateVector.from_state_amplitudes(
        eigenstates=(one, zero), amplitudes=state
    )
    result = operator.apply_to(state_from_string)  # testing apply_to

    state_torch = reduce(torch.kron, [zero_state_torch] * N)
    state_torch -= reduce(torch.kron, [one_state_torch] * N)
    state_torch /= torch.norm(state_torch, p=2)
    state_torch = state_torch.T

    mult_state = reduce(torch.kron, [X, Id, Z]) @ state_torch
    mult_state = mult_state.T

    assert torch.allclose(mult_state, result.vector.cpu())

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
        SparseOperator.from_operator_repr(
            eigenstates=("g", "1"), n_qudits=3, operations=operations
        )
    msg = "Every QuditOp key must be made up of two eigenstates among ('g', '1'); instead, got 'X'."
    assert str(ve.value) == msg
