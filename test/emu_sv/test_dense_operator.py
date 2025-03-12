import torch
import pytest
from functools import reduce

from emu_sv.dense_operator import DenseOperator
from emu_sv.state_vector import StateVector


dtype = torch.complex128
X = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
Y = torch.tensor([[0, 1j], [-1j, 0]], dtype=dtype)
Z = torch.tensor([[1, 0], [0, -1]], dtype=dtype)
Id = torch.tensor([[1, 0], [0, 1]], dtype=dtype)
zero_state_torch = torch.tensor([[1, 0]], dtype=dtype)
one_state_torch = torch.tensor([[0, 1]], dtype=dtype)


@pytest.mark.parametrize(
    ("zero", "one"),
    (("g", "r"), ("0", "1")),
)
def test_from_operator_repr_and_rmul_DenseOperator(zero: str, one: str) -> None:
    # creation 2 qubit operator X_0Z_1
    N = 2
    operations = [
        (
            1.0,
            [
                ({zero + one: 1.0, one + zero: 1.0}, {0}),  # X
                ({zero + zero: 1.0, one + one: -1.0}, {1}),  # Z
            ],
        )
    ]

    operator = DenseOperator.from_operator_repr(
        eigenstates=(one, zero),
        n_qudits=N,
        operations=operations,
    )
    expected = torch.kron(X, Z)
    assert torch.allclose(operator.matrix.cpu(), expected)

    # multiplying an operator by a number
    operator = 3 * operator  # testing __rmul__
    expected = 3 * expected
    assert torch.allclose(operator.matrix.cpu(), expected)


@pytest.mark.parametrize(
    ("zero", "one"),
    (("g", "r"), ("0", "1")),
)
def test_matmul_and_add_DenseOperator(zero: str, one: str) -> None:
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

    operator_1 = DenseOperator.from_operator_repr(
        eigenstates=(one, zero),
        n_qudits=N,
        operations=ops_1,
    )

    operator_2 = DenseOperator.from_operator_repr(
        eigenstates=(one, zero),
        n_qudits=N,
        operations=ops_2,
    )

    # sum of 2 operators
    op = operator_1 + operator_2  # testing __add__
    expected = torch.kron(2 * X, Id) + torch.kron(Id, Z)
    assert torch.allclose(op.matrix.cpu(), expected)

    # product of 2 operators
    op = operator_1 @ operator_2  # testing __matmul__
    expected = torch.kron(2 * X, Id) @ torch.kron(Id, Z)
    assert torch.allclose(op.matrix.cpu(), expected)


@pytest.mark.parametrize(
    ("zero", "one"),
    (("g", "r"), ("0", "1")),
)
def test_applyto_expect_DenseOperator(zero: str, one: str) -> None:
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

    operator = DenseOperator.from_operator_repr(
        eigenstates=(one, zero),
        n_qudits=N,
        operations=operations,
    )

    state = {one + one + one: -1.0, zero + zero + zero: 1.0}
    from_string = StateVector.from_state_string(
        basis={one, zero}, nqubits=N, strings=state
    )
    result = operator.apply_to(from_string)  # testing apply_to

    state_torch = reduce(torch.kron, [zero_state_torch] * N)
    state_torch -= reduce(torch.kron, [one_state_torch] * N)
    state_torch /= torch.norm(state_torch, p=2)
    state_torch = state_torch.T

    mult_state = reduce(torch.kron, [X, Id, Z]) @ state_torch
    mult_state = mult_state.T

    assert torch.allclose(mult_state, result.vector.cpu())

    # expectation value
    res = operator.expect(from_string)
    expected = state_torch.mH @ mult_state.T
    assert torch.isclose(torch.tensor(res, dtype=dtype), expected)
