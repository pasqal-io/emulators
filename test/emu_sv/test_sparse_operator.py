import torch
import pytest
from functools import reduce

from emu_sv import StateVector, SparseOperator
from emu_sv.sparse_operator import sparse_kron, sparse_add

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


def test_applyto_expect() -> None:
    # creation 2 qubit operator X_0Z_2
    N = 3
    operations = [
        (
            1.0,
            [
                ({"gr": 1.0, "rg": 1.0}, {0}),  # X
                ({"gg": 1.0, "rr": -1.0}, {2}),  # Z
            ],
        )
    ]

    operator = SparseOperator.from_operator_repr(
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


def test_sparse_kron_and_add():
    size = 100
    density = 4
    a = torch.zeros(size, size)
    b = torch.zeros(size, size)
    inds = torch.randint(0, 100, (2, density * size, 2))
    vals = torch.randn((2, density * size))
    for i in range(density * size):
        a[inds[0, i, 0], inds[0, i, 1]] = vals[0, i]
        b[inds[1, i, 0], inds[1, i, 1]] = vals[1, i]
    sparse_kr = sparse_kron(a.to_sparse_coo(), b.to_sparse_coo())

    assert sparse_kr.layout == torch.sparse_coo
    assert sparse_kr.is_coalesced()
    assert torch.allclose(sparse_kr.to_dense(), torch.kron(a, b))

    sparse_a = sparse_add(a.to_sparse_coo(), b.to_sparse_coo())
    assert sparse_a.layout == torch.sparse_coo
    assert sparse_a.is_coalesced()
    assert torch.allclose(sparse_a.to_dense(), a + b)
