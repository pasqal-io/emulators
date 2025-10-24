import pytest
import torch
import math

from emu_mps import MPO, MPS

dtype = torch.complex128


@pytest.mark.parametrize("basis", (("0", "1"), ("g", "r"), ("g", "r", "x")))
def test_mul(basis):
    num_sites = 3
    dim = len(basis)
    mps = MPS.make(num_sites, eigenstates=basis)
    factors = []
    for _ in range(num_sites):
        tensor = torch.zeros(1, dim, dim, 1, dtype=dtype)
        tensor[0, 0, 1, 0] = 1
        tensor[0, 1, 0, 0] = 1
        factors.append(tensor)
    mpo = MPO(factors)
    out = mpo.apply_to(mps)
    for i in out.factors:
        if dim == 2:
            assert torch.allclose(
                i, torch.tensor([[[0], [1]]], dtype=dtype, device=i.device)
            )
        else:
            assert torch.allclose(
                i, torch.tensor([[[0], [1], [0]]], dtype=dtype, device=i.device)
            )


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
        MPO.from_operator_repr(eigenstates=("g", "1"), n_qudits=3, operations=operations)
    msg = "Every QuditOp key must be made up of two eigenstates among ('g', '1'); instead, got 'X'."
    assert str(ve.value) == msg


@pytest.mark.parametrize("basis", (("0", "1"), ("g", "r"), ("g", "r", "x")))
def test_from_operator_string(basis):
    dim = len(basis)
    zero = basis[0]
    one = basis[1]

    assert zero == "0" or "g"  # in this test the order of basis is important
    assert one == "1" or "r"

    x = {zero + one: 2.0, one + zero: 2.0}
    z = {zero + zero: 3.0, one + one: -3.0}

    operations = [
        (
            1.0,
            [
                (x, [0, 2]),
                (z, [1]),
            ],
        )
    ]
    mpo = MPO.from_operator_repr(eigenstates=basis, n_qudits=3, operations=operations)
    x_matrix = torch.zeros(dim, dim, dtype=dtype, device=mpo.factors[0].device)
    x_matrix[0, 1] = 1.0
    x_matrix[1, 0] = 1.0
    x_matrix = x_matrix.reshape(1, dim, dim, 1)
    z_matrix = torch.zeros(dim, dim, dtype=dtype, device=mpo.factors[0].device)
    z_matrix[0, 0] = 1.0
    z_matrix[1, 1] = -1.0
    z_matrix = z_matrix.reshape(1, dim, dim, 1)

    assert torch.allclose(
        mpo.factors[0],
        2 * x_matrix,
    )
    assert torch.allclose(
        mpo.factors[1],
        3 * z_matrix,
    )
    assert torch.allclose(
        mpo.factors[2],
        2 * x_matrix,
    )


def test_from_operator_string_with_leak():
    basis = ("g", "r", "x")
    dim = len(basis)
    one_to_x_op = {basis[2] + basis[1]: 2.0}
    zero_to_x_op = {basis[2] + basis[0]: 3.0}

    operations = [
        (
            1.0,
            [
                (one_to_x_op, [0, 2]),
                (zero_to_x_op, [1]),
            ],
        )
    ]
    mpo = MPO.from_operator_repr(eigenstates=basis, n_qudits=3, operations=operations)

    one_to_x_matrix = torch.zeros(dim, dim, dtype=dtype, device=mpo.factors[0].device)
    one_to_x_matrix[2, 1] = 1.0
    one_to_x_matrix = one_to_x_matrix.reshape(1, dim, dim, 1)

    zero_to_x_matrix = torch.zeros(dim, dim, dtype=dtype, device=mpo.factors[0].device)
    zero_to_x_matrix[2, 0] = 1.0
    zero_to_x_matrix = zero_to_x_matrix.reshape(1, dim, dim, 1)

    assert torch.allclose(
        mpo.factors[0],
        2 * one_to_x_matrix,
    )
    assert torch.allclose(
        mpo.factors[1],
        3 * zero_to_x_matrix,
    )
    assert torch.allclose(
        mpo.factors[2],
        2 * one_to_x_matrix,
    )


@pytest.mark.parametrize("basis", (("0", "1"), ("g", "r"), ("g", "r", "x")))
def test_wrong_external_links(basis):
    dim = len(basis)
    sitedims = (dim, dim)
    factor1 = torch.rand(3, *sitedims, 5)
    factor2 = torch.rand(5, *sitedims, 5)
    good_left_factor = torch.rand(1, *sitedims, 3)
    wrong_left_factor = torch.rand(3, *sitedims, 3)
    good_right_factor = torch.rand(5, *sitedims, 1)
    wrong_right_factor = torch.rand(5, *sitedims, 5)

    MPO([good_left_factor, factor1, factor2, good_right_factor])

    with pytest.raises(ValueError) as ve:
        MPO([wrong_left_factor, factor1, factor2, good_right_factor])
    msg = "The dimension of the left (right) link of the first (last) tensor should be 1"
    assert str(ve.value) == msg

    with pytest.raises(ValueError) as ve:
        MPO([good_left_factor, factor1, factor2, wrong_right_factor])
    msg = "The dimension of the left (right) link of the first (last) tensor should be 1"
    assert str(ve.value) == msg


@pytest.mark.parametrize("basis", (("0", "1"), ("g", "r"), ("g", "r", "x")))
def test_expect(basis):
    dim = len(basis)
    nqubits = 5

    def shape(index: int, size: int) -> tuple[int, ...]:
        if index == 0:
            return (1,) + (dim,) * size + (10,)
        elif index == 4:
            return (10,) + (dim,) * size + (1,)
        else:
            return (10,) + (dim,) * size + (10,)

    state = MPS(
        [torch.randn(*shape(i, 1), dtype=dtype) for i in range(nqubits)],
        eigenstates=basis,
    )
    op = MPO([torch.randn(*shape(i, 2), dtype=dtype) for i in range(nqubits)])
    assert op.expect(state) == pytest.approx(state.inner(op.apply_to(state)))


@pytest.mark.parametrize("basis", (("0", "1"), ("g", "r"), ("g", "r", "x")))
def test_add_expectation_values(basis):
    """
    Test that the expectation value of MPOs, and the sum
    of expectation values of each MPO is equivalent.
    TODO: move to integration tests
    """

    num_sites = 3
    dim = len(basis)
    id_factor = torch.eye(dim, dim, dtype=dtype).reshape(1, dim, dim, 1)
    sigma_rr = torch.zeros(dim, dim, dtype=dtype)
    sigma_rr[1, 1] = 1.0
    sigma_rr = sigma_rr.reshape(1, dim, dim, 1)

    # arbitrary op list Oi
    mpo_list = []
    for i in range(num_sites):
        factors = [sigma_rr if (i == j) else id_factor for j in range(num_sites)]
        mpo_list.append(MPO(factors))
    mpo_sum = sum(mpo_list[1:], start=mpo_list[0])

    # arbitrary state |Ψ〉
    mps = MPS(
        [
            torch.rand(1, dim, 2, dtype=dtype),
            torch.rand(2, dim, 6, dtype=dtype),
            torch.rand(6, dim, 1, dtype=dtype),
        ],
        eigenstates=basis,
    )

    # compute 〈Ψ|O|Ψ〉= Σi〈Ψ|Oi|Ψ〉
    observable_expected = sum(op.expect(mps) for op in mpo_list)
    # compute 〈Ψ|O|Ψ〉=〈Ψ|Σi Oi|Ψ〉
    observable = mpo_sum.expect(mps)

    assert observable == pytest.approx(observable_expected, 1e-12)


@pytest.mark.parametrize("basis", (("0", "1"), ("g", "r"), ("g", "r", "x")))
def test_matmul(basis):
    dim = len(basis)
    # make random MPS
    mps = MPS(
        [
            torch.rand(1, dim, 2, dtype=dtype),
            torch.rand(2, dim, 6, dtype=dtype),
            torch.rand(6, dim, 5, dtype=dtype),
            torch.rand(5, dim, 1, dtype=dtype),
        ],
        eigenstates=basis,
    )
    # normalize
    mps = (1 / mps.norm()) * mps

    # make random MPO1
    mpo1 = MPO(
        [
            torch.rand(1, dim, dim, 4, dtype=dtype),
            torch.rand(4, dim, dim, 6, dtype=dtype),
            torch.rand(6, dim, dim, 5, dtype=dtype),
            torch.rand(5, dim, dim, 1, dtype=dtype),
        ]
    )
    # make random MPO2
    mpo2 = MPO(
        [
            torch.rand(1, dim, dim, 7, dtype=dtype),
            torch.rand(7, dim, dim, 6, dtype=dtype),
            torch.rand(6, dim, dim, 8, dtype=dtype),
            torch.rand(8, dim, dim, 1, dtype=dtype),
        ]
    )

    # test (O @ O)*|Ψ〉= O*(O*|Ψ〉)
    matmul_mps = (mpo1 @ mpo2).apply_to(mps)
    mul_mps = mpo1.apply_to(mpo2.apply_to(mps))

    # assert same projection on initial state |Ψ〉
    expected = mps.inner(matmul_mps)
    obtained = mps.inner(mul_mps)
    assert obtained == pytest.approx(expected)

    # assert orthogonality center at qubit #0
    assert matmul_mps.orthogonality_center == 0
    assert mul_mps.orthogonality_center == 0
    assert all(
        torch.allclose(
            torch.tensordot(f.conj(), f, dims=([1, 2], [1, 2])),
            torch.eye(f.shape[0], dtype=dtype, device=f.device),
        )
        for f in matmul_mps.factors[1:] + mul_mps.factors[1:]
    )


@pytest.mark.parametrize("basis", (("0", "1"), ("g", "r"), ("g", "r", "x")))
def test_rmul(basis):
    dim = len(basis)
    # make random MPS
    mps = MPS(
        [
            torch.rand(1, dim, 2, dtype=dtype),
            torch.rand(2, dim, 6, dtype=dtype),
            torch.rand(6, dim, 5, dtype=dtype),
            torch.rand(5, dim, 1, dtype=dtype),
        ],
        eigenstates=basis,
    )
    # normalize
    mps = (1 / mps.norm()) * mps

    # make random MPO
    mpo = MPO(
        [
            torch.rand(1, dim, dim, 4, dtype=dtype),
            torch.rand(4, dim, dim, 6, dtype=dtype),
            torch.rand(6, dim, dim, 5, dtype=dtype),
            torch.rand(5, dim, dim, 1, dtype=dtype),
        ]
    )
    # test 〈Ψ|a*O|Ψ〉== a〈Ψ|O|Ψ〉)
    unscaled_res = mpo.expect(mps)
    for scale in [2.0, -math.pi, -1 + 2j, -1 / 4]:
        scaled_mpo = scale * mpo
        assert scaled_mpo.expect(mps) == pytest.approx(scale * unscaled_res)
