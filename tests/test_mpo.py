import pytest
import torch
import math

from emu_mps import MPO, MPS, inner


def test_mul():
    num_sites = 3

    mps = MPS.make(num_sites)
    factors = []
    for _ in range(num_sites):
        tensor = torch.zeros(1, 2, 2, 1, dtype=torch.complex128)
        tensor[0, 0, 1, 0] = 1
        tensor[0, 1, 0, 0] = 1
        factors.append(tensor)
    mpo = MPO(factors)
    out = mpo * mps
    for i in out.factors:
        assert torch.allclose(
            i, torch.tensor([[[0], [1]]], dtype=torch.complex128, device=i.device)
        )


def test_from_operator_string():
    x = {"sigma_gr": 1.0, "sigma_rg": 1.0}
    z = {"sigma_gg": 1.0, "sigma_rr": -1.0}
    operators = {"X": x, "Z": z}
    operations = [
        (
            1.0,
            [
                ({"X": 2.0}, [0, 2]),
                ({"Z": 3.0}, [1]),
            ],
        )
    ]
    mpo = MPO.from_operator_string(("r", "g"), 3, operations, operators)
    assert torch.allclose(
        mpo.factors[0],
        torch.tensor(
            [[0.0, 2.0], [2.0, 0.0]], dtype=torch.complex128, device=mpo.factors[0].device
        ).reshape(1, 2, 2, 1),
    )
    assert torch.allclose(
        mpo.factors[1],
        torch.tensor(
            [[3.0, 0.0], [0.0, -3.0]],
            dtype=torch.complex128,
            device=mpo.factors[1].device,
        ).reshape(1, 2, 2, 1),
    )
    assert torch.allclose(
        mpo.factors[2],
        torch.tensor(
            [[0.0, 2.0], [2.0, 0.0]], dtype=torch.complex128, device=mpo.factors[2].device
        ).reshape(1, 2, 2, 1),
    )


def test_wrong_external_links():
    sitedims = (2, 2)
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


def test_add_expectation_values():
    """
    Test that the expectation value of MPOs, and the sum
    of expectation values of each MPO is equivalent.
    TODO: move to integration tests
    """
    dtype = torch.complex128
    num_sites = 3
    id_factor = torch.eye(2, 2, dtype=dtype).reshape(1, 2, 2, 1)
    sigma_rr = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=dtype).reshape(1, 2, 2, 1)

    # arbitrary op list Oi
    mpo_list = []
    for i in range(num_sites):
        factors = [sigma_rr if (i == j) else id_factor for j in range(num_sites)]
        mpo_list.append(MPO(factors))
    mpo_sum = sum(mpo_list[1:], start=mpo_list[0])

    # arbitrary state |Ψ〉
    mps = MPS(
        [
            torch.rand(1, 2, 2, dtype=dtype),
            torch.rand(2, 2, 6, dtype=dtype),
            torch.rand(6, 2, 1, dtype=dtype),
        ],
    )

    # compute 〈Ψ|O|Ψ〉= Σi〈Ψ|Oi|Ψ〉
    observable_expected = sum(inner(mps, op * mps) for op in mpo_list)
    # compute 〈Ψ|O|Ψ〉=〈Ψ|Σi Oi|Ψ〉
    observable = inner(mps, mpo_sum * mps)

    assert observable == pytest.approx(observable_expected, 1e-12)


def test_matmul():
    dtype = torch.complex128
    # make random MPS
    mps = MPS(
        [
            torch.rand(1, 2, 2, dtype=dtype),
            torch.rand(2, 2, 6, dtype=dtype),
            torch.rand(6, 2, 5, dtype=dtype),
            torch.rand(5, 2, 1, dtype=dtype),
        ],
    )
    # normalize
    mps = (1 / mps.norm()) * mps

    # make random MPO1
    mpo1 = MPO(
        [
            torch.rand(1, 2, 2, 4, dtype=dtype),
            torch.rand(4, 2, 2, 6, dtype=dtype),
            torch.rand(6, 2, 2, 5, dtype=dtype),
            torch.rand(5, 2, 2, 1, dtype=dtype),
        ]
    )
    # make random MPO2
    mpo2 = MPO(
        [
            torch.rand(1, 2, 2, 7, dtype=dtype),
            torch.rand(7, 2, 2, 6, dtype=dtype),
            torch.rand(6, 2, 2, 8, dtype=dtype),
            torch.rand(8, 2, 2, 1, dtype=dtype),
        ]
    )

    # test (O @ O)*|Ψ〉= O*(O*|Ψ〉)
    matmul_mps = (mpo1 @ mpo2) * mps
    mul_mps = mpo1 * (mpo2 * mps)

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
            torch.eye(f.shape[0], dtype=torch.complex128, device=f.device),
        )
        for f in matmul_mps.factors[1:] + mul_mps.factors[1:]
    )


def test_rmul():
    dtype = torch.complex128
    # make random MPS
    mps = MPS(
        [
            torch.rand(1, 2, 2, dtype=dtype),
            torch.rand(2, 2, 6, dtype=dtype),
            torch.rand(6, 2, 5, dtype=dtype),
            torch.rand(5, 2, 1, dtype=dtype),
        ],
    )
    # normalize
    mps = (1 / mps.norm()) * mps

    # make random MPO
    mpo = MPO(
        [
            torch.rand(1, 2, 2, 4, dtype=dtype),
            torch.rand(4, 2, 2, 6, dtype=dtype),
            torch.rand(6, 2, 2, 5, dtype=dtype),
            torch.rand(5, 2, 2, 1, dtype=dtype),
        ]
    )
    # test 〈Ψ|a*O|Ψ〉== a〈Ψ|O|Ψ〉)
    unscaled_res = inner(mps, mpo * mps)
    for scale in [2.0, -math.pi, -1 + 2j, -1 / 4]:
        scaled_mpo = scale * mpo
        assert inner(mps, scaled_mpo * mps) == pytest.approx(scale * unscaled_res)
