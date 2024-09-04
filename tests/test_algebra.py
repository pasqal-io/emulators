import pytest
import torch
from itertools import product
from emu_mps.algebra import add_factors, scale_factors, zip_right_step, zip_right

dtype = torch.complex128


def random_factors(
    num_sites: int, sitedims: tuple, linkdim: int = 1, dtype: type = torch.complex128
):
    """
    Returns `num_sites` MPS/O-like factor list, with uniform link dimension `linkdim`.
    """
    factors = [torch.rand(1, *sitedims, linkdim, dtype=dtype)]
    for _ in range(1, num_sites - 1):
        factors.append(torch.rand(linkdim, *sitedims, linkdim, dtype=dtype))
    factors.append(torch.rand(linkdim, *sitedims, 1, dtype=dtype))
    return factors


def test_add_wrong_len():
    linkdim = 6
    factors_1 = random_factors(3, (2,), linkdim=linkdim, dtype=dtype)
    factors_2 = random_factors(5, (2,), linkdim=linkdim, dtype=dtype)
    with pytest.raises(ValueError) as verr:
        add_factors(factors_1, factors_2)
    assert (
        str(verr.value) == "Cannot sum two matrix products of different number of sites"
    )


def test_add_factors_linkdims():
    num_sites = 4
    linkdim1 = 7
    linkdim2 = 4
    linkdim_sum = linkdim1 + linkdim2

    for sitedims in [(2,), (2, 2)]:  # MPS/O-like factors
        factors_1 = random_factors(num_sites, sitedims, linkdim=linkdim1, dtype=dtype)
        factors_2 = random_factors(num_sites, sitedims, linkdim=linkdim2, dtype=dtype)

        factors_sum = add_factors(factors_1, factors_2)

        assert factors_sum[0].shape == (1, *sitedims, linkdim_sum)
        for core in factors_sum[1:-2]:
            assert core.shape == (linkdim_sum, *sitedims, linkdim_sum)
        assert factors_sum[-1].shape == (linkdim_sum, *sitedims, 1)


def test_add_factors_blockdiag():
    num_sites = 5
    linkdim1 = 8
    linkdim2 = 10

    for sitedims in [(2,), (2, 2)]:  # MPS/O-like factors
        factors_2 = random_factors(num_sites, sitedims, linkdim=linkdim2, dtype=dtype)
        factors_1 = random_factors(num_sites, sitedims, linkdim=linkdim1, dtype=dtype)

        factors_sum = add_factors(factors_1, factors_2)

        # test block diag construction [A 0; 0 B]
        for i, factor in enumerate(factors_sum):
            A_view = factor[:linkdim1, ..., :linkdim1]
            A_expected = factors_1[i]
            assert torch.equal(A_view, A_expected)

            B_view = factor[-linkdim2:, ..., -linkdim2:]
            B_expected = factors_2[i]
            assert torch.equal(B_view, B_expected)

            if i == 0:
                pass
            elif i == len(factors_sum) - 1:
                pass
            else:
                pad_1_view = factor[-linkdim2:, ..., :linkdim1]
                pad_1_expected = torch.zeros(linkdim2, *sitedims, linkdim1, dtype=dtype)
                assert torch.equal(pad_1_view, pad_1_expected)

                pad_2_view = factor[:linkdim1, ..., -linkdim2:]
                pad_2_expected = torch.zeros(linkdim1, *sitedims, linkdim2, dtype=dtype)
                assert pad_2_view.shape == pad_2_expected.shape
                assert torch.equal(pad_2_view, pad_2_expected)


def test_scale_factors():
    num_sites = 5
    linkdim1 = 8
    for sitedims in [(2,), (2, 2)]:  # MPS/O-like factors
        factors = random_factors(num_sites, sitedims, linkdim=linkdim1, dtype=dtype)
        for scale in [3.0, 2j, -1 / 4]:
            scaled_factors = scale_factors(factors, scale, which=0)
            # all but 0 factor unchanged
            assert torch.equal(scaled_factors[0], scale * factors[0])
            for f1, f2 in zip(scaled_factors[1:], factors[1:]):
                assert f1 is f2

            scaled_factors_second = scale_factors(factors, scale, which=2)
            # all but 2 factor unchanged
            assert torch.equal(scaled_factors_second[2], scale * factors[2])
            for i in [0, 1, 3, 4]:
                assert scaled_factors_second[i] is factors[i]


def test_zip_right_step_mpompo_accuracy():
    """
    Test that the _zip_right function returns the same factors obtained with
    a different contraction order.
    """
    site_dims = [2, 3, 5]
    left_dims = [1, 3, 8]
    A_left_dims = [1, 5, 10]
    B_left_dims = [1, 4, 9]
    A_right_dims = [1, 7, 8]
    B_right_dims = [1, 3, 11]

    for sd, ld, ald, bld, ard, brd in product(
        site_dims, left_dims, A_left_dims, B_left_dims, A_right_dims, B_right_dims
    ):
        slider = torch.rand(ld, ald, bld, dtype=dtype)
        A = torch.rand(ald, sd, sd, ard, dtype=dtype)
        B = torch.rand(bld, sd, sd, brd, dtype=dtype)

        # different contraction order, slider * (A * B)
        expected = torch.tensordot(A, B, dims=([2], [1]))
        expected = torch.tensordot(slider, expected, dims=([1, 2], [0, 3]))
        expected = expected.transpose(2, 3)

        # zip_right contraction order, (slider * A) * B
        C, slider = zip_right_step(slider, A, B)
        zip_result = torch.tensordot(C, slider, dims=([-1], [0]))

        # test shapes
        assert len(slider.shape) == 3
        assert slider.shape[0] == C.shape[-1]
        assert slider.shape[1:] == (ard, brd)
        assert len(C.shape) == 4
        assert C.shape[:-1] == (ld, sd, sd)

        # test accuracy
        dist = torch.dist(expected, zip_result).item()
        assert dist == pytest.approx(0.0, abs=1e-10)


def test_zip_right_step_catch_dim_error():
    """
    Test that the _zip_right function raises an error if the shapes
    of the slider are wrong.
    """
    slider = torch.rand(5, 6, 7)
    A = torch.rand(11, 2, 2, 5)
    B = torch.rand(7, 2, 2, 5)
    with pytest.raises(ValueError) as exception_info:
        zip_right_step(slider, A, B)
    msg = (
        f"Contracted dimensions between the slider, {slider.shape[1:]} on dims 1 and 2, "
        f"and the two factors, {(A.shape[0], B.shape[0])} on dim 0, need to match."
    )
    assert str(exception_info.value) == msg


def test_zip_right_wrong_len():
    linkdim = 5
    factors_1 = random_factors(6, (2, 2), linkdim=linkdim, dtype=dtype)
    factors_2 = random_factors(3, (2, 2), linkdim=linkdim, dtype=dtype)
    with pytest.raises(ValueError) as verr:
        zip_right(factors_1, factors_2)
    assert str(verr.value) == "Cannot multiply two matrix products of different lengths."


def test_zip_right_return_valid_mpo_factors():
    linkdim = 7
    factors_1 = random_factors(6, (2, 2), linkdim=linkdim, dtype=dtype)
    factors_2 = random_factors(6, (2, 2), linkdim=linkdim, dtype=dtype)

    new_factors = zip_right(factors_1, factors_2)
    assert len(new_factors) == 6
    assert new_factors[0].shape[:-1] == (1, 2, 2)
    assert new_factors[-1].shape[1:] == (2, 2, 1)
    for i, f in enumerate(new_factors[1:], start=1):
        assert f.shape[0] == new_factors[i - 1].shape[-1]
