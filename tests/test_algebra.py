import pytest
import torch

from emu_mps.algebra import _add_factors, _mul_factors


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
        _add_factors(factors_1, factors_2)
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

        factors_sum = _add_factors(factors_1, factors_2)

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

        factors_sum = _add_factors(factors_1, factors_2)

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


def test_mul_factors():
    num_sites = 5
    linkdim1 = 8
    for sitedims in [(2,), (2, 2)]:  # MPS/O-like factors
        factors = random_factors(num_sites, sitedims, linkdim=linkdim1, dtype=dtype)
        for scale in [3.0, 2j, -1 / 4]:
            scaled_factors = _mul_factors(factors, scale)
            # all but 0 factor unchanged
            assert torch.equal(scaled_factors[0], scale * factors[0])
            for f1, f2 in zip(scaled_factors[1:], factors[1:]):
                assert f1 is f2
