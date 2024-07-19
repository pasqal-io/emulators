import torch
import pytest
from emu_ct.algebra import _add_factors, _mul_factors

dtype = torch.complex128


def random_mps_factors(num_sites: int, linkdim: int = 1, dtype: type = torch.complex128):
    """
    Returns `num_sites` MPS-like factor list, with uniform link dimension `linkdim`.
    """
    factors = [torch.rand(1, 2, linkdim, dtype=dtype)]
    for _ in range(1, num_sites - 1):
        factors.append(torch.rand(linkdim, 2, linkdim, dtype=dtype))
    factors.append(torch.rand(linkdim, 2, 1, dtype=dtype))
    return factors


def test_add_wrong_len():
    linkdim = 6
    factors_1 = random_mps_factors(3, linkdim=linkdim, dtype=dtype)
    factors_2 = random_mps_factors(5, linkdim=linkdim, dtype=dtype)
    with pytest.raises(ValueError) as verr:
        _add_factors(factors_1, factors_2)
    assert (
        str(verr.value) == "Cannot sum two matrix products of different number of sites"
    )


def test_add_mps_factors_linkdims():
    num_sites = 5
    linkdim1 = 7
    factors_1 = random_mps_factors(num_sites, linkdim=linkdim1, dtype=dtype)
    linkdim2 = 4
    factors_2 = random_mps_factors(num_sites, linkdim=linkdim2, dtype=dtype)

    linkdim_sum = linkdim1 + linkdim2
    factors_sum = _add_factors(factors_1, factors_2)

    assert factors_sum[0].shape == (1, 2, linkdim_sum)
    for core in factors_sum[1:-2]:
        assert core.shape == (linkdim_sum, 2, linkdim_sum)
    assert factors_sum[-1].shape == (linkdim_sum, 2, 1)


def test_add_mps_factors_blockdiag():
    num_sites = 5
    linkdim1 = 8
    factors_1 = random_mps_factors(num_sites, linkdim=linkdim1, dtype=dtype)
    linkdim2 = 10
    factors_2 = random_mps_factors(num_sites, linkdim=linkdim2, dtype=dtype)

    factors_sum = _add_factors(factors_1, factors_2)

    # test block diag construction [A 0; 0 B]
    for i, factor in enumerate(factors_sum):
        A_view = factor[:linkdim1, :, :linkdim1]
        A_expected = factors_1[i]
        assert torch.allclose(A_view, A_expected)

        B_view = factor[-linkdim2:, :, -linkdim2:]
        B_expected = factors_2[i]
        assert torch.allclose(B_view, B_expected)

        if i == 0:
            pass
        elif i == len(factors_sum) - 1:
            pass
        else:
            pad_1_view = factor[-linkdim2:, :, :linkdim1]
            pad_1_expected = torch.zeros(linkdim2, 2, linkdim1, dtype=dtype)
            assert torch.allclose(pad_1_view, pad_1_expected)

            pad_2_view = factor[:linkdim1, :, -linkdim2:]
            pad_2_expected = torch.zeros(linkdim1, 2, linkdim2, dtype=dtype)
            assert pad_2_view.shape == pad_2_expected.shape
            assert torch.allclose(pad_2_view, pad_2_expected)


def test_mul_factors():
    num_sites = 5
    linkdim1 = 8
    factors = random_mps_factors(num_sites, linkdim=linkdim1, dtype=dtype)
    for scale in [3.0, 2j, -1 / 4]:
        scaled_factors = _mul_factors(factors, scale)
        # all but 0 factor unchanged
        assert torch.equal(scaled_factors[0], scale * factors[0])
        for f1, f2 in zip(scaled_factors[1:], factors[1:]):
            assert torch.equal(f1[0], f2[0])
