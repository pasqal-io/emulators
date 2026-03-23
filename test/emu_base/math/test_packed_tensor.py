import pytest
import torch
from emu_base import PackedHermitianTensor


def test_roundtrip():
    n, b = 4, 3
    x = torch.randn(n, b, n, dtype=torch.complex64)
    h = 0.5 * (x + x.transpose(0, 2).conj())

    packed = PackedHermitianTensor(h)
    h2 = packed.unpack()

    assert h2.shape == h.shape
    assert torch.allclose(h2, h)


def test_roundtrip_minimal_shape():
    h = torch.tensor(
        [
            [
                [
                    2.0 + 0.0j,
                ]
            ]
        ]
    )  # shape (1, 1, 1)

    packed = PackedHermitianTensor(h)
    h2 = packed.unpack()

    assert packed.packed.shape == (1, 1)
    assert torch.allclose(h2, h)


def test_rejects_non_hermitian():
    h = torch.randn(3, 2, 3, dtype=torch.complex64)

    with pytest.raises(ValueError, match="not Hermitian"):
        PackedHermitianTensor(h)


def test_rejects_wrong_shape():
    h = torch.randn(3, 2, 4)

    with pytest.raises(ValueError, match="Expected shape"):
        PackedHermitianTensor(h)


def test_packed_shape():
    n, b = 4, 3
    x = torch.randn(n, b, n, dtype=torch.complex64)
    h = 0.5 * (x + x.transpose(0, 2).conj())

    packed = PackedHermitianTensor(h)

    assert packed.packed.shape == (b, n * (n + 1) // 2)


def test_roundtrip_real_symmetric():
    n, b = 4, 3
    x = torch.randn(n, b, n)
    h = 0.5 * (x + x.transpose(0, 2))

    packed = PackedHermitianTensor(h)
    h2 = packed.unpack()

    assert h2.shape == h.shape
    assert torch.allclose(h2, h)


def test_unpack_preserves_dtype():
    n, b = 4, 2
    x = torch.randn(n, b, n, dtype=torch.complex128)
    h = 0.5 * (x + x.transpose(0, 2).conj())

    packed = PackedHermitianTensor(h)
    h2 = packed.unpack()

    assert h2.dtype == h.dtype


def test_skip_hermitian_check():
    # check_hermitian=False is needed for hot paths: this test ensures the
    # class accepts structurally valid input without paying for symmetry checks.
    h = torch.randn(3, 2, 3, dtype=torch.complex64)

    packed = PackedHermitianTensor(h, check_hermitian=False)

    assert packed.packed.shape == (2, 6)


def test_packed_is_contiguous():
    n, b = 4, 3
    x = torch.randn(n, b, n, dtype=torch.complex64)
    h = 0.5 * (x + x.transpose(0, 2).conj())

    packed = PackedHermitianTensor(h)

    assert packed.packed.is_contiguous()


def test_custom_tolerance_controls_hermitian_check():
    n, b = 4, 2
    x = torch.randn(n, b, n, dtype=torch.complex64)
    h = 0.5 * (x + x.transpose(0, 2).conj())

    h_perturbed = h.clone()
    h_perturbed[0, 0, 1] += 1e-4

    with pytest.raises(ValueError, match="not Hermitian"):
        PackedHermitianTensor(h_perturbed, atol=1e-8)

    PackedHermitianTensor(h_perturbed, atol=1e-3)


def test_rejects_non_3d_input():
    h = torch.randn(3, 3)

    with pytest.raises(ValueError, match="Expected shape"):
        PackedHermitianTensor(h)
