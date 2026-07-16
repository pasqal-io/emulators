import torch
from emu_sv.algebra import expect_batch, apply


def test_expect_batch():
    vec = torch.tensor([0.0, 0.0, 0.5, 1.0], dtype=torch.complex128)
    n = torch.tensor(
        [[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]]], dtype=torch.complex128
    )

    assert torch.allclose(
        expect_batch(vec, n, 2),
        torch.tensor([[1.25, 0.0], [1.0, 0.25]], dtype=torch.complex128),
    )


def test_apply():
    had = torch.tensor([[1.0, -1.0], [1.0, 1.0]], dtype=torch.complex128)
    vec = torch.tensor([0.0, 0.0, 0.5, 1.0], dtype=torch.complex128)

    assert torch.allclose(
        apply(vec, 0, had), torch.tensor([-0.5, -1.0, 0.5, 1.0], dtype=torch.complex128)
    )
    assert torch.allclose(
        apply(vec, 1, had), torch.tensor([0.0, 0.0, -0.5, 1.5], dtype=torch.complex128)
    )
