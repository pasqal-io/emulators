from emu_base import matmul_2x2_with_batched
import torch


def test_matmul_2x2_with_batched():
    left = torch.randn(2, 2, dtype=torch.complex128)
    right = torch.randn(10, 2, 5, dtype=torch.complex128)

    assert torch.allclose(left @ right, matmul_2x2_with_batched(left, right))
