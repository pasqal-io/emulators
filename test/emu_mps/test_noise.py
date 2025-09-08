from unittest.mock import patch

import pytest
import torch

from emu_base.noise import pick_dark_qubits
from emu_base.jump_lindblad_operators import compute_noise_from_lindbladians


def test_compute_noise_from_lindbladians_empty():
    assert torch.allclose(
        compute_noise_from_lindbladians([]), torch.zeros(2, 2, dtype=torch.complex128)
    )


def test_compute_noise_from_lindbladians_wrong_shape():
    with pytest.raises(AssertionError) as exception_info:
        compute_noise_from_lindbladians(
            [torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.complex128)]
        )

    assert "Only single-qubit lindblad operators are supported" in str(
        exception_info.value
    )


def test_compute_noise_from_lindbladians():
    assert torch.allclose(
        compute_noise_from_lindbladians(
            [
                torch.tensor([[0.0, 0.0], [1.0j, 2.0]], dtype=torch.complex128),
                torch.tensor([[3.0, 0.0], [0.0, 0.0]], dtype=torch.complex128),
            ]
        ),
        torch.tensor([[-5.0j, -1.0], [1.0, -2.0j]], dtype=torch.complex128),
    )


@patch("emu_base.noise.torch.rand")
def test_pick_dark_qubits(random_mock):
    random_mock.return_value = torch.tensor([0.1, 0.8, 0.2, 0.4, 0.5])
    expected = torch.tensor([True, False, True, False, False], dtype=torch.bool)
    assert torch.equal(pick_dark_qubits(eta=0.25, n=5), expected)
