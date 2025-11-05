import math

from emu_base.jump_lindblad_operators import get_lindblad_operators
from unittest.mock import MagicMock
from pulser import NoiseModel
import torch
import pytest
import numpy as np

dtype = torch.complex128
device = "cpu"


def test_get_lindblad_operators_unknown_noise():
    noise_model = MagicMock()
    noise_model.noise_types = ("depolarizing", "leakage", "SPAM")

    with pytest.raises(ValueError) as ve:
        get_lindblad_operators(noise_type="leakage", noise_model=noise_model)

    assert str(ve.value) == "Unknown noise type: leakage"


@pytest.mark.parametrize("interaction", ("ising", "XY"))
def test_get_lindblad_op_with_rydberg_basis(interaction):
    """This test solved a bug between XY and Rydberg bases when jump
    operators are created"""

    # pulser convention of basis
    if interaction == "ising":
        basis0 = torch.tensor([0.0, 1.0], device=device, dtype=dtype).reshape(2, 1)
        basis1 = torch.tensor([1.0, 0.0], device=device, dtype=dtype).reshape(2, 1)
    elif interaction == "XY":
        basis0 = torch.tensor([1.0, 0.0], device=device, dtype=dtype).reshape(2, 1)
        basis1 = torch.tensor([0.0, 1.0], device=device, dtype=dtype).reshape(2, 1)

    eff_rate = [0.5]
    eff_ops = [basis0 @ basis1.T]  # |0><1|

    noise_model = NoiseModel(eff_noise_rates=eff_rate, eff_noise_opers=eff_ops)

    expected_emu_mps_operator = math.sqrt(eff_rate[0]) * torch.tensor(
        [[0.0, 1.0], [0.0, 0.0]], dtype=dtype, device=basis0.device
    )

    emu_mps_lindblad = get_lindblad_operators(
        noise_type="eff_noise", noise_model=noise_model, interact_type=interaction
    )

    assert torch.allclose(emu_mps_lindblad[0], expected_emu_mps_operator)


def test_get_lindblad_op_with_invalid_eff_ops():
    # pulser convention of basis but with numpy
    basis0 = np.array([0.0, 1.0]).reshape(2, 1)
    basis1 = np.array([1.0, 0.0]).reshape(2, 1)

    eff_rate = [0.5]
    eff_ops = [basis0 @ basis1.T]  # numpy array, not a torch.Tensor

    noise_model = NoiseModel(eff_noise_rates=eff_rate, eff_noise_opers=eff_ops)

    with pytest.raises(
        ValueError, match="Only 2 by 2 or 3 by 3 effective noise operator matrices"
    ):
        get_lindblad_operators(noise_type="eff_noise", noise_model=noise_model)
