import math

from emu_base.jump_lindblad_operators import get_lindblad_operators
from pulser import NoiseModel
import torch
import pytest
import numpy as np

dtype = torch.complex128
device = "cpu"


@pytest.mark.parametrize("interaction", ("ising", "XY"))
def test_get_lindblad_op_with_rydberg_basis(interaction):
    """This test solved a bug between XY and Rydberg bases when jump
    operators are created using pulser basis"""

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


def test_flipping_right_elements():
    """Flipping the right elements in a 3x3 lindblad operators when using
    Rydberg basis"""
    torch.manual_seed(42)
    n_atoms = 2
    dim = 3
    # assume jump operators given in pulser convention of basis
    eff_rate = [0.5] * n_atoms
    eff_ops0 = torch.rand(dim, dim, dtype=dtype, device=device)
    eff_ops1 = torch.rand(dim, dim, dtype=dtype, device=device)

    eff_ops = [eff_ops0, eff_ops1]

    noise_model = NoiseModel(eff_noise_rates=eff_rate, eff_noise_opers=eff_ops)

    emu_mps_lindblad = get_lindblad_operators(
        noise_type="eff_noise",
        noise_model=noise_model,
        interact_type="ising",
        dim=dim,
    )

    expected0 = (eff_ops0.clone().detach()) * math.sqrt(eff_rate[0])
    expected1 = (eff_ops1.clone().detach()) * math.sqrt(eff_rate[1])
    expected0[:2, :2] = torch.flip(  # flip for ising emu-mps basis
        expected0[:2, :2], (0, 1)
    )
    expected1[:2, :2] = torch.flip(  # flip for ising emu-mps basis
        expected1[:2, :2], (0, 1)
    )

    assert len(emu_mps_lindblad) == len(eff_ops)
    assert torch.allclose(emu_mps_lindblad[0], expected0)
    assert torch.allclose(emu_mps_lindblad[1], expected1)


def test_get_lindblad_op_with_wrong_dim():
    # pulser convention of basis but with numpy
    basis0 = np.array([0.0, 1.0, 0.0]).reshape(3, 1)
    basis1 = np.array([1.0, 0.0, 0.0]).reshape(3, 1)

    eff_rate = [0.5]
    eff_ops = [basis0 @ basis1.T]  # numpy array will raise an error

    noise_model = NoiseModel(eff_noise_rates=eff_rate, eff_noise_opers=eff_ops)

    with pytest.raises(ValueError, match="Only 2 by 2 effective noise operator matrices"):
        get_lindblad_operators(noise_type="eff_noise", noise_model=noise_model)
