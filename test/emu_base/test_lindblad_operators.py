from emu_base.jump_lindblad_operators import get_lindblad_operators
from unittest.mock import MagicMock
import pytest


def test_get_lindblad_operators_unknown_noise():
    noise_model = MagicMock()
    noise_model.noise_types = ("depolarizing", "leakage", "SPAM")

    with pytest.raises(ValueError) as ve:
        get_lindblad_operators(noise_type="leakage", noise_model=noise_model)

    assert str(ve.value) == "Unknown noise type: leakage"
