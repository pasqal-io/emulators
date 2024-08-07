import pulser
import pytest

from emu_mps.mps_config import MPSConfig


def test_unsupported_noise():
    with pytest.raises(NotImplementedError) as exc:
        MPSConfig(noise_model=pulser.noise_model.NoiseModel(noise_types=("amplitude",)))
    assert "Unsupported noise type(s): {'amplitude'}" in str(exc.value)

    MPSConfig(
        noise_model=pulser.noise_model.NoiseModel(
            noise_types=("SPAM",), state_prep_error=0.1, p_false_pos=0.0, p_false_neg=0.0
        )
    )

    MPSConfig(
        noise_model=pulser.noise_model.NoiseModel(
            noise_types=("SPAM",), state_prep_error=0.1, p_false_pos=0.1, p_false_neg=0.15
        )
    )
