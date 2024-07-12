from emu_ct.mps_config import MPSConfig
import pytest
import pulser


def test_unsupported_noise():
    with pytest.raises(NotImplementedError) as exc:
        MPSConfig(noise_model=pulser.noise_model.NoiseModel(noise_types=("amplitude",)))
    assert "Unsupported noise type(s): {'amplitude'}" in str(exc.value)

    with pytest.raises(NotImplementedError) as exc:
        # This throws because there are non-zero default values for p_false_pos and p_false_neg.
        MPSConfig(
            noise_model=pulser.noise_model.NoiseModel(
                noise_types=("SPAM",), state_prep_error=0.1
            )
        )
    assert "Unsupported: measurement errors" in str(exc.value)

    MPSConfig(
        noise_model=pulser.noise_model.NoiseModel(
            noise_types=("SPAM",), state_prep_error=0.1, p_false_pos=0.0, p_false_neg=0.0
        )
    )
