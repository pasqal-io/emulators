import pulser
import pytest

from emu_mps.mps_config import MPSConfig

import pulser.noise_model


def test_unsupported_noise():
    with pytest.raises(NotImplementedError) as exc:
        MPSConfig(
            noise_model=pulser.noise_model.NoiseModel(
                amp_sigma=0.1, laser_waist=5, runs=1, samples_per_run=1
            )
        )
    assert "Unsupported noise type(s): {'amplitude'}" in str(exc.value)

    MPSConfig(
        noise_model=pulser.noise_model.NoiseModel(
            runs=1,  # TODO: connect this with MCArlo
            samples_per_run=1,  # TODO: connect this with MCarlo or ignored
            state_prep_error=0.1,
            p_false_pos=0.0,
            p_false_neg=0.0,
        )
    )

    MPSConfig(
        noise_model=pulser.noise_model.NoiseModel(
            runs=1,
            samples_per_run=1,
            state_prep_error=0.1,
            p_false_pos=0.1,
            p_false_neg=0.15,
        )
    )
