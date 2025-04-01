import pulser
import pytest

from emu_mps.mps_config import MPSConfig

import pulser.noise_model

# copypaste mps_config.py specific attributes
# attributes from MPSConfig._expected_kwargs
mps_attributes = [
    "dt",
    "precision",
    "max_bond_dim",
    "max_krylov_dim",
    "extra_krylov_tolerance",
    "num_gpus_to_use",
]


def test_unsupported_noise() -> None:
    with pytest.raises(NotImplementedError) as exc:
        MPSConfig(
            noise_model=pulser.noise_model.NoiseModel(
                amp_sigma=0.1, laser_waist=5, runs=1, samples_per_run=1
            )
        )
    assert "Unsupported noise type: amp_sigma" in str(exc.value)

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

    pass


def test_serialise_default_config() -> None:
    default_config = MPSConfig()
    config_str = default_config.to_abstract_repr()

    deserialized_config = MPSConfig.from_abstract_repr(config_str)

    for attr in mps_attributes:
        assert getattr(deserialized_config, attr) == getattr(
            default_config, attr
        ), f"{attr} mismatch"


def test_serialise_config() -> None:
    # Arguments are arbitrary just to be != default, no deep meaning behind
    dt = 1
    precision = 1e-2
    max_bond_dim = 10
    max_krylov_dim = 15
    extra_krylov_tolerance = 1e-1
    num_gpus_to_use = 0

    default_config = MPSConfig(
        dt=dt,
        precision=precision,
        max_bond_dim=max_bond_dim,
        max_krylov_dim=max_krylov_dim,
        extra_krylov_tolerance=extra_krylov_tolerance,
        num_gpus_to_use=num_gpus_to_use,
    )

    config_str = default_config.to_abstract_repr()
    deserialized_config = MPSConfig.from_abstract_repr(config_str)

    values = [
        dt,
        precision,
        max_bond_dim,
        max_krylov_dim,
        extra_krylov_tolerance,
        num_gpus_to_use,
    ]

    for attr, val in zip(mps_attributes, values):
        assert getattr(deserialized_config, attr) == getattr(
            default_config, attr
        ), f"{attr} mismatch"
        assert getattr(deserialized_config, attr) == val, f"{attr} != {val} mismatch"
