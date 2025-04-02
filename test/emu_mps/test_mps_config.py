import pytest
import warnings
import re
import logging

import pulser

from emu_mps.mps_config import MPSConfig
from pulser.backend.config import EmulationConfig, BackendConfig
from pulser.backend import BitStrings

import pulser.noise_model

# copypaste mps_config.py specific attributes
# attributes from MPSConfig._expected_kwargs
# Arguments are arbitrary just to be != default, no deep meaning behind
mps_params = {
    "dt": 1,
    "precision": 1e-2,
    "max_bond_dim": 10,
    "max_krylov_dim": 15,
    "extra_krylov_tolerance": 1e-1,
    "num_gpus_to_use": 0,
    "interaction_cutoff": 1.1,
    "log_level": logging.ERROR,
    "log_file": None,
    "autosave_prefix": "my_file",
    "autosave_dt": 15,
}


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


def test_default_config_repr() -> None:
    default_config = MPSConfig()
    config_str = default_config.to_abstract_repr()

    deserialized_config = MPSConfig.from_abstract_repr(config_str)

    for attr, _ in mps_params.items():
        assert getattr(deserialized_config, attr) == getattr(
            default_config, attr
        ), f"{attr} mismatch"


@pytest.mark.parametrize(
    "config_var",
    [
        (EmulationConfig),
        (MPSConfig),
    ],
)
def test_config_repr(config_var: EmulationConfig | MPSConfig) -> None:
    # This test is required for the cloud workflow
    observables = [BitStrings(evaluation_times=[1.0])]  # to avoid waring
    default_config = config_var(
        **mps_params,
        observables=observables,
    )

    config_str = default_config.to_abstract_repr()
    deserialized_config = MPSConfig.from_abstract_repr(config_str)

    for attr, val in mps_params.items():
        assert getattr(deserialized_config, attr) == getattr(
            default_config, attr
        ), f"{attr} mismatch"
        assert getattr(deserialized_config, attr) == val, f"{attr} != {val} mismatch"


def test_default_constructors_for_all_config() -> None:
    # BackendConfig
    msg = (
        "'BackendConfig' received unexpected keyword arguments: "
        "{'blabla'}; only the following keyword arguments "
        "are expected: set(). "
    )
    # ^ and $ are for full regex match
    # re.escape() avoid interpreting message symbols "{}"", "()","."" as regex
    with pytest.warns(UserWarning, match="^" + re.escape(msg) + "$"):
        BackendConfig(blabla=10)

    # EmulationConfig No warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        EmulationConfig(blabla=10, observables=[BitStrings(evaluation_times=[1.0])])
        assert not w, "Unexpected warnings: No warning"

    # MPSConfig
    msg = (
        "'MPSConfig' received unexpected keyword arguments: "
        "{'blabla'}; only the following keyword arguments "
        "are expected:"
    )
    # re.escape() avoid interpreting message symbols "{}"", "()","."" as regex
    with pytest.warns(UserWarning, match="^" + re.escape(msg)):
        MPSConfig(blabla=10)
