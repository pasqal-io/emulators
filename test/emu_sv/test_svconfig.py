import pytest
import warnings
import re

from emu_sv import SVConfig
from pulser.backend.config import EmulationConfig, BackendConfig
from pulser.backend import BitStrings

# copypaste sv_config.py specific attributes
# attributes from SVConfig._expected_kwargs
sv_attributes = [
    "dt",
    "max_krylov_dim",
    "gpu",
    "krylov_tolerance",
]


def test_default_SVConfig_ctr() -> None:
    sv_config = SVConfig()
    assert len(sv_config.observables) == 1
    assert isinstance(sv_config.observables[0], BitStrings)
    assert sv_config.observables[0].evaluation_times == [
        1.0
    ]  # meaning very end of the simuation


def test_serialise_default_config() -> None:
    default_config = SVConfig()
    config_str = default_config.to_abstract_repr()

    deserialized_config = SVConfig.from_abstract_repr(config_str)

    for attr in sv_attributes:
        assert getattr(deserialized_config, attr) == getattr(
            default_config, attr
        ), f"{attr} mismatch"


def test_serialise_config() -> None:
    # Arguments are arbitrary just to be != default, no deep meaning behind
    dt = 1
    max_krylov_dim = 10
    gpu = False
    krylov_tolerance = 1e-5

    default_config = SVConfig(
        dt=dt,
        max_krylov_dim=max_krylov_dim,
        gpu=gpu,
        krylov_tolerance=krylov_tolerance,
    )

    config_str = default_config.to_abstract_repr()
    deserialized_config = SVConfig.from_abstract_repr(config_str)

    values = [dt, max_krylov_dim, gpu, krylov_tolerance]

    for attr, val in zip(sv_attributes, values):
        assert getattr(deserialized_config, attr) == getattr(
            default_config, attr
        ), f"{attr} mismatch"
        assert getattr(deserialized_config, attr) == val, f"{attr} != {val} mismatch"


def test_serialise_config_into_EmulationConfig() -> None:
    # This test is required for the cloud workflow
    # Arguments are arbitrary just to be != default, no deep meaning behind
    dt = 1
    max_krylov_dim = 10
    gpu = False
    krylov_tolerance = 1e-5
    observables = [BitStrings(evaluation_times=[1.0])]  # to avoid waring

    default_config = EmulationConfig(
        dt=dt,
        max_krylov_dim=max_krylov_dim,
        gpu=gpu,
        krylov_tolerance=krylov_tolerance,
        observables=observables,
    )

    config_str = default_config.to_abstract_repr()
    deserialized_config = SVConfig.from_abstract_repr(config_str)

    values = [dt, max_krylov_dim, gpu, krylov_tolerance, observables]

    for attr, val in zip(sv_attributes, values):
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

    # SVConfig
    msg = (
        "'SVConfig' received unexpected keyword arguments: "
        "{'blabla'}; only the following keyword arguments "
        "are expected:"
    )
    # re.escape() avoid interpreting message symbols "{}"", "()","."" as regex
    with pytest.warns(UserWarning, match="^" + re.escape(msg)):
        SVConfig(blabla=10)
