import pytest
import re
import logging


from emu_sv import SVConfig
from pulser.backend.config import EmulationConfig
from pulser.backend import BitStrings


# copypaste sv_config.py specific attributes
# attributes from SVConfig._expected_kwargs
# Arguments are arbitrary just to be != default, no deep meaning behind
sv_params = {
    "dt": 1,
    "max_krylov_dim": 15,
    "krylov_tolerance": 1e-1,
    "gpu": False,
    "interaction_cutoff": 1.1,
    "log_level": logging.ERROR,
    "log_file": None,
}


def test_default_SVConfig_ctr() -> None:
    sv_config = SVConfig()
    assert len(sv_config.observables) == 1
    assert isinstance(sv_config.observables[0], BitStrings)
    assert sv_config.observables[0].evaluation_times == [
        1.0
    ]  # meaning very end of the simuation


def test_default_config_repr() -> None:
    default_config = SVConfig()
    config_str = default_config.to_abstract_repr()
    deserialized_config = SVConfig.from_abstract_repr(config_str)

    for attr, _ in sv_params.items():
        assert getattr(deserialized_config, attr) == getattr(
            default_config, attr
        ), f"{attr} mismatch"


@pytest.mark.parametrize(
    "config_var",
    [
        (EmulationConfig),
        (SVConfig),
    ],
)
def test_config_repr(config_var: EmulationConfig | SVConfig) -> None:
    # This test is required for the cloud workflow
    observables = [BitStrings(evaluation_times=[1.0])]  # to avoid waring
    default_config = config_var(
        **sv_params,
        observables=observables,
    )

    config_str = default_config.to_abstract_repr()
    deserialized_config = SVConfig.from_abstract_repr(config_str)

    for attr, val in sv_params.items():
        assert getattr(deserialized_config, attr) == getattr(
            default_config, attr
        ), f"{attr} mismatch"
        assert getattr(deserialized_config, attr) == val, f"{attr} != {val} mismatch"


def test_expected_kwargs() -> None:
    msg = (
        "'SVConfig' received unexpected keyword arguments: "
        "{'blabla'}; only the following keyword arguments "
        "are expected:"
    )
    # re.escape() avoid interpreting message symbols "{}"", "()","."" as regex
    with pytest.raises(ValueError, match="^" + re.escape(msg)):
        SVConfig(blabla=10)
