import pytest
import re
import json
import logging
from emu_sv import SVConfig, StateVector
import pulser
import numpy as np
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
    assert len(sv_config.observables) == 0
    assert len(sv_config.callbacks) == 0


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


def test_serialization_with_state():
    natoms = 2
    reg = pulser.Register.rectangle(1, natoms, spacing=8.0, prefix="q")

    seq = pulser.Sequence(reg, pulser.MockDevice)
    seq.declare_channel("ch0", "rydberg_global")
    duration = 500
    pulse = pulser.Pulse.ConstantPulse(duration, 4 * np.pi, 0.0, 0.0)
    seq.add(pulse, "ch0")
    basis = ["r", "g"]
    amp_full = {"g" * natoms: (0.7071 + 0.0j), "r" * natoms: (0.7071 + 0.0j)}
    state = StateVector.from_state_amplitudes(eigenstates=basis, amplitudes=amp_full)

    config = SVConfig(initial_state=state)
    config_str = config.to_abstract_repr()

    my_config = json.loads(config_str)

    assert my_config["initial_state"]["eigenstates"] == basis
    assert my_config["initial_state"]["amplitudes"] == amp_full
