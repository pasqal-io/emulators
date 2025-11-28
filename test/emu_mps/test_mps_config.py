import pytest
import re
import logging
import torch
import pulser
import numpy as np
from emu_mps.mps_config import MPSConfig, MPS
from pulser.backend.config import EmulationConfig
from pulser.backend import BitStrings, StateResult
from emu_mps.observables import EntanglementEntropy
import json

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
    MPSConfig(
        noise_model=pulser.noise_model.NoiseModel(
            runs=1,  # TODO: connect this with MCArlo
            samples_per_run=1,  # TODO: connect this with MCarlo or ignored
            state_prep_error=0.1,
            p_false_pos=0.1,
            p_false_neg=0.15,
            laser_waist=5,
            amp_sigma=0.1,
            temperature=10.0,
            with_leakage=True,
            eff_noise_rates=(0.1,),
            eff_noise_opers=(torch.randn(3, 3, dtype=torch.float64),),
            hyperfine_dephasing_rate=1.5,
        )
    )


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
    # MPSConfig
    msg = (
        "'MPSConfig' received unexpected keyword arguments: "
        "{'blabla'}; only the following keyword arguments "
        "are expected:"
    )
    # re.escape() avoid interpreting message symbols "{}"", "()","."" as regex
    with pytest.raises(ValueError, match="^" + re.escape(msg)):
        MPSConfig(blabla=10)


def test_optimize_qubit_ordering_unsupported_observables(capsys) -> None:
    eval_times = [1.0]
    state_res = StateResult(evaluation_times=eval_times)
    entr = EntanglementEntropy(mps_site=0, evaluation_times=eval_times)

    config = MPSConfig(
        optimize_qubit_ordering=True,
    )

    out, _ = capsys.readouterr()

    assert "using `optimize_qubit_ordering = False` instead." not in out
    assert config.optimize_qubit_ordering

    config = MPSConfig(
        observables=[state_res, entr],
        optimize_qubit_ordering=True,
    )

    out, _ = capsys.readouterr()
    assert "using `optimize_qubit_ordering = False` instead." in out
    assert not config.optimize_qubit_ordering


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
    state = MPS.from_state_amplitudes(eigenstates=("r", "g"), amplitudes=amp_full)

    config = MPSConfig(initial_state=state)

    config_str = config.to_abstract_repr()
    my_config = json.loads(config_str)

    assert my_config["initial_state"]["eigenstates"] == basis
    assert my_config["initial_state"]["amplitudes"] == amp_full
