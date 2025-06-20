import pytest
import re
import logging
import torch

from emu_sv import SVConfig
from pulser.backend.config import EmulationConfig
from pulser.backend import BitStrings
from pulser import NoiseModel

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


def test_unsupported_noise() -> None:
    SVConfig(
        noise_model=NoiseModel(
            runs=1,  # TODO: connect this with MCArlo
            samples_per_run=1,  # TODO: connect this with MCarlo or ignored
            laser_waist=5,
            amp_sigma=0.1,
            temperature=10.0,
            with_leakage=True,
            eff_noise_rates=(0.1,),
            eff_noise_opers=(torch.randn(3, 3, dtype=torch.float64),),
            hyperfine_dephasing_rate=1.5,
        )
    )

    with pytest.raises(NotImplementedError) as exc:
        SVConfig(
            noise_model=NoiseModel(
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
    assert str(exc.value) == "State preparation errors are currently not supported in emu-sv."


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
