from typing import Any, ClassVar
from types import MethodType

import copy

from emu_base import DEVICE_COUNT
from emu_mps.custom_callback_implementations import (
    energy_mps_impl,
    energy_second_moment_mps_impl,
    energy_variance_mps_impl,
    correlation_matrix_mps_impl,
    qubit_occupation_mps_impl,
)
from pulser.backend import (
    Occupation,
    CorrelationMatrix,
    Energy,
    EnergySecondMoment,
    EnergyVariance,
    BitStrings,
    EmulationConfig,
)
import logging
import pathlib
import sys


class MPSConfig(EmulationConfig):
    """
    The configuration of the emu-mps MPSBackend. The kwargs passed to this class
    are passed on to the base class.
    See the API for that class for a list of available options.

    Args:
        dt: the timestep size that the solver uses. Note that observables are
            only calculated if the evaluation_times are divisible by dt.
        precision: up to what precision the state is truncated
        max_bond_dim: the maximum bond dimension that the state is allowed to have.
        max_krylov_dim:
            the size of the krylov subspace that the Lanczos algorithm maximally builds
        extra_krylov_tolerance:
            the Lanczos algorithm uses this*precision as the convergence tolerance
        num_gpus_to_use: during the simulation, distribute the state over this many GPUs
            0=all factors to cpu. As shown in the benchmarks, using multiple GPUs might
            alleviate memory pressure per GPU, but the runtime should be similar.
        autosave_prefix: filename prefix for autosaving simulation state to file
        autosave_dt: minimum time interval in seconds between two autosaves
            Saving the simulation state is only possible at specific times,
            therefore this interval is only a lower bound.
        kwargs: arguments that are passed to the base class

    Examples:
        >>> num_gpus_to_use = 2 #use 2 gpus if available, otherwise 1 or cpu
        >>> dt = 1 #this will impact the runtime
        >>> precision = 1e-6 #smaller dt requires better precision, generally
        >>> MPSConfig(num_gpus_to_use=num_gpus_to_use, dt=dt, precision=precision,
        >>>     with_modulation=True) #the last arg is taken from the base class
    """

    # Whether to warn if unexpected kwargs are received
    _enforce_expected_kwargs: ClassVar[bool] = True

    def __init__(
        self,
        *,
        dt: int = 10,
        precision: float = 1e-5,
        max_bond_dim: int = 1024,
        max_krylov_dim: int = 100,
        extra_krylov_tolerance: float = 1e-3,
        num_gpus_to_use: int = DEVICE_COUNT,
        interaction_cutoff: float = 0.0,
        log_level: int = logging.INFO,
        log_file: pathlib.Path | None = None,
        autosave_prefix: str = "emu_mps_save_",
        autosave_dt: int = 600,  # 10 minutes
        **kwargs: Any,
    ):
        kwargs.setdefault("observables", [BitStrings(evaluation_times=[1.0])])
        super().__init__(
            dt=dt,
            precision=precision,
            max_bond_dim=max_bond_dim,
            max_krylov_dim=max_krylov_dim,
            extra_krylov_tolerance=extra_krylov_tolerance,
            num_gpus_to_use=num_gpus_to_use,
            interaction_cutoff=interaction_cutoff,
            log_level=log_level,
            log_file=log_file,
            autosave_prefix=autosave_prefix,
            autosave_dt=autosave_dt,
            **kwargs,
        )

        if "doppler" in self.noise_model.noise_types:
            raise NotImplementedError("Unsupported noise type: doppler")
        if (
            "amplitude" in self.noise_model.noise_types
            and self.noise_model.amp_sigma != 0.0
        ):
            raise NotImplementedError("Unsupported noise type: amp_sigma")

        MIN_AUTOSAVE_DT = 10
        assert (
            self.autosave_dt > MIN_AUTOSAVE_DT
        ), f"autosave_dt must be larger than {MIN_AUTOSAVE_DT} seconds"

        self.monkeypatch_observables()

        self.logger = logging.getLogger("global_logger")
        if log_file is None:
            logging.basicConfig(
                level=log_level, format="%(message)s", stream=sys.stdout, force=True
            )  # default to stream = sys.stderr
        else:
            logging.basicConfig(
                level=log_level,
                format="%(message)s",
                filename=str(log_file),
                filemode="w",
                force=True,
            )
        if (self.noise_model.runs != 1 and self.noise_model.runs is not None) or (
            self.noise_model.samples_per_run != 1
            and self.noise_model.samples_per_run is not None
        ):
            self.logger.warning(
                "Warning: The runs and samples_per_run values of the NoiseModel are ignored!"
            )

    def _expected_kwargs(self) -> set[str]:
        return super()._expected_kwargs() | {
            "dt",
            "precision",
            "max_bond_dim",
            "max_krylov_dim",
            "extra_krylov_tolerance",
            "num_gpus_to_use",
            "interaction_cutoff",
            "log_level",
            "log_file",
            "autosave_prefix",
            "autosave_dt",
        }

    def monkeypatch_observables(self) -> None:
        obs_list = []
        for _, obs in enumerate(self.observables):  # monkey patch
            obs_copy = copy.deepcopy(obs)
            if isinstance(obs, Occupation):
                obs_copy.apply = MethodType(  # type: ignore[method-assign]
                    qubit_occupation_mps_impl, obs_copy
                )
            elif isinstance(obs, EnergyVariance):
                obs_copy.apply = MethodType(  # type: ignore[method-assign]
                    energy_variance_mps_impl, obs_copy
                )
            elif isinstance(obs, EnergySecondMoment):
                obs_copy.apply = MethodType(  # type: ignore[method-assign]
                    energy_second_moment_mps_impl, obs_copy
                )
            elif isinstance(obs, CorrelationMatrix):
                obs_copy.apply = MethodType(  # type: ignore[method-assign]
                    correlation_matrix_mps_impl, obs_copy
                )
            elif isinstance(obs, Energy):
                obs_copy.apply = MethodType(  # type: ignore[method-assign]
                    energy_mps_impl, obs_copy
                )
            obs_list.append(obs_copy)
        self.observables = tuple(obs_list)

    def init_logging(self) -> None:
        if self.log_file is None:
            logging.basicConfig(
                level=self.log_level, format="%(message)s", stream=sys.stdout, force=True
            )  # default to stream = sys.stderr
        else:
            logging.basicConfig(
                level=self.log_level,
                format="%(message)s",
                filename=str(self.log_file),
                filemode="w",
                force=True,
            )
