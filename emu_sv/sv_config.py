import copy
import logging
import pathlib
import sys
from types import MethodType
from typing import Any

from emu_sv.custom_callback_implementations import (
    correlation_matrix_sv_impl,
    energy_second_moment_sv_impl,
    energy_variance_sv_impl,
    qubit_occupation_sv_impl,
)

from pulser.backend import (
    CorrelationMatrix,
    EmulationConfig,
    EnergySecondMoment,
    EnergyVariance,
    Occupation,
    StateResult,
)


class SVConfig(EmulationConfig):
    """
    The configuration of the emu-sv SVBackend. The kwargs passed to this class
    are passed on to the base class.
    See the API for that class for a list of available options.

    Args:
        dt: the timestep size that the solver uses. Note that observables are
            only calculated if the evaluation_times are divisible by dt.
        max_krylov_dim:
            the size of the krylov subspace that the Lanczos algorithm maximally builds
        krylov_tolerance:
            the Lanczos algorithm uses this as the convergence tolerance
        gpu: Use 1 gpu if True, and a GPU is available, otherwise, cpu.
            Will cause errors if True when a gpu is not available
        kwargs: arguments that are passed to the base class

    Examples:
        >>> gpu = True
        >>> dt = 1 #this will impact the runtime
        >>> krylov_tolerance = 1e-8 #the simulation will be faster, but less accurate
        >>> SVConfig(gpu=gpu, dt=dt, krylov_tolerance=krylov_tolerance,
        >>>     with_modulation=True) #the last arg is taken from the base class
    """

    def __init__(
        self,
        *,
        dt: int = 10,
        max_krylov_dim: int = 100,
        krylov_tolerance: float = 1e-10,
        gpu: bool = True,
        interaction_cutoff: float = 0.0,
        log_level: int = logging.INFO,
        log_file: pathlib.Path | None = None,
        **kwargs: Any,
    ):
        kwargs.setdefault("observables", [StateResult(evaluation_times=[1.0])])
        super().__init__(**kwargs)
        self._backend_options["dt"] = dt
        self._backend_options["max_krylov_dim"] = max_krylov_dim
        self._backend_options["gpu"] = gpu
        self._backend_options["krylov_tolerance"] = krylov_tolerance
        self._backend_options["interaction_cutoff"] = interaction_cutoff
        self._backend_options["log_level"] = log_level
        self._backend_options["log_file"] = log_file

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
            "max_krylov_dim",
            "krylov_tolerance",
            "gpu",
        }

    def monkeypatch_observables(self) -> None:
        obs_list = []
        for _, obs in enumerate(self.observables):  # monkey patch
            obs_copy = copy.deepcopy(obs)
            if isinstance(obs, Occupation):
                obs_copy.apply = MethodType(  # type: ignore[method-assign]
                    qubit_occupation_sv_impl, obs_copy
                )
            elif isinstance(obs, EnergyVariance):
                obs_copy.apply = MethodType(  # type: ignore[method-assign]
                    energy_variance_sv_impl, obs_copy
                )
            elif isinstance(obs, EnergySecondMoment):
                obs_copy.apply = MethodType(  # type: ignore[method-assign]
                    energy_second_moment_sv_impl, obs_copy
                )
            elif isinstance(obs, CorrelationMatrix):
                obs_copy.apply = MethodType(  # type: ignore[method-assign]
                    correlation_matrix_sv_impl, obs_copy
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
