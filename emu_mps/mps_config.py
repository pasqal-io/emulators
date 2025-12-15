from typing import Any, ClassVar
from types import MethodType

import copy

from emu_base import init_logging
from emu_mps.mps import MPS, DEFAULT_MAX_BOND_DIM, DEFAULT_PRECISION
from emu_mps.mpo import MPO
from emu_mps.solver import Solver
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


class MPSConfig(EmulationConfig):
    """
    The configuration of the emu-mps MPSBackend. The kwargs passed to this
    class are passed on to the base class.
    See the API for that class for a list of available options.

    Args:
        dt: The timestep size that the solver uses. Note that observables are
            only calculated if the evaluation_times are divisible by dt.
        precision: Up to what precision the state is truncated.
            Defaults to `1e-5`.
        max_bond_dim: The maximum bond dimension that the state is allowed
            to have.
            Defaults to `1024`.
        max_krylov_dim:
            The size of the krylov subspace that the Lanczos algorithm
            maximally builds
        extra_krylov_tolerance:
            The Lanczos algorithm uses this*precision as the convergence
            tolerance
        num_gpus_to_use: number of GPUs to be used in a given simulation.
            - if it is set to a number `n > 0`, the state will be distributed
                across `n` GPUs.
            - if it is set to `n = 0`, the entire simulation runs on the CPU.
            - if it is `None` (the default value), the backend internally
                chooses the number of GPUs
            based on the hardware availability during runtime.
        As shown in the benchmarks, using multiple GPUs might
            alleviate memory pressure per GPU, but the runtime should
            be similar.
        optimize_qubit_ordering: Optimize the register ordering. Improves
            performance and accuracy, but disables certain features.
        interaction_cutoff: Set interaction coefficients Uᵢⱼ below this value
            to `0.0`. Potentially improves runtime and memory consumption.
        log_level: How much to log. Set to `logging.WARN` to get rid of the
            timestep info.
        log_file: If specified, log to this file rather than stout.
        autosave_prefix: filename prefix for autosaving simulation state to
            file
        autosave_dt: Minimum time interval in seconds between two autosaves.
            Saving the simulation state is only possible at specific times,
            therefore this interval is only a lower bound.
        solver: Chooses the solver algorithm to run a sequence.
            Two options are currently available:
            ``TDVP``, which performs ordinary time evolution,
            and ``DMRG``, which adiabatically follows the ground state
            of a given adiabatic pulse.
        kwargs: Arguments that are passed to the base class

    Examples:
        >>> num_gpus_to_use = 2 #use 2 gpus if available, otherwise 1 or cpu
        >>> dt = 1 #this will impact the runtime
        >>> precision = 1e-6 #smaller dt requires better precision, generally
        >>> MPSConfig(num_gpus_to_use=num_gpus_to_use, dt=dt, precision=precision,
        >>>     with_modulation=True) #the last arg is taken from the base class
    """

    # Whether to warn if unexpected kwargs are received
    _enforce_expected_kwargs: ClassVar[bool] = True
    _state_type = MPS
    _operator_type = MPO

    def __init__(
        self,
        *,
        dt: int = 10,
        precision: float = DEFAULT_PRECISION,
        max_bond_dim: int = DEFAULT_MAX_BOND_DIM,
        max_krylov_dim: int = 100,
        extra_krylov_tolerance: float = 1e-3,
        num_gpus_to_use: int | None = None,
        optimize_qubit_ordering: bool = True,
        interaction_cutoff: float = 0.0,
        log_level: int = logging.INFO,
        log_file: pathlib.Path | None = None,
        autosave_prefix: str = "emu_mps_save_",
        autosave_dt: int | float = float("inf"),  # disable autosave by default
        solver: Solver = Solver.TDVP,
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
            optimize_qubit_ordering=optimize_qubit_ordering,
            interaction_cutoff=interaction_cutoff,
            log_level=log_level,
            log_file=log_file,
            autosave_prefix=autosave_prefix,
            autosave_dt=autosave_dt,
            solver=solver,
            **kwargs,
        )
        self.logger = init_logging(log_level, log_file)

        MIN_AUTOSAVE_DT = 10
        assert (
            self.autosave_dt > MIN_AUTOSAVE_DT
        ), f"autosave_dt must be larger than {MIN_AUTOSAVE_DT} seconds"

        MIN_KRYLOV_TOL = 1.0e-12  # keep numerical stability
        prod_tol = precision * extra_krylov_tolerance
        if prod_tol < MIN_KRYLOV_TOL:
            new_extra_krylov_tolerance = MIN_KRYLOV_TOL / precision
            self.logger.warning(
                f"Requested Lanczos convergence tolerance "
                f"(precision * extra_krylov_tolerance = {prod_tol:.2e}) "
                f"is below minimum threshold {MIN_KRYLOV_TOL:.2e}. "
                f"Automatically adjusting extra_krylov_tolerance from "
                f"{extra_krylov_tolerance:.2e} to {new_extra_krylov_tolerance:.2e} "
                f"to maintain numerical stability."
            )
            self.extra_krylov_tolerance = new_extra_krylov_tolerance
        else:
            self.extra_krylov_tolerance = extra_krylov_tolerance

        self.monkeypatch_observables()

        if (self.noise_model.runs != 1 and self.noise_model.runs is not None) or (
            self.noise_model.samples_per_run != 1
            and self.noise_model.samples_per_run is not None
        ):
            self.logger.warning(
                "Warning: The runs and samples_per_run values of the NoiseModel are ignored!"
            )
        self._backend_options[
            "optimize_qubit_ordering"
        ] &= self.check_permutable_observables()

    def _expected_kwargs(self) -> set[str]:
        return super()._expected_kwargs() | {
            "dt",
            "precision",
            "max_bond_dim",
            "max_krylov_dim",
            "extra_krylov_tolerance",
            "num_gpus_to_use",
            "optimize_qubit_ordering",
            "interaction_cutoff",
            "log_level",
            "log_file",
            "autosave_prefix",
            "autosave_dt",
            "solver",
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

    def check_permutable_observables(self) -> bool:
        allowed_permutable_obs = set(
            [
                "bitstrings",
                "occupation",
                "correlation_matrix",
                "statistics",
                "energy",
                "energy_variance",
                "energy_second_moment",
            ]
        )

        actual_obs = set([obs._base_tag for obs in self.observables])
        not_allowed = actual_obs.difference(allowed_permutable_obs)
        if not_allowed:
            self.logger.warning(
                f"emu-mps allows only {allowed_permutable_obs} observables with"
                " `optimize_qubit_ordering = True`."
                f" you provided unsupported {not_allowed}"
                " using `optimize_qubit_ordering = False` instead."
            )
        return not_allowed == set()
