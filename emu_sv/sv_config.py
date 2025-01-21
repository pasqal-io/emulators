from emu_base.base_classes import (
    CorrelationMatrix,
    QubitDensity,
    Energy,
    EnergyVariance,
    SecondMomentOfEnergy,
)

import copy


from emu_base import BackendConfig
from emu_sv import StateVector
from typing import Any

from emu_sv.custom_callback_implementations import (
    custom_qubit_density,
    custom_energy,
    custom_energy_variance,
    custom_second_momentum_energy,
    custom_correlation_matrix,
)

from types import MethodType


class SVConfig(BackendConfig):
    """
    The configuration of the emu-sv SVBackend. The kwargs passed to this class
    are passed on to the base class.
    See the API for that class for a list of available options.

    Args:
        initial_state: the initial state to use in the simulation
        dt: the timestep size that the solver uses. Note that observables are
            only calculated if the evaluation_times are divisible by dt.
        max_krylov_dim:
            the size of the krylov subspace that the Lanczos algorithm maximally builds
        krylov_tolerance:
            the Lanczos algorithm uses this as the convergence tolerance
        gpu: Use 1 gpu if True, otherwise, cpu.
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
        initial_state: StateVector | None = None,
        dt: int = 10,
        max_krylov_dim: int = 100,
        krylov_tolerance: float = 1e-12,
        gpu: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.initial_state = initial_state
        self.dt = dt
        self.max_krylov_dim = max_krylov_dim
        self.gpu = gpu
        self.krylov_tolerance = krylov_tolerance

        for num, obs in enumerate(self.callbacks):  # monkey patch
            obs_copy = copy.deepcopy(obs)
            if isinstance(obs, QubitDensity):
                obs_copy.apply = MethodType(custom_qubit_density, obs)  # type: ignore[method-assign]
                self.callbacks[num] = obs_copy
            elif isinstance(obs, Energy):
                obs_copy.apply = MethodType(custom_energy, obs)  # type: ignore[method-assign]
                self.callbacks[num] = obs_copy
            elif isinstance(obs, EnergyVariance):
                obs_copy.apply = MethodType(custom_energy_variance, obs)  # type: ignore[method-assign]
                self.callbacks[num] = obs_copy
            elif isinstance(obs, SecondMomentOfEnergy):
                obs_copy.apply = MethodType(custom_second_momentum_energy, obs)  # type: ignore[method-assign]
                self.callbacks[num] = obs_copy
            elif isinstance(obs, CorrelationMatrix):
                obs_copy.apply = MethodType(custom_correlation_matrix, obs)  # type: ignore[method-assign]
                self.callbacks[num] = obs_copy
