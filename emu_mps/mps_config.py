from typing import Any

from emu_base import BackendConfig, State, DEVICE_COUNT


class MPSConfig(BackendConfig):
    """
    The configuration of the emu-ct MPSBackend. The kwargs passed to this class
    are passed on to the base class.
    See the API for that class for a list of available options.

    Args:
        initial_state: the initial state to use in the simulation
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

    def __init__(
        self,
        *,
        initial_state: State | None = None,
        dt: int = 10,
        precision: float = 1e-5,
        max_bond_dim: int = 1024,
        max_krylov_dim: int = 100,
        extra_krylov_tolerance: float = 1e-3,
        num_gpus_to_use: int = DEVICE_COUNT,
        autosave_prefix: str = "emu_mps_save_",
        autosave_dt: int = 600,  # 10 minutes
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.initial_state = initial_state
        self.dt = dt
        self.precision = precision
        self.max_bond_dim = max_bond_dim
        self.max_krylov_dim = max_krylov_dim
        self.num_gpus_to_use = num_gpus_to_use
        self.extra_krylov_tolerance = extra_krylov_tolerance

        if self.noise_model is not None:
            if "doppler" in self.noise_model.noise_types:
                raise NotImplementedError("Unsupported noise type: doppler")
            if (
                "amplitude" in self.noise_model.noise_types
                and self.noise_model.amp_sigma != 0.0
            ):
                raise NotImplementedError("Unsupported noise type: amp_sigma")

        self.autosave_prefix = autosave_prefix
        self.autosave_dt = autosave_dt

        MIN_AUTOSAVE_DT = 10
        assert (
            self.autosave_dt > MIN_AUTOSAVE_DT
        ), f"autosave_dt must be larger than {MIN_AUTOSAVE_DT} seconds"
