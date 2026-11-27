import time
import typing
import torch
import logging

from emu_sv.hamiltonians import (
    FusedRydbergHamiltonian,
    RydbergHamiltonian,
    RydbergLindbladian,
)

from pulser.backend import Results, Observable, State, EmulationConfig, AggregationMethod
from emu_base import (
    SequenceData,
    get_max_rss_cpu,
    get_max_rss_gpu,
    is_evaluation_time,
)

from emu_sv.enums import Solver, Exponentiator
from emu_sv.state_vector import StateVector
from emu_sv.density_matrix_state import DensityMatrix
from emu_sv.dense_state import DenseState
from emu_sv.sv_config import SVConfig
from emu_sv.time_evolution import (
    EvolveStateVector,
    EvolveDensityMatrix,
    EvolveMonteCarlo,
    BaseStepper,
    Exponentiation,
    Lanczos,
    Chebyshev,
)
from emu_sv.time_evolution.steppers import JUMP_TIME_TOLERANCE_FRACTION

_TIME_CONVERSION_COEFF = 0.001  # Omega and delta are given in rad/μs, dt in ns


class Statistics(Observable):
    def __init__(
        self,
        evaluation_times: typing.Sequence[float] | None,
        data: list[float],
        timestep_count: int,
    ):
        super().__init__(
            evaluation_times=evaluation_times,
            default_aggregation_method=AggregationMethod.SKIP,
        )
        self.data = data
        self.timestep_count = timestep_count

    @property
    def _base_tag(self) -> str:
        return "statistics"

    def __call__(
        self,
        config: EmulationConfig,
        t: float,
        state: State,
        hamiltonian: typing.Any,
        result: Results,
    ) -> None:
        """
        Store the statistics at time t. Statistics are taken at every time step, so
        unlike `Observable.__call__`, this does not look t up in the evaluation
        times, which would take O(number of steps) at each step.
        """
        value = self.apply(config=config, state=state, hamiltonian=hamiltonian)
        result._store(observable=self, time=t, value=value)

    def apply(
        self,
        *,
        config: EmulationConfig,
        state: State,
        **kwargs: typing.Any,
    ) -> dict:
        """Calculates the observable to store in the Results."""
        assert isinstance(state, StateVector | DensityMatrix)
        assert isinstance(config, SVConfig)
        duration = self.data[-1]
        max_mem = get_max_rss_gpu() if state.data.is_cuda else get_max_rss_cpu()
        logging.getLogger("emulators").info(
            f"step = {len(self.data)}/{self.timestep_count}, "
            + f"RSS = {max_mem:.3f} MB, "
            + f"Δt = {duration:.3f} s"
        )

        return {
            "RSS": max_mem,
            "duration": duration,
        }


class SVBackendImpl:
    """
    This class is used to handle the state vector and density matrix evolution.

    The pulses and the interaction matrix are sampled at the midpoint of each time
    step, so the Hamiltonian is constant over a step. Consecutive steps with the
    same Hamiltonian, and no observable to evaluate in between, are fused into a
    single exponentiation (see `fused_intervals`): one large step is much cheaper
    than many small ones, for Chebyshev and Lanczos alike, as both need a number of
    operator applications that grows sublinearly with the step at small steps.
    """

    well_prepared_qubits_filter: typing.Optional[torch.Tensor]

    def __init__(self, config: SVConfig, data: SequenceData):
        self.pulser_lindblads = data.lindblad_ops
        use_density_matrix = bool(self.pulser_lindblads) and not (
            (config.solver == Solver.DEFAULT and data.has_stochastic_noise)
            or config.solver == Solver.MONTECARLO
        )
        state_type = DensityMatrix if use_density_matrix else StateVector

        self._config = config
        self._data = data
        self.target_times = data.target_times
        self.omega = data.omega
        self.delta = data.delta
        self.phi = data.phi
        self.nsteps = data.omega.shape[0]
        self.nqubits = data.omega.shape[1]
        self.interaction_matrix = data.interaction_matrix

        requested_gpu = self._config.gpu
        if requested_gpu is None:
            requested_gpu = True

        self.resolved_gpu = requested_gpu

        self.init_initial_state(config, state_type)

        # the Hamiltonian is a RydbergLindbladian for density matrices, and for state
        # vectors a RydbergHamiltonian, applied by the fused kernel where it can be
        self.stepper = self.init_stepper(use_density_matrix)

        self.time = time.time()
        self.results = Results(
            atom_order=data.qubit_ids,
            total_duration=int(self.target_times[-1]),
        )
        self.statistics = Statistics(
            evaluation_times=None,
            data=[],
            timestep_count=self.nsteps,
        )
        self._current_H: None | RydbergLindbladian | RydbergHamiltonian = None
        if self._config.initial_state is not None and (
            self._config.initial_state.n_qudits != self.nqubits
        ):
            raise ValueError(
                "Mismatch in number of atoms: initial state has "
                f"{self._config.initial_state.n_qudits} and the sequence has {self.nqubits}"
            )
        self.init_dark_qubits()

        if self._config.initial_state is not None and self._data.state_prep_error > 0.0:
            raise NotImplementedError(
                "Initial state and state preparation error can not be together."
            )

    def init_initial_state(self, config: SVConfig, state_type: typing.Type) -> None:
        self.state: DensityMatrix | StateVector
        config_state = config.initial_state
        if config_state is not None:
            if type(config_state) is DenseState:
                if config_state.data.dim() == 1:
                    config_state = StateVector(
                        config_state.data, gpu=config_state.data.is_cuda
                    )
                else:
                    config_state = DensityMatrix(
                        config_state.data, gpu=config_state.data.is_cuda
                    )
            if state_type == DensityMatrix and type(config_state) is StateVector:
                config_state = DensityMatrix.from_state_vector(config_state)
            assert isinstance(config_state, state_type)
            self.state = state_type(config_state.data.clone(), gpu=self.resolved_gpu)
        else:
            self.state = state_type.make(self.nqubits, gpu=self.resolved_gpu)

    def init_stepper(self, use_density_matrix: bool) -> BaseStepper:
        max_krylov_dim = self._config.max_krylov_dim
        hamiltonian_type = (
            FusedRydbergHamiltonian
            if not use_density_matrix
            and FusedRydbergHamiltonian.is_available(self.state.data.device, self.nqubits)
            else RydbergHamiltonian
        )
        exponentiator = self.init_exponentiator(use_density_matrix)
        if use_density_matrix:
            return EvolveDensityMatrix(exponentiator, max_krylov_dim)
        if self.pulser_lindblads:
            return EvolveMonteCarlo(
                exponentiator,
                max_krylov_dim,
                hamiltonian_type,
                # jump times are located relative to the configured step, not to
                # the (possibly fused) step actually taken
                jump_time_tolerance=JUMP_TIME_TOLERANCE_FRACTION
                * self._config.dt
                * _TIME_CONVERSION_COEFF,
            )
        return EvolveStateVector(exponentiator, max_krylov_dim, hamiltonian_type)

    def init_exponentiator(self, use_density_matrix: bool) -> Exponentiation:
        """
        The exponentiator requested in the config. By default, Chebyshev if there's
        no Lindblad noise, and Lanczos otherwise. Chebyshev works on state vectors,
        with or without the non-Hermitian jump noise of the Monte Carlo stepper, but
        not on density matrices (the Lindbladian is not of the form H - iΓ).
        """
        requested = self._config.exponentiator
        if requested == Exponentiator.CHEBYSHEV and use_density_matrix:
            raise ValueError(
                "Exponentiator.CHEBYSHEV is not available for density-matrix "
                "evolution; use Exponentiator.LANCZOS."
            )
        if requested == Exponentiator.CHEBYSHEV or (
            requested == Exponentiator.DEFAULT and not use_density_matrix
        ):
            return Chebyshev()
        return Lanczos(self._config.max_krylov_dim)

    def init_dark_qubits(self) -> None:
        if self._data.state_prep_error > 0.0:
            self.well_prepared_qubits_filter = torch.tensor(self._data.bad_atoms)
        else:
            self.well_prepared_qubits_filter = None

        if self.well_prepared_qubits_filter is not None:

            original = self.interaction_matrix

            indices = torch.where(self.well_prepared_qubits_filter)[0]

            # The masked matrix is built once per matrix object `original` returns
            # (it returns the same object as long as its content is the same), so
            # that this callable does too: `fused_intervals` relies on it.
            cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

            def interaction_matrix(t: float) -> torch.Tensor:
                mat = original(t)
                cached = cache.get(id(mat))
                if cached is not None and cached[0] is mat:
                    return cached[1]
                masked = mat.clone()
                masked[:, indices] = 0.0
                masked[:, :, indices] = 0.0
                if len(cache) >= 4:  # bounded, in case `original` makes new tensors
                    del cache[next(iter(cache))]
                cache[id(mat)] = (mat, masked)  # keeping `mat` alive pins its id
                return masked

            self.interaction_matrix = interaction_matrix

            self.omega[:, self.well_prepared_qubits_filter] = 0.0
            self.delta[:, self.well_prepared_qubits_filter] = 0.0
            self.phi[:, self.well_prepared_qubits_filter] = 0.0

    def step(self, start: int, end: int) -> None:
        """One step of the evolution, from time index `start` to `end`"""
        dt = self.target_times[end] - self.target_times[start]
        self._evolve_step(dt, start, end)
        self._apply_observables(end)
        self._save_statistics(end)

    def _midpoint(self, start: int, end: int) -> float:
        return 0.5 * (self.target_times[start] + self.target_times[end])

    def _evolve_step(self, dt: float, start: int, end: int) -> None:
        """
        Evolve from time index `start` to `end` (`start` + 1 by default) with the
        Hamiltonian of step `start`: the pulses are sampled at the midpoint of each
        step and the interaction matrix is read at the midpoint of the interval,
        which `fused_intervals` guarantees is the same for every step in it.
        """
        self._current_H = None  # save a bit of memory
        self.state.data, self._current_H = self.stepper.apply(
            dt * _TIME_CONVERSION_COEFF,
            self.omega[start],
            self.delta[start],
            self.phi[start],
            self.interaction_matrix(self._midpoint(start, end))[0],
            self.state.data,
            self._config.krylov_tolerance,
            self.pulser_lindblads,
        )

    def _callbacks_at(self, step_idx: int) -> list[Observable]:
        """The observables evaluated at time index `step_idx`."""
        norm_time = self.target_times[step_idx] / self.target_times[-1]
        return [
            callback
            for callback in self._config.observables
            if is_evaluation_time(self._config, callback, norm_time)
        ]

    def fused_intervals(self) -> list[tuple[int, int]]:
        """
        The time-index intervals (start, end) to evolve over in one exponentiation
        each, covering all steps in order. Consecutive steps are fused when their
        pulse parameters are equal, their interaction matrices are the same object
        (which the Pulser adapter guarantees when they are equal in content), and
        no observable is evaluated at the time between them. Nothing is fused when
        gradients are requested: the parameters of the two steps are then distinct
        inputs of the computation, which a fused step would not both differentiate.
        """
        nsteps = self.nsteps
        if nsteps == 0:
            return []
        if torch.is_grad_enabled() and any(
            t.requires_grad for t in (self.omega, self.delta, self.phi, self.state.data)
        ):
            return [(i, i + 1) for i in range(nsteps)]
        same_pulses = (
            (self.omega[1:] == self.omega[:-1]).all(dim=1)
            & (self.delta[1:] == self.delta[:-1]).all(dim=1)
            & (self.phi[1:] == self.phi[:-1]).all(dim=1)
        ).tolist()
        intervals = []
        start = 0
        previous = self.interaction_matrix(self._midpoint(0, 1))
        for i in range(1, nsteps):
            matrix = self.interaction_matrix(self._midpoint(i, i + 1))
            fuse = (
                same_pulses[i - 1]
                and matrix is previous
                and not (torch.is_grad_enabled() and matrix.requires_grad)
                and not self._callbacks_at(i)
            )
            if not fuse:
                intervals.append((start, i))
                start = i
                previous = matrix
        intervals.append((start, nsteps))
        return intervals

    def _apply_observables(self, step_idx: int) -> None:
        norm_time = self.target_times[step_idx] / self.target_times[-1]
        callbacks_for_current_time_step = self._callbacks_at(step_idx)
        if not callbacks_for_current_time_step:
            return
        norm = self.state.norm()
        state: DensityMatrix | StateVector
        if not torch.allclose(norm, torch.tensor(1.0, dtype=torch.float64)):
            state = (1 / norm.item()) * self.state
        else:
            state = self.state
        if not self._current_H:
            self._current_H = self.stepper.get_hamiltonian(
                omegas=self.omega[0],
                deltas=self.delta[0],
                phis=self.phi[0],
                pulser_lindblads=self.pulser_lindblads,
                interaction_matrix=self.interaction_matrix(
                    0.5 * (self.target_times[step_idx] + self.target_times[step_idx + 1])
                )[0],
                device=state.data.device,
            )
        for callback in callbacks_for_current_time_step:
            callback(
                self._config,
                norm_time,
                state,
                self._current_H,  # type: ignore[arg-type]
                self.results,
            )

    def _save_statistics(self, step_idx: int) -> None:
        norm_time = self.target_times[step_idx] / self.target_times[-1]

        self.statistics.data.append(time.time() - self.time)
        self.statistics(
            self._config,
            norm_time,
            self.state,
            self._current_H,  # type: ignore[arg-type]
            self.results,
        )
        self.time = time.time()

    def _run(self) -> Results:
        self._apply_observables(0)  # at t == 0 for pulser compatibility
        intervals = self.fused_intervals()
        self.statistics.timestep_count = len(intervals)
        for start, end in intervals:
            self.step(start, end)

        return self.results
