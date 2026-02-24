import time
import typing
import torch
import logging

from emu_sv.hamiltonian import RydbergHamiltonian
from emu_sv.lindblad_operator import RydbergLindbladian

from pulser.backend import Results, Observable, State, EmulationConfig
from emu_base import SequenceData, get_max_rss

from emu_sv.state_vector import StateVector
from emu_sv.density_matrix_state import DensityMatrix
from emu_sv.sv_config import SVConfig
from emu_sv.time_evolution import EvolveStateVector, EvolveDensityMatrix

_TIME_CONVERSION_COEFF = 0.001  # Omega and delta are given in rad/μs, dt in ns


class Statistics(Observable):
    def __init__(
        self,
        evaluation_times: typing.Sequence[float] | None,
        data: list[float],
        timestep_count: int,
    ):
        super().__init__(evaluation_times=evaluation_times)
        self.data = data
        self.timestep_count = timestep_count

    @property
    def _base_tag(self) -> str:
        return "statistics"

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
        max_mem = get_max_rss(state.data.is_cuda)
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
    """

    well_prepared_qubits_filter: typing.Optional[torch.Tensor]

    def __init__(self, config: SVConfig, data: SequenceData):
        self.pulser_lindblads = data.lindblad_ops
        stepper: type[EvolveStateVector] | type[EvolveDensityMatrix]
        state_type: type[StateVector] | type[DensityMatrix]
        if self.pulser_lindblads:
            stepper = EvolveDensityMatrix
            state_type = DensityMatrix
        else:
            stepper = EvolveStateVector
            state_type = StateVector

        self.stepper = stepper
        self._config = config
        self._data = data
        self.target_times = data.target_times
        self.omega = data.omega
        self.delta = data.delta
        self.phi = data.phi
        self.nsteps = data.omega.shape[0]
        self.nqubits = data.omega.shape[1]
        self.full_interaction_matrix = data.full_interaction_matrix

        requested_gpu = self._config.gpu
        if requested_gpu is None:
            requested_gpu = True

        self.resolved_gpu = requested_gpu

        self.state: DensityMatrix | StateVector
        if config.initial_state is not None:
            assert isinstance(config.initial_state, state_type)
            self.state = state_type(
                config.initial_state.data.clone(), gpu=self.resolved_gpu
            )
        else:
            self.state = state_type.make(self.nqubits, gpu=self.resolved_gpu)

        self.time = time.time()
        self.results = Results(
            atom_order=data.qubit_ids,
            total_duration=int(self.target_times[-1]),
        )
        self.statistics = Statistics(
            evaluation_times=[t / self.target_times[-1] for t in self.target_times],
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

        if (
            self._config.initial_state is not None
            and self._data.noise_model.state_prep_error > 0.0
        ):
            raise NotImplementedError(
                "Initial state and state preparation error can not be together."
            )

    def init_dark_qubits(self) -> None:
        if self._data.noise_model.state_prep_error > 0.0:
            bad_atoms = self._data.bad_atoms
            self.well_prepared_qubits_filter = torch.tensor(
                [bool(bad_atoms[x]) for x in self._data.qubit_ids]
            )
        else:
            self.well_prepared_qubits_filter = None

        if self.well_prepared_qubits_filter is not None:

            self.full_interaction_matrix[self.well_prepared_qubits_filter, :] = 0.0
            self.full_interaction_matrix[:, self.well_prepared_qubits_filter] = 0.0
            self.omega[:, self.well_prepared_qubits_filter] = 0.0
            self.delta[:, self.well_prepared_qubits_filter] = 0.0
            self.phi[:, self.well_prepared_qubits_filter] = 0.0

    def step(self, step_idx: int) -> None:
        """One step of the evolution"""
        dt = self._compute_dt(step_idx)
        self._evolve_step(dt, step_idx)
        step_idx += 1
        self._apply_observables(step_idx)
        self._save_statistics(step_idx)

    def _compute_dt(self, step_idx: int) -> float:
        return self.target_times[step_idx + 1] - self.target_times[step_idx]

    def _evolve_step(self, dt: float, step_idx: int) -> None:
        """One step evolution"""
        self._current_H = None  # save a bit of memory
        self.state.data, self._current_H = self.stepper.apply(
            dt * _TIME_CONVERSION_COEFF,
            self.omega[step_idx],
            self.delta[step_idx],
            self.phi[step_idx],
            self.full_interaction_matrix,
            self.state.data,
            self._config.krylov_tolerance,
            self.pulser_lindblads,
        )

    def _is_evaluation_time(
        self,
        observable: Observable,
        t: float,
        tolerance: float = 1e-10,
    ) -> bool:
        """Return True if ``t`` is a genuine sampling time for this observable.

        Filters out nearby points that are close to, but not in, the
        observable's evaluation times (within ``tolerance``).
        Prevent false matches by using Pulser's tolerance
        tol = 0.5 / total_duration. (deep inside pulser Observable class)
        """
        times = observable.evaluation_times

        is_observable_eval_time = (
            times is not None
            and self._config.is_time_in_evaluation_times(t, times, tol=tolerance)
        )

        is_default_eval_time = self._config.is_evaluation_time(t, tol=tolerance)

        return is_observable_eval_time or is_default_eval_time

    def _apply_observables(self, step_idx: int) -> None:
        norm_time = self.target_times[step_idx] / self.target_times[-1]
        callbacks_for_current_time_step = [
            callback
            for callback in self._config.observables
            if self._is_evaluation_time(callback, norm_time)
        ]
        if not self._current_H and callbacks_for_current_time_step:
            self._current_H = self.stepper.get_hamiltonian(
                omegas=self.omega[0],
                deltas=self.delta[0],
                phis=self.phi[0],
                pulser_lindblads=self.pulser_lindblads,
                interaction_matrix=self.full_interaction_matrix,
                device=self.state.data.device,
            )
        for callback in callbacks_for_current_time_step:
            callback(
                self._config,
                norm_time,
                self.state,
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
        for step in range(self.nsteps):
            self.step(step)

        return self.results
