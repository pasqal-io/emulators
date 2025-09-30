from abc import abstractmethod
import time
import typing
import torch

from emu_sv.hamiltonian import RydbergHamiltonian
from emu_sv.lindblad_operator import RydbergLindbladian
from pulser import Sequence

from pulser.backend import Results, Observable, State, EmulationConfig
from emu_base import PulserData, get_max_rss

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
        max_mem = get_max_rss(
            isinstance(state, StateVector)
            and state.vector.is_cuda
            or isinstance(state, DensityMatrix)
            and state.matrix.is_cuda
        )

        config.logger.info(
            f"step = {len(self.data)}/{self.timestep_count}, "
            + f"RSS = {max_mem:.3f} MB, "
            + f"Δt = {duration:.3f} s"
        )

        return {
            "RSS": max_mem,
            "duration": duration,
        }


class BaseSVBackendImpl:
    """
    This class is used to handle the state vector and density matrix evolution.
    """

    well_prepared_qubits_filter: typing.Optional[torch.Tensor]

    def __init__(self, config: SVConfig, pulser_data: PulserData):
        self._config = config
        self._pulser_data = pulser_data
        self.target_times = pulser_data.target_times
        self.omega = pulser_data.omega
        self.delta = pulser_data.delta
        self.phi = pulser_data.phi
        self.nsteps = pulser_data.omega.shape[0]
        self.nqubits = pulser_data.omega.shape[1]
        self.full_interaction_matrix = pulser_data.full_interaction_matrix
        self.state: State
        self.time = time.time()
        self.results = Results(atom_order=(), total_duration=self.target_times[-1])
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
            and self._config.noise_model.state_prep_error > 0.0
        ):
            raise NotImplementedError(
                "Initial state and state preparation error can not be together."
            )

    def init_dark_qubits(self) -> None:
        if self._config.noise_model.state_prep_error > 0.0:
            bad_atoms = self._pulser_data.hamiltonian.bad_atoms
            self.well_prepared_qubits_filter = torch.tensor(
                [bool(bad_atoms[x]) for x in self._pulser_data.qubit_ids]
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
        self._apply_observables(step_idx)
        self._save_statistics(step_idx)

    def _compute_dt(self, step_idx: int) -> float:
        return self.target_times[step_idx + 1] - self.target_times[step_idx]

    @abstractmethod
    def _evolve_step(self, dt: float, step_idx: int) -> None:
        """One step evolution"""

    def _apply_observables(self, step_idx: int) -> None:
        norm_time = self.target_times[step_idx + 1] / self.target_times[-1]
        for callback in self._config.observables:
            callback(
                self._config,
                norm_time,
                self.state,
                self._current_H,  # type: ignore[arg-type]
                self.results,
            )

    def _save_statistics(self, step_idx: int) -> None:
        norm_time = self.target_times[step_idx + 1] / self.target_times[-1]
        self.statistics.data.append(time.time() - self.time)
        self.statistics(
            self._config,
            norm_time,
            self.state,
            self._current_H,  # type: ignore[arg-type]
            self.results,
        )
        self.time = time.time()
        self._current_H = None

    def _run(self) -> Results:
        for step in range(self.nsteps):
            self.step(step)

        return self.results


class SVBackendImpl(BaseSVBackendImpl):

    def __init__(self, config: SVConfig, pulser_data: PulserData):
        """
        For running sequences without noise. The state will evolve according
        to e^(-iH t)

        Args:
            config: The configuration for the emulator.
            pulser_data: The data for the sequence to be emulated.
        """
        super().__init__(config, pulser_data)

        self.state: StateVector = (
            StateVector.make(self.nqubits, gpu=self._config.gpu)
            if self._config.initial_state is None
            else StateVector(
                self._config.initial_state.vector.clone(),
                gpu=self._config.gpu,
            )
        )

        self.stepper = EvolveStateVector.apply

    def _evolve_step(self, dt: float, step_idx: int) -> None:
        self.state.vector, self._current_H = self.stepper(
            dt * _TIME_CONVERSION_COEFF,
            self.omega[step_idx],
            self.delta[step_idx],
            self.phi[step_idx],
            self.full_interaction_matrix,
            self.state.vector,
            self._config.krylov_tolerance,
        )


class NoisySVBackendImpl(BaseSVBackendImpl):

    def __init__(self, config: SVConfig, pulser_data: PulserData):
        """
        Initializes the NoisySVBackendImpl, master equation version.
        This class handles the Lindblad operators and
        solves the Lindblad master equation

        Args:
            config: The configuration for the emulator.
            pulser_data: The data for the sequence to be emulated.
        """

        super().__init__(config, pulser_data)

        self.pulser_lindblads = pulser_data.lindblad_ops

        self.state: DensityMatrix = (
            DensityMatrix.make(self.nqubits, gpu=self._config.gpu)
            if self._config.initial_state is None
            else DensityMatrix(
                self._config.initial_state.matrix.clone(), gpu=self._config.gpu
            )
        )

    def _evolve_step(self, dt: float, step_idx: int) -> None:
        self.state.matrix, self._current_H = EvolveDensityMatrix.evolve(
            dt * _TIME_CONVERSION_COEFF,
            self.omega[step_idx],
            self.delta[step_idx],
            self.phi[step_idx],
            self.full_interaction_matrix,
            self.state.matrix,
            self._config.krylov_tolerance,
            self.pulser_lindblads,
        )


def create_impl(sequence: Sequence, config: SVConfig) -> BaseSVBackendImpl:
    """
    Creates the backend implementation for the given sequence and config.

    Args:
        sequence: The sequence to be emulated.
        config: configuration for the emulator.

    Returns:
        An instance of SVBackendImpl.
    """
    pulse_data = PulserData(sequence=sequence, config=config, dt=config.dt)
    if pulse_data.has_lindblad_noise:
        return NoisySVBackendImpl(config, pulse_data)
    else:
        return SVBackendImpl(config, pulse_data)
