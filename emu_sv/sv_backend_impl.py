import time
import typing

from pulser import Sequence
import torch
from resource import RUSAGE_SELF, getrusage

from pulser.backend import Results, Observable, State, EmulationConfig
from emu_base import PulserData

from emu_sv.state_vector import StateVector
from emu_sv.density_matrix_state import DensityMatrix
from emu_sv.sv_config import SVConfig
from emu_sv.time_evolution import EvolveStateVector
from emu_sv.time_evolution import do_noisy_time_step


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
        if isinstance(state, StateVector) and state.vector.is_cuda:
            max_mem_per_device = (
                torch.cuda.max_memory_allocated(device) * 1e-6
                for device in range(torch.cuda.device_count())
            )
            max_mem = max(max_mem_per_device)
        elif isinstance(state, DensityMatrix) and state.matrix.is_cuda:
            max_mem_per_device = (
                torch.cuda.max_memory_allocated(device) * 1e-6
                for device in range(torch.cuda.device_count())
            )
            max_mem = max(max_mem_per_device)
        else:
            max_mem = getrusage(RUSAGE_SELF).ru_maxrss * 1e-3

        config.logger.info(
            f"step = {len(self.data)}/{self.timestep_count}, "
            + f"RSS = {max_mem:.3f} MB, "
            + f"Δt = {duration:.3f} s"
        )

        return {
            "RSS": max_mem,
            "duration": duration,
        }


_TIME_CONVERSION_COEFF = 0.001  # Omega and delta are given in rad/μs, dt in ns


class SVBackendImpl:

    results: Results

    def __init__(self, config: SVConfig, pulser_data: PulserData):
        """
        Initializes the SVBackendImpl.

        Args:
            config: The configuration for the emulator.
            pulser_data: The data for the sequence to be emulated.
        """
        self._config = config
        self._pulser_data = pulser_data
        self.target_times = pulser_data.target_times
        self.time = 0.0
        self.statistics = None
        self.omega = pulser_data.omega
        self.delta = pulser_data.delta
        self.phi = pulser_data.phi
        self.nsteps = pulser_data.omega.shape[0]
        self.nqubits = pulser_data.omega.shape[1]

        self.time = time.time()

        self.results = Results(atom_order=(), total_duration=self.target_times[-1])
        self.statistics = Statistics(
            evaluation_times=[t / self.target_times[-1] for t in self.target_times],
            data=[],
            timestep_count=self.nsteps,
        )

        if self._config.initial_state is not None:
            state = self._config.initial_state
            state = StateVector(state.vector.clone(), gpu=state.vector.is_cuda)
        else:
            state = StateVector.make(self.nqubits, gpu=self._config.gpu)

        stepper = EvolveStateVector.apply
        for step in range(self.nsteps):
            dt = self.target_times[step + 1] - self.target_times[step]
            state.vector, H = stepper(
                dt * _TIME_CONVERSION_COEFF,
                self.omega[step],
                self.delta[step],
                self.phi[step],
                pulser_data.full_interaction_matrix,
                state.vector,
                self._config.krylov_tolerance,
            )

            # callbacks in observables and self.statistics in H
            # have "# type: ignore[arg-type]" because H has it's own type
            # meaning H is not inherited from Operator class.
            # We decided that ignore[arg-type] is better compared to
            # having many unused NotImplemented methods
            for callback in self._config.observables:
                callback(
                    self._config,
                    self.target_times[step + 1] / self.target_times[-1],
                    state,
                    H,  # type: ignore[arg-type]
                    self.results,
                )

            self.statistics.data.append(time.time() - self.time)
            self.statistics(
                self._config,
                self.target_times[step + 1] / self.target_times[-1],
                state,
                H,  # type: ignore[arg-type]
                self.results,
            )
            self.time = time.time()
            del H


class NoisySVBackendImpl(SVBackendImpl):
    results: Results

    def __init__(self, config: SVConfig, pulser_data: PulserData):
        """
        Initializes the SVBackendImpl.

        Args:
            config: The configuration for the emulator.
            pulser_data: The data for the sequence to be emulated.
        """
        self._config = config
        self._pulser_data = pulser_data
        self.target_times = pulser_data.target_times
        self.time = 0.0
        self.statistics = None
        self.omega = pulser_data.omega
        self.delta = pulser_data.delta
        self.phi = pulser_data.phi
        self.nsteps = pulser_data.omega.shape[0]
        self.nqubits = pulser_data.omega.shape[1]
        self.pulser_lindblads = pulser_data.lindblad_ops

        self.time = time.time()

        self.results = Results(atom_order=(), total_duration=self.target_times[-1])
        self.statistics = Statistics(
            evaluation_times=[t / self.target_times[-1] for t in self.target_times],
            data=[],
            timestep_count=self.nsteps,
        )

        if self._config.initial_state is not None:  # fix this with state vector
            state = self._config.initial_state
            state = DensityMatrix(state.matrix.clone(), gpu=self._config.gpu)
        else:
            state = DensityMatrix.make(self.nqubits, gpu=self._config.gpu)

        for step in range(self.nsteps):
            dt = self.target_times[step + 1] - self.target_times[step]
            state.matrix, H = do_noisy_time_step(
                dt * _TIME_CONVERSION_COEFF,
                self.omega[step],
                self.delta[step],
                self.phi[step],
                pulser_data.full_interaction_matrix,
                state.matrix,
                self._config.krylov_tolerance,
                self.pulser_lindblads,
            )

            # callbacks in observables and self.statistics in H
            # have "# type: ignore[arg-type]" because H has it's own type
            # meaning H is not inherited from Operator class.
            # We decided that ignore[arg-type] is better compared to
            # having many unused NotImplemented methods
            for callback in self._config.observables:
                callback(
                    self._config,
                    self.target_times[step + 1] / self.target_times[-1],
                    state,
                    H,  # type: ignore[arg-type]
                    self.results,
                )

            self.statistics.data.append(time.time() - self.time)
            self.statistics(
                self._config,
                self.target_times[step + 1] / self.target_times[-1],
                state,
                H,  # type: ignore[arg-type]
                self.results,
            )
            self.time = time.time()
            del H


def create_impl(sequence: Sequence, config: SVConfig) -> SVBackendImpl:
    """
    Creates the backend implementation for the given sequence and configuration.

    Args:
        sequence: The sequence to be emulated.
        config: The configuration for the emulator.

    Returns:
        An instance of SVBackendImpl.
    """
    pulse_data = PulserData(sequence=sequence, config=config, dt=config.dt)
    if pulse_data.has_lindblad_noise:
        return NoisySVBackendImpl(config, pulse_data)

    return SVBackendImpl(config, pulse_data)
