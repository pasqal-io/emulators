import torch
from resource import RUSAGE_SELF, getrusage
import time
import typing

from pulser.backend import EmulatorBackend, Results, Observable, State, EmulationConfig

from emu_base import PulserData

from emu_sv.state_vector import StateVector
from emu_sv.sv_config import SVConfig
from emu_sv.time_evolution import do_time_step


_TIME_CONVERSION_COEFF = 0.001  # Omega and delta are given in rad/μs, dt in ns


class SVBackend(EmulatorBackend):
    """
    A backend for emulating Pulser sequences using state vectors and sparse matrices.
    """

    default_config = SVConfig()

    def run(self) -> Results:
        """
        Emulates the given sequence.

        Returns:
            the simulation results
        """
        assert isinstance(self._config, SVConfig)

        pulser_data = PulserData(
            sequence=self._sequence, config=self._config, dt=self._config.dt
        )
        self.target_times = pulser_data.target_times
        self.time = time.time()
        omega, delta, phi = pulser_data.omega, pulser_data.delta, pulser_data.phi

        nsteps = omega.shape[0]
        nqubits = omega.shape[1]

        self.results = Results(atom_order=(), total_duration=self.target_times[-1])
        self.statistics = Statistics(
            evaluation_times=[t / self.target_times[-1] for t in self.target_times],
            data=[],
            timestep_count=nsteps,
        )

        if self._config.initial_state is not None:
            state = self._config.initial_state
            state = StateVector(state.vector.clone(), gpu=state.vector.is_cuda)
        else:
            state = StateVector.make(nqubits, gpu=self._config.gpu)

        for step in range(nsteps):
            dt = self.target_times[step + 1] - self.target_times[step]

            state.vector, H = do_time_step(
                dt * _TIME_CONVERSION_COEFF,
                omega[step],
                delta[step],
                phi[step],
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

        return self.results


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
        assert isinstance(state, StateVector)
        assert isinstance(config, SVConfig)
        duration = self.data[-1]
        if state.vector.is_cuda:
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
