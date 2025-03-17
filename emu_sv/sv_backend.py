import torch
from resource import RUSAGE_SELF, getrusage
from time import time

from pulser import Sequence

# from emu_base.base_classes.backend import BackendConfig
# from emu_base.base_classes.backend import Backend

from pulser.backend import Backend, EmulationConfig, Results

# from emu_base.base_classes.results import Results

from emu_base import DEVICE_COUNT
from emu_base.pulser_adapter import PulserData

from emu_sv.state_vector import StateVector
from emu_sv.sv_config import SVConfig
from emu_sv.time_evolution import do_time_step

_TIME_CONVERSION_COEFF = 0.001  # Omega and delta are given in rad/ms, dt in ns


class SVBackend(Backend):
    """
    A backend for emulating Pulser sequences using state vectors and sparse matrices.
    """

    def __init__(self, sequence: Sequence, sv_config: EmulationConfig):
        super().__init__(sequence=sequence, mimic_qpu=False)
        assert isinstance(sv_config, EmulationConfig)
        self.config = sv_config

    def run(self) -> Results:
        """
        Emulates the given sequence.

        Args:
            sequence: a Pulser sequence to simulate
            sv_config: the backends config. Should be of type SVConfig

        Returns:
            the simulation results
        """
        # TODO: do I have SVconfig
        assert isinstance(self.config, SVConfig)

        results = Results()

        data = PulserData(sequence=self.sequence, config=self.config, dt=self.config.dt)
        omega, delta, phi = data.omega, data.delta, data.phi

        target_times = data.target_times

        nsteps = omega.shape[0]
        nqubits = omega.shape[1]
        device = "cuda" if self.config.gpu and DEVICE_COUNT > 0 else "cpu"

        if self.config.initial_state is not None:
            state = self.config.initial_state
            state.vector = state.vector.to(device)
        else:
            state = StateVector.make(nqubits, gpu=self.config.gpu)

        for step in range(nsteps):
            start = time()
            dt = target_times[step + 1] - target_times[step]

            state.vector, H = do_time_step(
                dt * _TIME_CONVERSION_COEFF,
                omega[step],
                delta[step],
                phi[step],
                data.full_interaction_matrix,
                state.vector,
                self.config.krylov_tolerance,
            )

            # TODO: remove this type ignore thing
            for callback in self.config.callbacks:
                callback(
                    self.config,
                    (step + 1) * self.config.dt,
                    state,  # type: ignore[arg-type]
                    H,  # type: ignore[arg-type]
                    results,
                )

            end = time()
            self.log_step_statistics(
                results,
                step=step,
                duration=end - start,
                timestep_count=nsteps,
                state=state,
                sv_config=self.config,
            )

        return results

    @staticmethod
    def log_step_statistics(
        results: Results,
        *,
        step: int,
        duration: float,
        timestep_count: int,
        state: StateVector,
        sv_config: SVConfig,
    ) -> None:
        if state.vector.is_cuda:
            max_mem_per_device = (
                torch.cuda.max_memory_allocated(device) * 1e-6
                for device in range(torch.cuda.device_count())
            )
            max_mem = max(max_mem_per_device)
        else:
            max_mem = getrusage(RUSAGE_SELF).ru_maxrss * 1e-3

        sv_config.logger.info(
            f"step = {step + 1}/{timestep_count}, "
            + f"RSS = {max_mem:.3f} MB, "
            + f"Δt = {duration:.3f} s"
        )

        if results.statistics is None:
            assert step == 0
            results.statistics = {"steps": []}

        assert "steps" in results.statistics
        assert len(results.statistics["steps"]) == step

        results.statistics["steps"].append(
            {
                "RSS": max_mem,
                "duration": duration,
            }
        )
