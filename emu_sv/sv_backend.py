from emu_base.base_classes.backend import Backend, BackendConfig
from emu_base.base_classes.results import Results
from emu_sv.sv_config import SVConfig
from pulser import Sequence
from emu_base.pulser_adapter import PulserData
from emu_sv.time_evolution import do_time_step
from emu_sv import StateVector
import torch
from time import time
from resource import RUSAGE_SELF, getrusage
from emu_base import DEVICE_COUNT
from copy import deepcopy

_TIME_CONVERSION_COEFF = 0.001  # Omega and delta are given in rad/ms, dt in ns


class SVBackend(Backend):
    """
    A backend for emulating Pulser sequences using state vectors and sparse matrices.
    """

    def run(self, sequence: Sequence, sv_config: BackendConfig) -> Results:
        """
        Emulates the given sequence.

        Args:
            sequence: a Pulser sequence to simulate
            sv_config: the backends config. Should be of type SVConfig

        Returns:
            the simulation results
        """
        assert isinstance(sv_config, SVConfig)

        self.validate_sequence(sequence)

        results = Results()

        data = PulserData(sequence=sequence, config=sv_config, dt=sv_config.dt)
        omega, delta, phi = data.omega, data.delta, data.phi

        target_times = data.target_times

        nsteps = omega.shape[0]
        nqubits = omega.shape[1]
        device = "cuda" if sv_config.gpu and DEVICE_COUNT > 0 else "cpu"

        if sv_config.initial_state is not None:
            state = deepcopy(sv_config.initial_state)
            state.vector = state.vector.to(device)
        else:
            state = StateVector.make(nqubits, gpu=sv_config.gpu)

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
                sv_config.krylov_tolerance,
            )

            for callback in sv_config.callbacks:
                callback(
                    sv_config,
                    target_times[step + 1],
                    state,
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
                sv_config=sv_config,
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
