from time import time
from pulser import Sequence

from emu_base import Backend, BackendConfig, Results
from emu_mps.mps_config import MPSConfig
from emu_mps.mps_backend_impl import MPSBackendImpl


class MPSBackend(Backend):
    """
    A backend for emulating Pulser sequences using Matrix Product States (MPS),
    aka tensor trains.
    """

    def run(self, sequence: Sequence, mps_config: BackendConfig) -> Results:
        """
        Emulates the given sequence.

        Args:
            sequence: a Pulser sequence to simulate
            mps_config: the backends config. Should be of type MPSConfig

        Returns:
            the simulation results
        """
        assert isinstance(mps_config, MPSConfig)

        self.validate_sequence(sequence)

        results = Results()

        impl = MPSBackendImpl(sequence, mps_config)
        impl.init_dark_qubits()
        impl.init_initial_state(mps_config.initial_state)
        impl.init_lindblad_noise()

        for step in range(impl.timestep_count):
            start = time()

            impl.do_time_step(step)

            impl.fill_results(results, step)

            end = time()
            impl.log_step_statistics(results, step=step, duration=end - start)

        return results
