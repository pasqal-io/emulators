from pulser import Sequence

from emu_base import Backend, BackendConfig, Results
from emu_mps.mps_config import MPSConfig
from emu_mps.mps_backend_impl import create_impl


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

        impl = create_impl(sequence, mps_config)
        impl.init()  # This is separate from the constructor for testing purposes.

        while not impl.is_finished():
            impl.progress()

        return impl.results
