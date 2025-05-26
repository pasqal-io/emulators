from pulser.backend import EmulatorBackend
from pulser.backend import Results
from emu_sv.sv_config import SVConfig
from emu_sv.sv_backend_impl import create_impl


class SVBackend(EmulatorBackend):
    """
    A backend for emulating Pulser sequences using state vectors and sparse matrices.
    Noisy simulation is supported by solving the Lindblad equation and using effective
    noise channel or jump operators
    """

    default_config = SVConfig()

    def run(self) -> Results:
        """
        Emulates the given sequence.

        Returns:
            the simulation results
        """
        assert isinstance(self._config, SVConfig)

        impl = create_impl(self._sequence, self._config)
        return impl._run()
