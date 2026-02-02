from pulser.backend import EmulatorBackend
from pulser.backend import Results
from emu_sv.sv_config import SVConfig
from emu_sv.sv_backend_impl import create_impl
from emu_base import PulserData


class SVBackend(EmulatorBackend):
    """
    A backend for emulating Pulser sequences using state vectors and sparse matrices.
    Noisy simulation is supported by solving the Lindblad equation and using effective
    noise channel or jump operators
    """

    default_config = SVConfig()

    def run(self) -> Results | list[Results]:
        """
        Emulates the given sequence.

        Returns:
            the simulation results
        """
        assert isinstance(self._config, SVConfig)
        pulser_data = PulserData(
            sequence=self._sequence, config=self._config, dt=self._config.dt
        )
        results = []
        for sequence_data in pulser_data.get_sequences():
            impl = create_impl(sequence_data, self._config)
            results.append(impl._run())
        return Results.aggregate(results)
