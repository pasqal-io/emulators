from pulser.backend import EmulatorBackend, Results, BitStrings
from emu_sv.sv_config import SVConfig
from emu_sv.sv_backend_impl import SVBackendImpl
from emu_base import PulserData


class SVBackend(EmulatorBackend):
    """
    A backend for emulating Pulser sequences using state vectors and sparse matrices.
    Noisy simulation is supported by solving the Lindblad equation and using effective
    noise channel or jump operators

    Args:
        config (SVConfig): Configuration for the SV backend.
    """

    default_config = SVConfig(observables=[BitStrings(evaluation_times=[1.0])])

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
            impl = SVBackendImpl(self._config, sequence_data)
            results.append(impl._run())
        return Results.aggregate(results)
