from pulser.backend import EmulatorBackend, Results, BitStrings
from emu_sv.sv_config import SVConfig
from emu_sv.sv_backend_impl import SVBackendImpl
from emu_base import PulserData, SequenceData


class SVBackend(EmulatorBackend):
    """
    A backend for emulating Pulser sequences using state vectors and sparse matrices.
    Noisy simulation is supported by solving the Lindblad equation and using effective
    noise channel or jump operators

    Args:
        sequence: The sequence to be executed.
        config (SVConfig): Configuration for the SV backend.
        mimic_qpu: Whether to enforce Register constrains imposed
            by the device in the Sequence.
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
            results.append(self._run_from_sequence_data(sequence_data, self._config))
        return Results.aggregate(results)

    @staticmethod
    def _run_from_sequence_data(sequence_data: SequenceData, config: SVConfig) -> Results:
        impl = SVBackendImpl(config, sequence_data)
        return impl._run()
