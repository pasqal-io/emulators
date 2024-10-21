import warnings
from abc import ABC, abstractmethod

from pulser import Sequence

from emu_base.base_classes.config import BackendConfig
from emu_base.base_classes.results import Results


class Backend(ABC):
    """
    Base class for different emulation backends.
    Forces backends to implement a run method.
    """

    @staticmethod
    def validate_sequence(sequence: Sequence) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)

        if not isinstance(sequence, Sequence):
            raise TypeError(
                "The provided sequence has to be a valid " "pulser.Sequence instance."
            )
        if sequence.is_parametrized() or sequence.is_register_mappable():
            raise ValueError(
                "Not supported"
                "The provided sequence needs to be built to be simulated. Call"
                " `Sequence.build()` with the necessary parameters."
            )
        if not sequence._schedule:
            raise ValueError("The provided sequence has no declared channels.")
        if all(sequence._schedule[x][-1].tf == 0 for x in sequence.declared_channels):
            raise ValueError("No instructions given for the channels in the sequence.")

    @abstractmethod
    def run(self, sequence: Sequence, config: BackendConfig) -> Results:
        """
        Emulates the given sequence.

        Args:
            sequence: a Pulser sequence to simulate
            config: the config. Should be of the appropriate type for the backend

        Returns:
            the simulation results
        """
        pass
