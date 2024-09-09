from abc import ABC, abstractmethod
from typing import Any

from emu_mps.base_classes.config import BackendConfig
from emu_mps.base_classes.operator import Operator
from emu_mps.base_classes.results import Results
from emu_mps.base_classes.state import State


class Callback(ABC):
    def __init__(self, evaluation_times: set[int]):
        """
        The callback base class that can be subclassed to add new kinds of results
        to the Results object returned by the Backend

        Args:
            evaluation_times: the times at which to add a result to Results
        """
        self.evaluation_times = evaluation_times

    def __call__(
        self, config: BackendConfig, t: int, state: State, H: Operator, result: Results
    ) -> None:
        """
        This function is called after each time step performed by the emulator.
        By default it calls apply to compute a result and put it in Results
        if t in self.evaluation_times.
        It can be overloaded to define general callbacks that don't put results
        in the Results object.

        Args:
            config: the config object passed to the run method
            t: the current time in ns
            state: the current state
            H: the Hamiltonian at this time
            result: the results object
        """
        if t in self.evaluation_times:
            result[self.name()][t] = self.apply(config, t, state, H)

    @abstractmethod
    def name(self) -> str:
        """
        The name of the observable, can be used to index into the Results object.
        Some Callbacks might have multiple instances, such as a callback to compute
        a fidelity on some given state. In that case, this method could make sure
        each instance has a unique name.

        Returns:
            the name of the callback
        """
        pass

    @abstractmethod
    def apply(self, config: BackendConfig, t: int, state: State, H: Operator) -> Any:
        """
        This method must be implemented by subclasses. The result of this method
        gets put in the Results object.

        Args:
            config: the config object passed to the run method
            t: the current time in ns
            state: the current state
            H: the Hamiltonian at this time

        Returns:
            the result to put in Results
        """
        pass
