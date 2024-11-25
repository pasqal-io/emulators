from abc import ABC, abstractmethod
from typing import Any, Optional, TYPE_CHECKING
from enum import Enum, auto

from emu_base.base_classes.config import BackendConfig
from emu_base.base_classes.operator import Operator
from emu_base.base_classes.state import State

if TYPE_CHECKING:
    from emu_base.base_classes.results import Results


class AggregationType(Enum):
    """
    Defines how to combine multiple values from different simulation results.
    """

    MEAN = auto()  # statistics.fmean or list/matrix-wise equivalent
    BAG_UNION = auto()  # Counter.__add__


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
        self, config: BackendConfig, t: int, state: State, H: Operator, result: "Results"
    ) -> None:
        """
        This function is called after each time step performed by the emulator.
        By default it calls apply to compute a result and put it in `result`
        if `t` in `self.evaluation_times`.
        It can be overloaded to define any custom behaviour for a `Callback`.

        Args:
            config: the config object passed to the run method
            t: the current time in ns
            state: the current state
            H: the Hamiltonian at this time
            result: the results object
        """
        if t in self.evaluation_times:
            value_to_store = self.apply(config, t, state, H)
            result.store(callback=self, time=t, value=value_to_store)

    @property
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

    @property
    def default_aggregation_type(self) -> Optional[AggregationType]:
        """
        Defines how to combine by default multiple values from different simulation results.
        None means no default, therefore aggregator function is always user-provided.
        """
        return None
