from abc import ABC, abstractmethod
from typing import Any
from emu_ct.base_classes.config import BackendConfig
from emu_ct.base_classes.state import State
from emu_ct.base_classes.results import Results
from emu_ct.base_classes.operator import Operator


class Callback(ABC):
    def __init__(self, times: set[int]):
        self.times = times

    def __call__(
        self, config: BackendConfig, t: int, state: State, H: Operator, result: Results
    ) -> None:
        if t in self.times:
            result[self.name()][t] = self.apply(config, t, state, H)

    @abstractmethod
    def name(self) -> str:
        # return the name of the observable
        pass

    @abstractmethod
    def apply(self, config: BackendConfig, t: int, state: State, H: Operator) -> Any:
        pass
