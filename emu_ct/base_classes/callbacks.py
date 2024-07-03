from abc import ABC, abstractmethod
from typing import Any
from emu_ct.base_classes.state import State
from emu_ct.base_classes.results import Results
from emu_ct.base_classes.operator import Operator
from copy import deepcopy


class Callback(ABC):
    def __init__(self, times: list[int]):
        self.times = times

    def __call__(self, t: int, state: State, H: Operator, result: Results) -> None:
        if t in self.times:
            result[self.name()][t] = self.apply(t, state, H)

    @staticmethod
    @abstractmethod
    def name() -> str:
        # return the name of the observable
        pass

    @abstractmethod
    def apply(self, t: int, state: State, H: Operator) -> Any:
        pass


class StateResult(Callback):
    def __init__(self, times: list[int]):
        super().__init__(times)

    @staticmethod
    def name() -> str:
        return "state"

    def apply(self, t: int, state: State, H: Operator) -> Any:
        return deepcopy(state)


class BitStrings(Callback):
    def __init__(self, times: list[int], num_shots: int = 1000):
        super().__init__(times)
        self.num_shots = num_shots

    @staticmethod
    def name() -> str:
        return "bitstrings"

    def apply(self, t: int, state: State, H: Operator) -> Any:
        return state.sample(self.num_shots)
