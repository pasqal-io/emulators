from __future__ import annotations
from .state import State
from abc import ABC, abstractmethod


class Operator(ABC):
    @abstractmethod
    def __mul__(self, state: State) -> State:
        pass
