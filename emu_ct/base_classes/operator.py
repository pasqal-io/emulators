from __future__ import annotations

from abc import ABC, abstractmethod

from emu_ct.base_classes.state import State


class Operator(ABC):
    @abstractmethod
    def __mul__(self, state: State) -> State:
        pass
