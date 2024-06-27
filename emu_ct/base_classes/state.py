from __future__ import annotations
from abc import ABC, abstractmethod
from collections import Counter


class State(ABC):
    @abstractmethod
    def inner(self, other: State) -> float | complex:
        pass

    @abstractmethod
    def sample(self, num_shots: int) -> Counter[str]:
        pass
