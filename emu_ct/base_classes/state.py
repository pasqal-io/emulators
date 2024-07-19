from __future__ import annotations
from abc import ABC, abstractmethod
from collections import Counter
from numbers import Number

"""
note that self is the left state in the inner product,
so this function is linear in other, and anti-linear in self
"""


class State(ABC):
    @abstractmethod
    def inner(self, other: State) -> float | complex:
        pass

    @abstractmethod
    def sample(self, num_shots: int) -> Counter[str]:
        pass

    @abstractmethod
    def __add__(self, other: State) -> State:
        pass

    @abstractmethod
    def __rmul__(self, scalar: Number) -> State:
        pass
