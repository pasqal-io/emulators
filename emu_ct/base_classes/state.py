from __future__ import annotations
from typing import Any
from abc import ABC, abstractmethod
from collections import Counter

"""
note that self is the left state in the inner product,
so this function is linear in other, and anti-linear in self
"""


class State(ABC):
    @abstractmethod
    def inner(self, other: State) -> float | complex:
        pass

    @abstractmethod
    def sample(
        self, num_shots: int, p_false_pos: float = 0.0, p_false_neg: float = 0.0
    ) -> Counter[str]:
        pass

    @abstractmethod
    def __add__(self, other: State) -> State:
        pass

    @abstractmethod
    def __rmul__(self, scalar: complex) -> State:
        pass

    @staticmethod
    @abstractmethod
    def from_state_string(
        *, basis: tuple[str], nqubits: int, strings: dict[str, complex], **kwargs: Any
    ) -> State:
        pass
