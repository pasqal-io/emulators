from __future__ import annotations
from typing import Any
from abc import ABC, abstractmethod
from collections import Counter


class State(ABC):
    """
    Base class enforcing an API for quantum states.
    Each backend will implement its own type of state, and the
    below methods.
    """

    @abstractmethod
    def inner(self, other: State) -> float | complex:
        """
        Compute the inner product between this state and other.
        Note that self is the left state in the inner product,
        so this function is linear in other, and anti-linear in self

        Args:
            other: the other state

        Returns:
            inner product
        """
        pass

    @abstractmethod
    def sample(
        self, num_shots: int, p_false_pos: float = 0.0, p_false_neg: float = 0.0
    ) -> Counter[str]:
        """
        Sample bitstrings from the state, taking into account error rates.

        Args:
            num_shots: how many bitstrings to sample
            p_false_pos: the rate at which a 0 is read as a 1
            p_false_neg: the rate at which a 1 is read as a 0

        Returns:
            the measured bitstrings, by count
        """
        pass

    @abstractmethod
    def __add__(self, other: State) -> State:
        """
        Computes the sum of two states.

        Args:
            other: the other state

        Returns:
            the summed state
        """
        pass

    @abstractmethod
    def __rmul__(self, scalar: complex) -> State:
        """
        Scale the state by a scale factor.

        Args:
            scalar: the scale factor

        Returns:
            the scaled state
        """
        pass

    @staticmethod
    @abstractmethod
    def from_state_string(
        *, basis: tuple[str], nqubits: int, strings: dict[str, complex], **kwargs: Any
    ) -> State:
        """
        Construct a state from the pulser abstract representation
        https://pulser.readthedocs.io/en/stable/conventions.html

        Args:
            basis: A tuple containing the basis states (e.g., ('r', 'g')).
            nqubits: the number of qubits.
            strings: A dictionary mapping state strings to complex or floats amplitudes

        Returns:
            the state in whatever format the backend provides.
        """
        pass
