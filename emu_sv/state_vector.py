from __future__ import annotations

from collections import Counter
from typing import Any, Iterable
import math


from emu_base import State

import torch

dtype = torch.complex128


class StateVector(State):
    """
    Represents a quantum state vector in a computational basis.

    This class extends the `State` class to handle state vectors,
    providing various utilities for initialization, normalization,
    manipulation, and measurement. The state vector must have a length
    that is a power of 2, representing 2ⁿ basis states for n qubits.

    Attributes:
        vector (torch.Tensor): 1D tensor representation of a state vector.

    Methods:
        __init__(vector: torch.Tensor, gpu: bool = False):
            Initializes the state vector. Ensures the length is a power of 2.

        _normalize() -> None:
            Normalizes the state vector to ensure it represents a valid quantum state.

        zero(num_sites: int, gpu: bool = False) -> StateVector:
            Creates a zero uninitialized "state" vector for a specified number of qubits.
            Warning, this has no physical meaning as-is!

        make(num_sites: int, gpu: bool = False) -> StateVector:
            Creates a ground state vector |000...0> for a specified number of qubits.

        inner(other: State) -> torch.Tensor:
            Computes the inner product of the current state vector with another.

        sample(num_shots: int = 1000, p_false_pos: float = 0.0, p_false_neg: float = 0.0)
        -> Counter[str]:
            Samples measurement outcomes based on the state vector's probabilities.

        _index_to_bitstring(index: int) -> str:
            Converts an integer index to its corresponding bitstring representation.

        __add__(other: State) -> StateVector:
            Computes the sum of two state vectors.

        __rmul__(scalar: complex) -> StateVector:
            Scales the state vector by a scalar value.

        norm() -> torch.Tensor:
            Computes the norm of the state vector.

        __repr__() -> str:
            Returns a string representation of the state vector.

        from_state_string(
            *,
            basis: tuple[str],
            nqubits: int,
            strings: dict[str, complex],
            gpu: bool = False,
            **kwargs: Any
        ) -> StateVector:
            Constructs a state vector from a string-based representation of amplitudes.


    Static Methods:
        inner(left: StateVector, right: StateVector) -> torch.Tensor:
            Computes the inner product of two state vectors.
    """

    def __init__(
        self,
        vector: torch.Tensor,
        *,
        gpu: bool = False,
    ):
        # NOTE: this accepts also zero vectors.

        assert math.log2(
            len(vector)
        ).is_integer(), "The number of elements in the vector should be power of 2"

        device = "cuda" if gpu else "cpu"
        self.vector = vector.to(dtype=dtype, device=device)

    def _normalize(self) -> None:
        # NOTE: use this in the callbacks
        """Checks if the input is normalized or not"""
        norm_state = torch.linalg.vector_norm(self.vector)

        if not torch.allclose(norm_state, torch.tensor(1.0, dtype=torch.float64)):
            self.vector = self.vector / norm_state

    @classmethod
    def zero(cls, num_sites: int, gpu: bool = False) -> StateVector:
        """
        Returns a zero uninitialized "state" vector. Warning, this has no physical meaning as-is!
        The vector in the output StateVector instance has the shape (2,)*number of qubits

        Args:
            num_sites: the number of qubits
            gpu: whether gpu or cpu

        Example:
        -------
        >>> StateVector.zero(2)
        tensor([0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j], dtype=torch.complex128)
        """

        device = "cuda" if gpu else "cpu"
        vector = torch.zeros(2**num_sites, dtype=dtype, device=device)
        return cls(vector, gpu=gpu)

    @classmethod
    def make(cls, num_sites: int, gpu: bool = False) -> StateVector:
        """
        Returns a State vector in ground state |000..0>.
        The vector in the output of StateVector has the shape (2,)*number of qubits

        Args:
            num_sites: the number of qubits
            gpu: whether gpu or cpu

        Example:
        -------
        >>> StateVector.make(2)
        tensor([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j], dtype=torch.complex128)
        """

        result = cls.zero(num_sites=num_sites, gpu=gpu)
        result.vector[0] = 1.0
        return result

    def inner(self, other: State) -> float | complex:
        assert isinstance(
            other, StateVector
        ), "Other state also needs to be a StateVector"
        assert (
            self.vector.shape == other.vector.shape
        ), "States do not have the same number of sites"

        return torch.vdot(self.vector, other.vector).item()

    def sample(
        self, num_shots: int = 1000, p_false_pos: float = 0.0, p_false_neg: float = 0.0
    ) -> Counter[str]:
        """Probability distribution over measurement outcomes"""

        probabilities = torch.abs(self.vector) ** 2
        probabilities /= probabilities.sum()  # multinomial does not normalize the input

        outcomes = torch.multinomial(probabilities, num_shots, replacement=True)

        # Convert outcomes to bitstrings and count occurrences
        counts = Counter([self._index_to_bitstring(outcome) for outcome in outcomes])

        # NOTE: false positives and negatives
        return counts

    def _index_to_bitstring(self, index: int) -> str:
        """
        Convert an integer index into its corresponding bitstring representation.
        """
        nqubits = int(math.log2(self.vector.reshape(-1).shape[0]))
        return format(index, f"0{nqubits}b")

    def __add__(self, other: State) -> StateVector:
        """Sum of two state vectors"""
        assert isinstance(
            other, StateVector
        ), "Other state also needs to be a StateVector"
        result = self.vector + other.vector
        return StateVector(result)

    def __rmul__(self, scalar: complex) -> StateVector:
        """Scalar multiplication with a State vector"""
        result = scalar * self.vector

        return StateVector(result)

    def norm(self) -> float | complex:
        """Norm of the state"""
        norm: float | complex = torch.linalg.vector_norm(self.vector).item()
        return norm

    def __repr__(self) -> str:
        return repr(self.vector)

    @staticmethod
    def from_state_string(
        *,
        basis: Iterable[str],
        nqubits: int,
        strings: dict[str, complex],
        **kwargs: Any,
    ) -> StateVector:
        """Transforms a state given by a string into a state vector.

        Construct a state from the pulser abstract representation
        https://pulser.readthedocs.io/en/stable/conventions.html

        Args:
            basis: A tuple containing the basis states (e.g., ('r', 'g')).
            nqubits: the number of qubits.
            strings: A dictionary mapping state strings to complex or floats amplitudes.

        Returns:
            The resulting state.

        Example:
        -------
        >>> basis = ("r","g")
        >>> n = 2
        >>> st=StateVector.from_state_string(basis=basis,nqubits=n,strings={"rr":1.0,"gg":1.0})
        >>> print(st)
        tensor([0.7071+0.j, 0.0000+0.j, 0.0000+0.j, 0.7071+0.j],
               dtype=torch.complex128)
        """

        basis = set(basis)
        if basis == {"r", "g"}:
            one = "r"
        elif basis == {"0", "1"}:
            one = "1"
        else:
            raise ValueError("Unsupported basis provided")

        accum_state = StateVector.zero(num_sites=nqubits, **kwargs)

        for state, amplitude in strings.items():
            bin_to_int = int(
                state.replace(one, "1").replace("g", "0"), 2
            )  # "0" basis is already in "0"
            accum_state.vector[bin_to_int] = torch.tensor([amplitude])

        accum_state._normalize()

        return accum_state


def inner(left: StateVector, right: StateVector) -> torch.Tensor:
    """
    Wrapper around StateVector.inner.

    Args:
        left:  StateVector argument
        right: StateVector argument

    Returns:
        the inner product
    Example:
    -------
    >>> factor = math.sqrt(2.0)
    >>> basis = ("r","g")
    >>> nqubits = 2
    >>> string_state1 = {"gg":1.0,"rr":1.0}
    >>> state1 = StateVector.from_state_string(basis=basis, nqubits=nqubits,strings=string_state1)
    >>> string_state2 = {"gr":1.0/factor,"rr":1.0/factor}
    >>> state2 = StateVector.from_state_string(basis=basis,nqubits=nqubits,strings=string_state2)
    >>> inner(state1,state2).item()
    (0.4999999999999999+0j)
    """

    assert (left.vector.shape == right.vector.shape) and (
        left.vector.dim() == 1
    ), "Shape of a and b should be the same and both needs to be 1D tesnor"
    return torch.inner(left.vector, right.vector)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
