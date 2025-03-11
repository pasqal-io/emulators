from __future__ import annotations

from collections import Counter
from typing import Any, Iterable
import math


from emu_base import State, DEVICE_COUNT

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
        vector: 1D tensor representation of a state vector.
        gpu: store the vector on GPU if True, otherwise on CPU
    """

    def __init__(
        self,
        vector: torch.Tensor,
        *,
        gpu: bool = True,
    ):
        # NOTE: this accepts also zero vectors.

        assert math.log2(
            len(vector)
        ).is_integer(), "The number of elements in the vector should be power of 2"

        device = "cuda" if gpu and DEVICE_COUNT > 0 else "cpu"
        self.vector = vector.to(dtype=dtype, device=device)

    def _normalize(self) -> None:
        # NOTE: use this in the callbacks
        """Checks if the input is normalized or not"""
        norm_state = torch.linalg.vector_norm(self.vector)

        if not torch.allclose(norm_state, torch.tensor(1.0, dtype=torch.float64)):
            self.vector = self.vector / norm_state

    @classmethod
    def zero(cls, num_sites: int, gpu: bool = True) -> StateVector:
        """
        Returns a zero uninitialized "state" vector. Warning, this has no physical meaning as-is!

        Args:
            num_sites: the number of qubits
            gpu: whether gpu or cpu

        Returns:
            The zero state

        Examples:
            >>> StateVector.zero(2)
            tensor([0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j], dtype=torch.complex128)
        """

        device = "cuda" if gpu and DEVICE_COUNT > 0 else "cpu"
        vector = torch.zeros(2**num_sites, dtype=dtype, device=device)
        return cls(vector, gpu=gpu)

    @classmethod
    def make(cls, num_sites: int, gpu: bool = True) -> StateVector:
        """
        Returns a State vector in ground state |000..0>.
        The vector in the output of StateVector has the shape (2,)*number of qubits

        Args:
            num_sites: the number of qubits
            gpu: whether gpu or cpu

        Returns:
            The described state

        Examples:
            >>> StateVector.make(2)
            tensor([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j], dtype=torch.complex128)
        """

        result = cls.zero(num_sites=num_sites, gpu=gpu)
        result.vector[0] = 1.0
        return result

    def inner(self, other: State) -> float | complex:
        """
        Compute <self, other>. The type of other must be StateVector.

        Args:
            other: the other state

        Returns:
            the inner product
        """
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
        """
        Samples bitstrings, taking into account the specified error rates.

        Args:
            num_shots: how many bitstrings to sample
            p_false_pos: the rate at which a 0 is read as a 1
            p_false_neg: teh rate at which a 1 is read as a 0

        Returns:
            the measured bitstrings, by count
        """

        probabilities = torch.abs(self.vector) ** 2

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
        """Sum of two state vectors

        Args:
            other: the vector to add to this vector

        Returns:
            The summed state
        """
        assert isinstance(
            other, StateVector
        ), "Other state also needs to be a StateVector"
        result = self.vector + other.vector
        return StateVector(result)

    def __rmul__(self, scalar: complex) -> StateVector:
        """Scalar multiplication

        Args:
            scalar: the scalar to multiply with

        Returns:
            The scaled state
        """
        result = scalar * self.vector

        return StateVector(result)

    def norm(self) -> float | complex:
        """Returns the norm of the state

        Returns:
            the norm of the state
        """
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

        Examples:
            >>> basis = ("r","g")
            >>> n = 2
            >>> st=StateVector.from_state_string(basis=basis,nqubits=n,strings={"rr":1.0,"gg":1.0})
            >>> print(st)
            tensor([0.7071+0.j, 0.0000+0.j, 0.0000+0.j, 0.7071+0.j], dtype=torch.complex128)
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

    Examples:
        >>> factor = math.sqrt(2.0)
        >>> basis = ("r","g")
        >>> nqubits = 2
        >>> string_state1 = {"gg":1.0,"rr":1.0}
        >>> state1 = StateVector.from_state_string(basis=basis,
        >>>     nqubits=nqubits,strings=string_state1)
        >>> string_state2 = {"gr":1.0/factor,"rr":1.0/factor}
        >>> state2 = StateVector.from_state_string(basis=basis,
        >>>     nqubits=nqubits,strings=string_state2)
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
