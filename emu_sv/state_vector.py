from __future__ import annotations

import math
from collections import Counter
from typing import Sequence, Type, TypeVar, Mapping

import torch

from emu_sv.utils import index_to_bitstring

from emu_base import DEVICE_COUNT, apply_measurement_errors
from pulser.backend import State
from pulser.backend.state import Eigenstate

StateVectorType = TypeVar("StateVectorType", bound="StateVector")
# Default tensor data type
dtype = torch.complex128


class StateVector(State[complex, torch.Tensor]):
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
        eigenstates: Sequence[Eigenstate] = ("r", "g"),
    ):
        # NOTE: this accepts also zero vectors.

        assert math.log2(
            len(vector)
        ).is_integer(), "The number of elements in the vector should be power of 2"

        super().__init__(eigenstates=eigenstates)
        device = "cuda" if gpu and DEVICE_COUNT > 0 else "cpu"
        self.vector = vector.to(dtype=dtype, device=device)

    @property
    def n_qudits(self) -> int:
        """The number of qudits in the state."""
        nqudits = math.log2(self.vector.view(-1).shape[0])
        return int(nqudits)

    def _normalize(self) -> None:
        """Normalizes the state vector to ensure it has unit norm.

        If the vector norm is not 1, the method scales the vector
        to enforce normalization.

        Note:
            This method is intended to be used in callbacks.
        """

        norm = torch.linalg.vector_norm(self.vector)

        if not torch.allclose(norm, torch.ones_like(norm)):
            self.vector = self.vector / norm

    @classmethod
    def zero(
        cls,
        num_sites: int,
        gpu: bool = True,
        eigenstates: Sequence[Eigenstate] = ("r", "g"),
    ) -> StateVector:
        """
        Returns a zero uninitialized "state" vector. Warning, this has no physical meaning as-is!

        Args:
            num_sites: the number of qubits
            gpu: whether gpu or cpu

        Returns:
            The zero state

        Examples:
            >>> StateVector.zero(2,gpu=False)
            tensor([0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j], dtype=torch.complex128)
        """

        device = "cuda" if gpu and DEVICE_COUNT > 0 else "cpu"
        vector = torch.zeros(2**num_sites, dtype=dtype, device=device)
        return cls(vector, gpu=gpu, eigenstates=eigenstates)

    @classmethod
    def make(cls, num_sites: int, gpu: bool = True) -> StateVector:
        """
        Returns a State vector in the ground state |00..0>.

        Args:
            num_sites: the number of qubits
            gpu: whether gpu or cpu

        Returns:
            The described state

        Examples:
            >>> StateVector.make(2,gpu=False)
            tensor([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j], dtype=torch.complex128)


        """

        result = cls.zero(num_sites=num_sites, gpu=gpu)
        result.vector[0] = 1.0
        return result

    def inner(self, other: State) -> torch.Tensor:
        """
        Compute <self|other>. The type of other must be StateVector.

        Args:
            other: the other state

        Returns:
            the inner product
        """
        assert isinstance(other, StateVector), "Other state must be a StateVector"
        assert (
            self.vector.shape == other.vector.shape
        ), "States do not have the same shape"

        # by our internal convention inner and norm return to cpu
        return torch.vdot(self.vector, other.vector.to(self.vector.device)).cpu()

    def sample(
        self,
        *,
        num_shots: int = 1000,
        one_state: Eigenstate | None = None,
        p_false_pos: float = 0.0,
        p_false_neg: float = 0.0,
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
        counts = Counter(
            [index_to_bitstring(self.n_qudits, outcome) for outcome in outcomes]
        )

        if p_false_neg > 0 or p_false_pos > 0:
            counts = apply_measurement_errors(
                counts,
                p_false_pos=p_false_pos,
                p_false_neg=p_false_neg,
            )
        return counts

    def __add__(self, other: State) -> StateVector:
        """Sum of two state vectors

        Args:
            other: the vector to add to this vector

        Returns:
            The summed state
        """
        assert isinstance(other, StateVector), "`Other` state can only be a StateVector"
        assert (
            self.eigenstates == other.eigenstates
        ), f"`Other` state has basis {other.eigenstates} != {self.eigenstates}"
        return StateVector(
            self.vector + other.vector,
            gpu=self.vector.is_cuda,
            eigenstates=self.eigenstates,
        )

    def __rmul__(self, scalar: complex) -> StateVector:
        """Scalar multiplication

        Args:
            scalar: the scalar to multiply with

        Returns:
            The scaled state
        """
        return StateVector(
            scalar * self.vector,
            gpu=self.vector.is_cuda,
            eigenstates=self.eigenstates,
        )

    def norm(self) -> torch.Tensor:
        """Returns the norm of the state

        Returns:
            the norm of the state
        """
        nrm: torch.Tensor = torch.linalg.vector_norm(self.vector).cpu()
        return nrm

    def __repr__(self) -> str:
        return repr(self.vector)

    @classmethod
    def _from_state_amplitudes(
        cls: Type[StateVectorType],
        *,
        eigenstates: Sequence[Eigenstate],
        n_qudits: int,
        amplitudes: Mapping[str, complex],
    ) -> tuple[StateVector, Mapping[str, complex]]:
        """Transforms a state given by a string into a state vector.

        Construct a state from the pulser abstract representation
        https://pulser.readthedocs.io/en/stable/conventions.html

        Args:
            eigenstates: A tuple containing the basis states (e.g., ('r', 'g')).
            amplitudes: A dictionary mapping state strings to complex or floats amplitudes.

        Returns:
            The normalised resulting state.

        Examples:
            >>> basis = ("r","g")
            >>> n = 2
            >>> st=StateVector.from_state_string(basis=basis,
                ... nqubits=n,strings={"rr":1.0,"gg":1.0},gpu=False)
            >>> st = StateVector.from_state_amplitudes(
            ...     eigenstates=basis,
            ...     amplitudes={"rr": 1.0, "gg": 1.0}
            ... )
            >>> print(st)
            tensor([0.7071+0.j, 0.0000+0.j, 0.0000+0.j, 0.7071+0.j],
                   dtype=torch.complex128)
        """
        basis = set(eigenstates)
        if basis == {"r", "g"}:
            one = "r"
        elif basis == {"0", "1"}:
            raise NotImplementedError(
                "{'0','1'} basis is related to XY Hamiltonian, which is not implemented"
            )
        else:
            raise ValueError("Unsupported basis provided")

        accum_state = StateVector.zero(num_sites=n_qudits, eigenstates=eigenstates)

        for state, amplitude in amplitudes.items():
            bin_to_int = int(
                state.replace(one, "1").replace("g", "0"), 2
            )  # "0" basis is already in "0"
            accum_state.vector[bin_to_int] = torch.tensor([amplitude])

        accum_state._normalize()

        return accum_state, amplitudes

    def overlap(self, other: StateVector, /) -> torch.Tensor:
        return torch.abs(self.inner(other)) ** 2


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
        >>> string_state1 = {"gg":1.0,"rr":1.0}
        >>> state1 = StateVector.from_state_string(basis=basis,
            ... nqubits=nqubits,strings=string_state1)
        >>> string_state2 = {"gr":1.0/factor,"rr":1.0/factor}
        >>> state2 = StateVector.from_state_string(basis=basis,
            ... nqubits=nqubits,strings=string_state2)

        >>> state1 = StateVector.from_state_amplitudes(eigenstates=basis,
        ...     amplitudes=string_state1)
        >>> string_state2 = {"gr":1.0/factor,"rr":1.0/factor}
        >>> state2 = StateVector.from_state_amplitudes(eigenstates=basis,
        ...     amplitudes=string_state2)
        >>> inner(state1,state2).item()
        (0.49999999144286444+0j)
    """

    assert (left.vector.shape == right.vector.shape) and (left.vector.dim() == 1), (
        "Shape of left.vector and right.vector should be",
        " the same and both need to be 1D tesnor",
    )
    return left.inner(right)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
