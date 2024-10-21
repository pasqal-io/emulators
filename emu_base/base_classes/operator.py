from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from emu_base.base_classes.state import State


QuditOp = dict[str, complex]  # single qubit operator
TensorOp = list[tuple[QuditOp, list[int]]]  # QuditOp applied to list of qubits
FullOp = list[tuple[complex, TensorOp]]  # weighted sum of TensorOp


class Operator(ABC):
    @abstractmethod
    def __mul__(self, other: State) -> State:
        """
        Apply the operator to a state

        Args:
            other: the state to apply this operator to

        Returns:
            the resulting state
        """
        pass

    @abstractmethod
    def __add__(self, other: Operator) -> Operator:
        """
        Computes the sum of two operators.

        Args:
            other: the other operator

        Returns:
           the summed operator
        """
        pass

    @abstractmethod
    def expect(self, state: State) -> float | complex:
        """
        Compute the expectation value of self on the given state.

        Args:
            state: the state with which to compute

        Returns:
            the expectation
        """

    @staticmethod
    @abstractmethod
    def from_operator_string(
        basis: tuple[str, ...],
        nqubits: int,
        operations: FullOp,
        operators: dict[str, QuditOp] = {},
        /,
        **kwargs: Any,
    ) -> Operator:
        """
        Create an operator in the backend-specific format from the
        pulser abstract representation
        https://www.notion.so/pasqal/Abstract-State-and-Operator-Definition
        by default it supports strings 'ij', where i and j in basis,
        to denote |i><j|, but additional symbols can be defined in operators

        Args:
            basis: the eigenstates in the basis to use
            nqubits: how many qubits there are in the state
            operations: which bitstrings make up the state with what weight
            operators: additional symbols to be used in operations

        Returns:
            the operator in whatever format the backend provides.
        """
        pass

    @abstractmethod
    def __rmul__(self, scalar: complex) -> Operator:
        """
        Scale the operator by a scale factor.

        Args:
            scalar: the scale factor

        Returns:
            the scaled operator
        """
        pass

    @abstractmethod
    def __matmul__(self, other: Operator) -> Operator:
        """
        Compose two operators. The ordering is that
        self is applied after other.

        Args:
            other: the operator to compose with self

        Returns:
            the composed operator
        """
        pass
