from __future__ import annotations

from typing import Any
from abc import ABC, abstractmethod
from emu_ct.base_classes.state import State

QuditOp = dict[str, complex]  # single qubit operator
TensorOp = list[tuple[QuditOp, list[int]]]  # QuditOp applied to list of qubits
FullOp = list[tuple[complex, TensorOp]]  # weighted sum of TensorOp


class Operator(ABC):
    @abstractmethod
    def __mul__(self, other: State) -> State:
        pass

    @abstractmethod
    def __add__(self, other: Operator) -> Operator:
        pass

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
        create the operator operators[0] otimes operators[1] ...
        where the identity is filled in for qubits not targeted in any qubits
        dictionary
        """
        pass

    @abstractmethod
    def __rmul__(self, scalar: complex) -> Operator:
        pass
