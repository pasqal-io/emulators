from __future__ import annotations
from typing import Any

from dataclasses import dataclass
from abc import ABC, abstractmethod
from pulser.register.base_register import QubitId

from emu_ct.base_classes.state import State

"""
This represents the single qubit operator sum_i coefficients_i*operators_i
applied to each qubit in targets.
The strings supported in operators are 'sigma_ij' which represents |i><j|
see Operator.from_operator_string for an addendum
"""


@dataclass
class OperatorString:
    coefficients: list[float]
    operators: list[str]


TargetedOperatorString = tuple[OperatorString, set[str]]


class Operator(ABC):
    @abstractmethod
    def __mul__(self, other: State) -> State:
        pass

    @abstractmethod
    def __add__(self, other: Operator) -> Operator:
        pass

    """
    create the operator operators[0] otimes operators[1] ...
    where the identity is filled in for qubits not targeted in any qubits
    dictionary
    """

    @staticmethod
    @abstractmethod
    def from_operator_string(
        basis: tuple[str],
        qubits: list[QubitId],
        operations: list[TargetedOperatorString],
        operators: dict[str, OperatorString] = {},
        /,
        **kwargs: Any,
    ) -> Operator:
        pass
