from __future__ import annotations
import itertools
from typing import Any, Iterable

import torch
from emu_base.base_classes.operator import FullOp, QuditOp
from emu_base import Operator, State, DEVICE_COUNT
from emu_sv.state_vector import StateVector

dtype = torch.complex128


def _validate_operator_targets(operations: FullOp, nqubits: int) -> None:
    """Check for `operator_for_string` method"""
    for tensorop in operations:
        target_qids = (factor[1] for factor in tensorop[1])
        target_qids_list = list(itertools.chain(*target_qids))
        target_qids_set = set(target_qids_list)
        if len(target_qids_set) < len(target_qids_list):
            # Either the qubit id has been defined twice in an operation:
            for qids in target_qids:
                if len(set(qids)) < len(qids):
                    raise ValueError("Duplicate atom ids in argument list.")
            # Or it was defined in two different operations
            raise ValueError("Each qubit can be targeted by only one operation.")
        if max(target_qids_set) >= nqubits:
            raise ValueError(
                "The operation targets more qubits than there are in the register."
            )


class DenseOperator(Operator):
    """Operators in EMU-SV are dense matrices"""

    def __init__(
        self,
        matrix: torch.Tensor,
        *,
        gpu: bool = True,
    ):
        device = "cuda" if gpu and DEVICE_COUNT > 0 else "cpu"
        self.matrix = matrix.to(dtype=dtype, device=device)

    def __repr__(self) -> str:
        return repr(self.matrix)

    def __matmul__(self, other: Operator) -> DenseOperator:
        """
        Apply this operator to a other. The ordering is that
        self is applied after other.

        Args:
            other: the operator to compose with self

        Returns:
            the composed operator
        """
        assert isinstance(
            other, DenseOperator
        ), "DenseOperator can only be multiplied with Operator"

        return DenseOperator(self.matrix @ other.matrix)

    def __add__(self, other: Operator) -> DenseOperator:
        """
        Returns the sum of two matrices

        Args:
            other: the other operator

        Returns:
            the summed operator
        """
        assert isinstance(other, DenseOperator), "MPO can only be added to another MPO"

        return DenseOperator(self.matrix + other.matrix)

    def __rmul__(self, scalar: complex) -> DenseOperator:
        """
        Multiply a DenseOperator by scalar.

        Args:
            scalar: the scale factor to multiply with

        Returns:
            the scaled MPO
        """

        return DenseOperator(self.matrix * scalar)

    def __mul__(self, other: State) -> StateVector:
        """
        Applies this DenseOperator to the given StateVector.

        Args:
            other: the state to apply this operator to

        Returns:
            the resulting state
        """
        assert isinstance(
            other, StateVector
        ), "DenseOperator can only be applied to another DenseOperator"

        return StateVector(self.matrix @ other.vector)

    def expect(self, state: State) -> float | complex:
        """
        Compute the expectation value of self on the given state.

        Args:
            state: the state with which to compute

        Returns:
            the expectation
        """
        assert isinstance(
            state, StateVector
        ), "currently, only expectation values of StateVectors are \
        supported"

        return torch.vdot(state.vector, self.matrix @ state.vector).item()

    @staticmethod
    def from_operator_string(
        basis: Iterable[str],
        nqubits: int,
        operations: FullOp,
        operators: dict[str, QuditOp] = {},
        /,
        **kwargs: Any,
    ) -> DenseOperator:
        """
        See the base class

        Args:
            basis: the eigenstates in the basis to use e.g. ('r', 'g')
            nqubits: how many qubits there are in the state
            operations: which bitstrings make up the state with what weight
            operators: additional symbols to be used in operations

        Returns:
            the operator in MPO form.
        """

        _validate_operator_targets(operations, nqubits)

        operators_with_tensors: dict[str, torch.Tensor | QuditOp] = dict(operators)

        basis = set(basis)
        if basis == {"r", "g"}:
            # operators_with_tensors will now contain the basis for single qubit ops,
            # and potentially user defined strings in terms of these
            operators_with_tensors |= {
                "gg": torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex128),
                "gr": torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.complex128),
                "rg": torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.complex128),
                "rr": torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.complex128),
            }
        elif basis == {"0", "1"}:
            # operators_with_tensors will now contain the basis for single qubit ops,
            # and potentially user defined strings in terms of these
            operators_with_tensors |= {
                "00": torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex128),
                "01": torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.complex128),
                "10": torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.complex128),
                "11": torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.complex128),
            }
        else:
            raise ValueError("Unsupported basis provided")

        accum_res = torch.zeros(2**nqubits, 2**nqubits, dtype=torch.complex128)
        for coeff, tensorop in operations:
            # this function will recurse through the operators_with_tensors,
            # and replace any definitions in terms of strings by the computed matrix
            def replace_operator_string(op: QuditOp | torch.Tensor) -> torch.Tensor:
                if isinstance(op, torch.Tensor):
                    return op

                result = torch.zeros(2, 2, dtype=torch.complex128)
                for opstr, coeff in op.items():
                    tensor = replace_operator_string(operators_with_tensors[opstr])
                    operators_with_tensors[opstr] = tensor
                    result += tensor * coeff
                return result

            total_op_per_qubit = [torch.eye(2, 2, dtype=torch.complex128)] * nqubits

            for op in tensorop:
                factor = replace_operator_string(op[0])
                for target_qubit in op[1]:
                    total_op_per_qubit[target_qubit] = factor

            dense_op = total_op_per_qubit[0]
            for single_qubit_operator in total_op_per_qubit[1:]:
                dense_op = torch.kron(dense_op, single_qubit_operator)

            accum_res += coeff * dense_op
        return DenseOperator(accum_res)
