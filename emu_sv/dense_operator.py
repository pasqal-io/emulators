from __future__ import annotations

import torch

from typing import Sequence, Type

from emu_sv.matrix_operator import MatrixOperator

from pulser.backend import Operator
from pulser.backend.operator import FullOp
from pulser.backend.state import Eigenstate

dtype = torch.complex128


class DenseOperator(MatrixOperator):
    """DenseOperator in emu-sv uses dense matrices. This class represents a
    quantum operator backed by a dense PyTorch tensor for state-vector
    simulation.

    Args:
        matrix (torch.Tensor): Square complex tensor of shape (2ⁿ, 2ⁿ)
            representing the operator in the computational basis.
        gpu (bool, optional): If True, place the operator on a CUDA device when
            available. Default: True (and only 1 GPU).
    """

    def __matmul__(self, other: Operator) -> DenseOperator:
        """
        Compose two DenseOperators via matrix multiplication.

        Args:
            other: a DenseOperator instance.

        Returns:
            A new DenseOperator representing the product `self @ other`.
        """
        assert isinstance(
            other, DenseOperator
        ), "DenseOperator can only be multiplied with a DenseOperator."
        return DenseOperator(self.data @ other.data)

    @classmethod
    def _from_operator_repr(
        cls: Type[DenseOperator],
        *,
        eigenstates: Sequence[Eigenstate],
        n_qudits: int,
        operations: FullOp[complex],
    ) -> tuple[DenseOperator, FullOp[complex]]:
        """
        Construct a DenseOperator from an operator representation.

        Args:
            eigenstates: the eigenstates of the basis to use, e.g. ("r", "g")
                or ("0", "1").
            n_qudits: number of qudits in the system.
            operations: which bitstrings make up the state with what weight.

        Returns:
            A DenseOperator instance corresponding to the given representation.
        """
        matrix = cls._build(
            eigenstates=eigenstates,
            n_qudits=n_qudits,
            operations=operations,
            sparse=False,
        )
        return DenseOperator(matrix), operations
