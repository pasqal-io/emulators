from __future__ import annotations

import torch

from typing import Sequence, Type

from emu_sv.matrix_operator import MatrixOperator

from pulser.backend.operator import FullOp
from pulser.backend.state import Eigenstate

dtype = torch.complex128


def sparse_add(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return torch.sparse_coo_tensor(
        torch.cat((self.indices(), other.indices()), dim=1),
        torch.cat((self.values(), other.values())),
        size=self.shape,
    ).coalesce()


def sparse_kron(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = a.coalesce(), b.coalesce()
    sa, sb = a.shape, b.shape
    shape = (sa[0] * sb[0], sa[1] * sb[1])
    i = (
        torch.tensor(sb).reshape(2, 1, 1) * a.indices().reshape(2, -1, 1)
        + b.indices().reshape(2, 1, -1)
    ).reshape(2, -1)
    v = torch.outer(a.values(), b.values()).flatten()
    return torch.sparse_coo_tensor(i, v, shape, is_coalesced=True)


class SparseOperator(MatrixOperator):
    """This operator is used to represent a sparse matrix in CSR (Compressed
    Sparse Row) format for efficient computation on the emu-sv emulator

    Args:
        matrix (torch.Tensor): The CSR matrix representation of the operator.

        gpu (bool): If True (by default), run on GPU when available; otherwise
            fall back to CPU. If False, always run on CPU.
    """

    @classmethod
    def _from_operator_repr(
        cls: Type[SparseOperator],
        *,
        eigenstates: Sequence[Eigenstate],
        n_qudits: int,
        operations: FullOp[complex],
    ) -> tuple[SparseOperator, FullOp[complex]]:
        """
        Construct a SparseOperator from an operator representation.

        Args:
            eigenstates: the eigenstates of the basis to use, e.g. ("r", "g") or ("0", "1").
            n_qudits: number of qudits in the system.
            operations: which bitstrings make up the state with what weight.

        Returns:
            A SparseOperator instance corresponding to the given representation.
        """
        matrix = cls._build(
            eigenstates=eigenstates,
            n_qudits=n_qudits,
            operations=operations,
            sparse=True,
        )
        return SparseOperator(matrix.to_sparse_csr()), operations
