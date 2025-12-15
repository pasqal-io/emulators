from __future__ import annotations

from functools import reduce

import torch

from typing import Sequence, Type

from emu_base import DEVICE_COUNT
from emu_sv.state_vector import StateVector

from pulser.backend import (
    Operator,
    State,
)
from pulser.backend.operator import FullOp, QuditOp
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


class SparseOperator(Operator[complex, torch.Tensor, StateVector]):
    """This operator is used to represent a sparse matrix in CSR (Compressed Sparse Row)
    format for efficient computation on the EMU-SV emulator

    Args:
        matrix (torch.Tensor): The CSR matrix representation of the operator.
        gpu (bool): Use GPU for computation. True uses the CPU if GPU not available.
    """

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

    def __add__(self, other: Operator) -> SparseOperator:
        """
        Element-wise addition of two SparseOperators.

        Args:
            other: a SparseOperator instance.

        Returns:
            A new SparseOperator representing the sum.
        """
        assert isinstance(
            other, SparseOperator
        ), "SparseOperator can only be added to another SparseOperator."
        # TODO: figure out a better algorithm.
        # self.matrix + other.matrix doesn't work on mac.
        return SparseOperator(
            sparse_add(
                self.matrix.to_sparse_coo(), other.matrix.to_sparse_coo()
            ).to_sparse_csr()
        )

    def __rmul__(self, scalar: complex) -> SparseOperator:
        """
        Scalar multiplication of the SparseOperator.

        Args:
            scalar: a number to scale the operator.

        Returns:
            A new SparseOperator scaled by the given scalar.
        """

        return SparseOperator(scalar * self.matrix)

    def __matmul__(self, other: Operator) -> SparseOperator:
        """
        Compose two SparseOperators via matrix multiplication.

        Args:
            other: a SparseOperator instance.

        Returns:
            A new SparseOperator representing the product `self @ other`.
        """
        raise NotImplementedError()

    def apply_to(self, other: State) -> StateVector:
        """
        Apply the SparseOperator to a given StateVector.

        Args:
            other: a StateVector instance.

        Returns:
            A new StateVector after applying the operator.
        """
        assert isinstance(
            other, StateVector
        ), "SparseOperator can only be applied to a StateVector."

        return StateVector(self.matrix @ other.vector)

    def expect(self, state: State) -> torch.Tensor:
        """
        Compute the expectation value of the operator with respect to a state.

        Args:
            state: a StateVector instance.

        Returns:
            The expectation value as a float or complex number.
        """
        assert isinstance(
            state, StateVector
        ), "Only expectation values of StateVectors are supported."

        return torch.vdot(state.vector, self.apply_to(state).vector).cpu()

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

        assert len(set(eigenstates)) == 2, "Only qubits are supported in EMU-SV."

        operators_with_tensors: dict[str, torch.Tensor | QuditOp] = dict()

        if set(eigenstates) == {"r", "g"}:
            # operators_with_tensors will now contain the basis for single qubit ops,
            # and potentially user defined strings in terms of {r, g} or {0, 1}
            operators_with_tensors |= {
                "gg": torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=dtype).to_sparse_coo(),
                "rg": torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=dtype).to_sparse_coo(),
                "gr": torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=dtype).to_sparse_coo(),
                "rr": torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=dtype).to_sparse_coo(),
            }
        elif set(eigenstates) == {"0", "1"}:
            raise NotImplementedError(
                "{'0','1'} basis is related to XY Hamiltonian, which is not implemented"
            )
        else:
            raise ValueError("An unsupported basis of eigenstates has been provided.")

        accum_res = torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.int32),
            torch.zeros(0, dtype=dtype),
            (2**n_qudits, 2**n_qudits),
        )
        for coeff, oper_torch_with_target_qubits in operations:

            def build_torch_operator_from_string(
                oper: QuditOp | torch.Tensor,
            ) -> torch.Tensor:
                if isinstance(oper, torch.Tensor):
                    return oper

                result: torch.Tensor = torch.zeros((2, 2), dtype=dtype).to_sparse_coo()
                for opstr, coeff in oper.items():
                    tensor = build_torch_operator_from_string(
                        operators_with_tensors[opstr]
                    )
                    operators_with_tensors[opstr] = tensor
                    result += tensor * coeff
                return result

            single_qubit_gates = [torch.eye(2, dtype=dtype).to_sparse_coo()] * n_qudits

            for operator_torch, target_qubits in oper_torch_with_target_qubits:
                factor = build_torch_operator_from_string(operator_torch)
                for target_qubit in target_qubits:
                    single_qubit_gates[target_qubit] = factor

            accum_res = sparse_add(
                accum_res, coeff * reduce(sparse_kron, single_qubit_gates)
            )

        return SparseOperator(accum_res.to_sparse_csr()), operations

    def __deepcopy__(self, memo: dict) -> SparseOperator:
        """torch CSR tensor does not deepcopy automatically"""
        cls = self.__class__
        result = cls(torch.clone(self.matrix), gpu=self.matrix.is_cuda)
        memo[id(self)] = result
        return result
