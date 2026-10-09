from __future__ import annotations

from functools import reduce

import torch

from typing import Sequence, Type, TypeVar

from emu_base import DEVICE_COUNT
from emu_sv.state_vector import StateVector

from pulser.backend import (
    Operator,
    State,
)
from pulser.backend.operator import FullOp, QuditOp
from pulser.backend.state import Eigenstate

dtype = torch.complex128
MatrixOperatorType = TypeVar("MatrixOperatorType", bound="MatrixOperator")

# above this number of qubits, a dense matrix no longer fits in memory
MAX_DENSE_QUBITS = 14


class MatrixOperator(Operator[complex, torch.Tensor, StateVector]):
    """Base class for the emu-sv operators, which are all backed by a 2ⁿ x 2ⁿ
    torch tensor, either dense (`DenseOperator`) or sparse (`SparseOperator`).

    Args:
        matrix (torch.Tensor): Square complex tensor of shape (2ⁿ, 2ⁿ)
            representing the operator in the computational basis.
        gpu (bool, optional): If True, place the operator on a CUDA device when
            available. Default: True (and only 1 GPU).
    """

    def __init__(
        self,
        matrix: torch.Tensor,
        *,
        gpu: bool = True,
    ):
        super().__init__()
        device = "cuda" if gpu and DEVICE_COUNT > 0 else "cpu"
        self.data = matrix.to(dtype=dtype, device=device)

    def __repr__(self) -> str:
        return repr(self.data)

    def __rmul__(self: MatrixOperatorType, scalar: complex) -> MatrixOperatorType:
        """
        Scalar multiplication of the operator.

        Args:
            scalar: a number to scale the operator.

        Returns:
            A new operator of the same type, scaled by the given scalar.
        """

        return type(self)(scalar * self.data, gpu=self.data.is_cuda)

    def __add__(self, other: Operator) -> MatrixOperator:
        """
        Element-wise addition of two operators.

        Args:
            other: an operator of the same type.

        Returns:
            A new operator representing the sum.
        """
        t = type(self)
        assert isinstance(
            other, t
        ), f"Operator of type {t} can only be added to another of the same type."
        if self.data.is_sparse_csr:
            from emu_sv.sparse_operator import sparse_add

            # TODO: figure out a better algorithm.
            # self.data + other.data doesn't work on mac.
            return MatrixOperator(
                sparse_add(
                    self.data.to_sparse_coo(), other.data.to_sparse_coo()
                ).to_sparse_csr()
            )
        else:
            return t(self.data + other.data)

    def __matmul__(self, other: Operator) -> MatrixOperator:
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
        Apply the operator to a given StateVector.

        Args:
            other: a StateVector instance.

        Returns:
            A new StateVector after applying the operator.
        """
        assert isinstance(
            other, StateVector
        ), "emu-sv operators can only be applied to a StateVector."

        return StateVector(self.data @ other.data)

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

        return torch.vdot(state.data, self.apply_to(state).data).cpu()

    @classmethod
    def _build(
        cls,
        *,
        eigenstates: Sequence[Eigenstate],
        n_qudits: int,
        operations: FullOp[complex],
        sparse: bool,
    ) -> torch.Tensor:
        """
        Build the matrix of an operator representation.

        Args:
            eigenstates: the eigenstates of the basis to use, e.g. ("r", "g")
                or ("0", "1").
            n_qudits: number of qudits in the system.
            operations: which bitstrings make up the state with what weight.
            sparse: whether to accumulate in sparse COO format rather than dense.

        Returns:
            The matrix, as a dense tensor or as a sparse COO tensor.
        """
        from emu_sv.sparse_operator import sparse_add, sparse_kron

        assert len(set(eigenstates)) == 2, "Only qubits are supported in EMU-SV."

        operators_with_tensors: dict[str, torch.Tensor | QuditOp] = dict()

        if set(eigenstates) == {"r", "g"}:
            # operators_with_tensors will now contain the basis for single qubit ops,
            # and potentially user defined strings in terms of {r, g} or {0, 1}
            basis = {
                "gg": torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=dtype),
                "rg": torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=dtype),
                "gr": torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=dtype),
                "rr": torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=dtype),
            }
            operators_with_tensors |= {
                opstr: tensor.to_sparse_coo() if sparse else tensor
                for opstr, tensor in basis.items()
            }
        elif set(eigenstates) == {"0", "1"}:
            raise NotImplementedError(
                "{'0','1'} basis is related to XY Hamiltonian, which is not implemented"
            )
        else:
            raise ValueError("An unsupported basis of eigenstates has been provided.")

        zero_qubit_op = torch.zeros((2, 2), dtype=dtype)
        identity = torch.eye(2, dtype=dtype)
        accum_res: torch.Tensor

        if sparse:
            zero_qubit_op = zero_qubit_op.to_sparse_coo()
            identity = identity.to_sparse_coo()
            accum_res = torch.sparse_coo_tensor(
                torch.zeros(2, 0, dtype=torch.int32),
                torch.zeros(0, dtype=dtype),
                (2**n_qudits, 2**n_qudits),
            )
        else:
            accum_res = torch.zeros(2**n_qudits, 2**n_qudits, dtype=dtype)

        for coeff, oper_torch_with_target_qubits in operations:

            def build_torch_operator_from_string(
                oper: QuditOp | torch.Tensor,
            ) -> torch.Tensor:
                if isinstance(oper, torch.Tensor):
                    return oper

                result = zero_qubit_op.clone()
                for opstr, coeff in oper.items():
                    tensor = build_torch_operator_from_string(
                        operators_with_tensors[opstr]
                    )
                    operators_with_tensors[opstr] = tensor
                    result += tensor * coeff
                return result

            single_qubit_gates = [identity] * n_qudits

            for operator_torch, target_qubits in oper_torch_with_target_qubits:
                factor = build_torch_operator_from_string(operator_torch)
                for target_qubit in target_qubits:
                    single_qubit_gates[target_qubit] = factor

            if sparse:
                accum_res = sparse_add(
                    accum_res, coeff * reduce(sparse_kron, single_qubit_gates)
                )
            else:
                accum_res += coeff * reduce(torch.kron, single_qubit_gates)

        return accum_res

    @classmethod
    def _from_operator_repr(
        cls: Type[MatrixOperator],
        *,
        eigenstates: Sequence[Eigenstate],
        n_qudits: int,
        operations: FullOp[complex],
    ) -> tuple[MatrixOperator, FullOp[complex]]:
        """
        Construct an operator from an operator representation, choosing the dense
        representation for small systems and the sparse one for large systems.

        Args:
            eigenstates: the eigenstates of the basis to use, e.g. ("r", "g")
                or ("0", "1").
            n_qudits: number of qudits in the system.
            operations: which bitstrings make up the state with what weight.

        Returns:
            A DenseOperator for at most 14 qudits, a SparseOperator otherwise.
        """

        matrix = cls._build(
            eigenstates=eigenstates,
            n_qudits=n_qudits,
            operations=operations,
            sparse=n_qudits > 14,
        )
        return MatrixOperator(matrix=matrix), operations

    def __deepcopy__(self, memo: dict) -> MatrixOperator:
        """torch CSR tensor does not deepcopy automatically"""
        cls = self.__class__
        result = cls(torch.clone(self.data), gpu=self.data.is_cuda)
        result._eigenstates = self._eigenstates
        result._n_qudits = self._n_qudits
        result._operations = self._operations
        memo[id(self)] = result
        return result
