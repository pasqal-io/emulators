from __future__ import annotations
import itertools
from typing import Any, List, cast, Iterable, Optional

import torch

from emu_mps.algebra import add_factors, scale_factors, zip_right
from emu_base.base_classes.operator import FullOp, QuditOp
from emu_base import Operator, State
from emu_mps.mps import MPS
from emu_mps.utils import new_left_bath, assign_devices, DEVICE_COUNT


def _validate_operator_targets(operations: FullOp, nqubits: int) -> None:
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


class MPO(Operator):
    """
    Matrix Product Operator.

    Each tensor has 4 dimensions ordered as such: (left bond, output, input, right bond).

    Args:
        factors: the tensors making up the MPO
    """

    def __init__(
        self, factors: List[torch.Tensor], /, num_gpus_to_use: Optional[int] = None
    ):
        self.factors = factors
        self.num_sites = len(factors)
        if not self.num_sites > 1:
            raise ValueError("For 1 qubit states, do state vector")
        if factors[0].shape[0] != 1 or factors[-1].shape[-1] != 1:
            raise ValueError(
                "The dimension of the left (right) link of the first (last) tensor should be 1"
            )
        assert all(
            factors[i - 1].shape[-1] == factors[i].shape[0]
            for i in range(1, self.num_sites)
        )

        if num_gpus_to_use is not None:
            assign_devices(self.factors, min(DEVICE_COUNT, num_gpus_to_use))

    def __repr__(self) -> str:
        return "[" + ", ".join(map(repr, self.factors)) + "]"

    def __mul__(self, other: State) -> MPS:
        """
        Applies this MPO to the given MPS.
        The returned MPS is:

            - othogonal on the first site
            - truncated up to `other.precision`
            - distributed on the same devices of `other`

        Args:
            other: the state to apply this operator to

        Returns:
            the resulting state
        """
        assert isinstance(other, MPS), "MPO can only be multiplied with MPS"
        factors = zip_right(
            self.factors,
            other.factors,
            max_error=other.precision,
            max_rank=other.max_bond_dim,
        )
        return MPS(factors, orthogonality_center=0)

    def __add__(self, other: Operator) -> MPO:
        """
        Returns the sum of two MPOs, computed with a direct algorithm.
        The result is currently not truncated

        Args:
            other: the other operator

        Returns:
            the summed operator
        """
        assert isinstance(other, MPO), "MPO can only be added to another MPO"
        sum_factors = add_factors(self.factors, other.factors)
        return MPO(sum_factors)

    def __rmul__(self, scalar: complex) -> MPO:
        """
        Multiply an MPO by scalar.
        Assumes the orthogonal centre is on the first factor.

        Args:
            scalar: the scale factor to multiply with

        Returns:
            the scaled MPO
        """
        factors = scale_factors(self.factors, scalar, which=0)
        return MPO(factors)

    def __matmul__(self, other: Operator) -> MPO:
        """
        Compose two operators. The ordering is that
        self is applied after other.

        Args:
            other: the operator to compose with self

        Returns:
            the composed operator
        """
        assert isinstance(other, MPO), "MPO can only be applied to another MPO"
        factors = zip_right(self.factors, other.factors)
        return MPO(factors)

    def expect(self, state: State) -> float | complex:
        """
        Compute the expectation value of self on the given state.

        Args:
            state: the state with which to compute

        Returns:
            the expectation
        """
        assert isinstance(
            state, MPS
        ), "currently, only expectation values of MPSs are \
        supported"
        acc = torch.ones(
            1, 1, 1, dtype=state.factors[0].dtype, device=state.factors[0].device
        )
        n = len(self.factors) - 1
        for i in range(n):
            acc = new_left_bath(acc, state.factors[i], self.factors[i]).to(
                state.factors[i + 1].device
            )
        acc = new_left_bath(acc, state.factors[n], self.factors[n])
        return acc.item()  # type: ignore [no-any-return]

    @staticmethod
    def from_operator_string(
        basis: Iterable[str],
        nqubits: int,
        operations: FullOp,
        operators: dict[str, QuditOp] = {},
        /,
        **kwargs: Any,
    ) -> MPO:
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

        basis = set(basis)
        if basis == {"r", "g"}:
            # operators will now contain the basis for single qubit ops, and potentially
            # user defined strings in terms of these
            operators |= {
                "gg": torch.tensor(
                    [[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex128
                ).reshape(1, 2, 2, 1),
                "gr": torch.tensor(
                    [[0.0, 0.0], [1.0, 0.0]], dtype=torch.complex128
                ).reshape(1, 2, 2, 1),
                "rg": torch.tensor(
                    [[0.0, 1.0], [0.0, 0.0]], dtype=torch.complex128
                ).reshape(1, 2, 2, 1),
                "rr": torch.tensor(
                    [[0.0, 0.0], [0.0, 1.0]], dtype=torch.complex128
                ).reshape(1, 2, 2, 1),
            }
        elif basis == {"0", "1"}:
            # operators will now contain the basis for single qubit ops, and potentially
            # user defined strings in terms of these
            operators |= {
                "00": torch.tensor(
                    [[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex128
                ).reshape(1, 2, 2, 1),
                "01": torch.tensor(
                    [[0.0, 0.0], [1.0, 0.0]], dtype=torch.complex128
                ).reshape(1, 2, 2, 1),
                "10": torch.tensor(
                    [[0.0, 1.0], [0.0, 0.0]], dtype=torch.complex128
                ).reshape(1, 2, 2, 1),
                "11": torch.tensor(
                    [[0.0, 0.0], [0.0, 1.0]], dtype=torch.complex128
                ).reshape(1, 2, 2, 1),
            }
        else:
            raise ValueError("Unsupported basis provided")

        mpos = []
        for coeff, tensorop in operations:
            # this function will recurse through the operators, and replace any definitions
            # in terms of strings by the computed tensor
            def replace_operator_string(op: QuditOp | torch.Tensor) -> torch.Tensor:
                if isinstance(op, dict):
                    for opstr, coeff in op.items():
                        tensor = replace_operator_string(operators[opstr])
                        operators[opstr] = tensor
                        op[opstr] = tensor * coeff
                    op = sum(cast(list[torch.Tensor], op.values()))
                return op

            factors = [
                torch.eye(2, 2, dtype=torch.complex128).reshape(1, 2, 2, 1)
            ] * nqubits

            for i, op in enumerate(tensorop):
                tensorop[i] = (replace_operator_string(op[0]), op[1])

            for op in tensorop:
                for i in op[1]:
                    factors[i] = op[0]
            mpos.append(coeff * MPO(factors, **kwargs))
        return sum(mpos[1:], start=mpos[0])
