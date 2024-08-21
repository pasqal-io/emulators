import itertools
from typing import Any, List, cast

import torch

from emu_mps.algebra import add_factors, mul_factors, zip_right
from emu_mps.base_classes.operator import FullOp, Operator, QuditOp
from emu_mps.base_classes.state import State
from emu_mps.mps import MPS


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
    """

    def __init__(
        self,
        factors: List[torch.Tensor],
        /,
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

    def __repr__(self) -> str:
        return "[" + ", ".join(map(repr, self.factors)) + "]"

    def __mul__(self, other: State) -> MPS:
        """
        Applies this MPO to the given MPS.

        The returned MPS is:
            - othogonal on the first site
            - truncated up to `other.precision`
            - distributed on the same devices of `other`
        """
        assert isinstance(other, MPS), "MPO can only be multiplied with MPS"
        factors = zip_right(
            self.factors,
            other.factors,
            max_error=other.precision,
            max_rank=other.max_bond_dim,
        )
        return MPS(factors)

    def __add__(self, other: Operator) -> Operator:
        """
        Returns the sum of two MPOs, computed with a direct algorithm.
        """
        assert isinstance(other, MPO), "MPO can only be added to another MPO"
        sum_factors = add_factors(self.factors, other.factors)
        return MPO(sum_factors)

    def __rmul__(self, scalar: complex) -> Operator:
        """
        Multiply an MPS by scalar.
        Assumes the MPS is orthogonalized on the site 0.
        """
        factors = mul_factors(self.factors, scalar)
        return MPO(factors)

    def __matmul__(self, other: Operator) -> Operator:
        """
        Returns the multiplication of two MPOs.
        """
        assert isinstance(other, MPO), "MPO can only be applied to another MPO"
        factors = zip_right(self.factors, other.factors)
        return MPO(factors)

    @staticmethod
    def from_operator_string(
        basis: tuple[str, ...],
        nqubits: int,
        operations: FullOp,
        operators: dict[str, QuditOp] = {},
        /,
        **kwargs: Any,
    ) -> Operator:
        assert set(basis) == {
            "r",
            "g",
        }, "only the rydberg-ground basis is currently supported"
        _validate_operator_targets(operations, nqubits)
        nqubits
        mpos = []
        for coeff, tensorop in operations:
            # operators will now contain the 'sigma_ij' elements defined, and potentially
            # user defined strings in terms of the 'sigma_ij'
            operators |= {
                "sigma_gg": torch.tensor(
                    [[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex128
                ).reshape(1, 2, 2, 1),
                "sigma_gr": torch.tensor(
                    [[0.0, 0.0], [1.0, 0.0]], dtype=torch.complex128
                ).reshape(1, 2, 2, 1),
                "sigma_rg": torch.tensor(
                    [[0.0, 1.0], [0.0, 0.0]], dtype=torch.complex128
                ).reshape(1, 2, 2, 1),
                "sigma_rr": torch.tensor(
                    [[0.0, 0.0], [0.0, 1.0]], dtype=torch.complex128
                ).reshape(1, 2, 2, 1),
            }

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
