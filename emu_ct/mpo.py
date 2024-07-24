from typing import Any, List, cast
import itertools
from emu_ct.mps import MPS
import torch
from emu_ct.base_classes.state import State
from emu_ct.base_classes.operator import Operator, OperatorString, TargetedOperatorString
from pulser.register.base_register import QubitId
import emu_ct.algebra


def _validate_qubit_ids(
    operations: list[TargetedOperatorString], qubit_ids: list[QubitId]
) -> None:
    target_qids = [operation[1] for operation in operations]
    assert "global" not in target_qids, "global is not supported yet."
    target_qids_list = list(itertools.chain(*target_qids))
    target_qids_set = set(target_qids_list)
    if len(target_qids_set) < len(target_qids_list):
        # Either the qubit id has been defined twice in an operation:
        for qids in target_qids:
            if len(set(qids)) < len(qids):
                raise ValueError("Duplicate atom ids in argument list.")
        # Or it was defined in two different operations
        raise ValueError("Each qubit can be targeted by only one operation.")
    if not target_qids_set.issubset(qubit_ids):
        raise ValueError("Invalid qubit names: " f"{target_qids_set - set(qubit_ids)}")


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
        result = "["
        for fac in self.factors:
            result += repr(fac)
            result += ", "
        result += "]"
        return result

    def __mul__(self, other: State) -> MPS:
        """
        Applies this MPO to the given MPS.
        """
        assert isinstance(other, MPS), "MPO can only be multiplied with MPS"
        assert (
            self.num_sites == other.num_sites
        ), "MPO and MPS don't have the same number of sites"

        # Move own factors to the same devices as the MPS's factors
        self.factors = [
            self.factors[i].to(other.factors[i].device) for i in range(self.num_sites)
        ]

        out_factors = []
        # to keep track of the bond dimensions of the output mps
        # so we can do the logic in a single sweep from either left-to-right or vice-versa
        left_bond_size = 1
        for i in range(self.num_sites):
            factor, left_bond_size = self._contract_factors(
                self.factors[i], other.factors[i], left_bond_size
            )
            out_factors.append(factor)
        mps = MPS(out_factors)
        return mps

    @staticmethod
    def _contract_factors(
        mpo_factor: Any, mps_factor: Any, left_bond_size: int
    ) -> tuple[Any, int]:
        factor = torch.einsum("ijkl,mko->imjlo", mpo_factor, mps_factor)
        factor = factor.reshape(left_bond_size, 2, -1)
        left_bond_size = factor.shape[-1]
        return factor, left_bond_size

    def __add__(self, other: Operator) -> Operator:
        """
        Returns the sum of two MPOs, computed with a direct algorithm.
        """
        assert isinstance(other, MPO), "MPO can only be added to another MPO"
        sum_factors = emu_ct.algebra._add_factors(self.factors, other.factors)
        return MPO(sum_factors)

    @staticmethod
    def from_operator_string(
        basis: tuple[str],
        qubits: list[QubitId],
        operations: list[TargetedOperatorString],
        operators: dict[str, OperatorString] = {},
        /,
        **kwargs: Any,
    ) -> Operator:
        assert set(basis) == {
            "r",
            "g",
        }, "only the rydberg-ground basis is currently supported"
        _validate_qubit_ids(operations, qubits)
        nqubits = len(qubits)

        id_to_idx: dict[str, int] = {}
        for i, name in enumerate(qubits):
            id_to_idx[name] = i
        operations_indexed = [
            (op[0], set([id_to_idx[target] for target in op[1]])) for op in operations
        ]

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
        def replace_operator_string(op: OperatorString | torch.Tensor) -> torch.Tensor:
            if isinstance(op, OperatorString):
                for i, factor in enumerate(op.operators):
                    tensor = replace_operator_string(operators[factor])
                    operators[factor] = tensor
                    op.operators[i] = tensor * op.coefficients[i]
                op = sum(cast(list[torch.Tensor], op.operators))
            return op

        factors = [torch.eye(2, 2, dtype=torch.complex128).reshape(1, 2, 2, 1)] * nqubits

        for i, op in enumerate(operations_indexed):
            operations_indexed[i] = (replace_operator_string(op[0]), op[1])

        for op in operations_indexed:
            for i in op[1]:
                factors[i] = op[0]
        return MPO(factors, **kwargs)
