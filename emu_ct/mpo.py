from typing import Any
from .mps import MPS
import torch


class MPO:
    """
    Matrix Product Operator
    """

    def __init__(self, factors: list):
        self.factors = factors
        self.num_sites = len(factors)
        if not self.num_sites > 1:
            raise ValueError("For 1 qubit states, do state vector")

    def __repr__(self) -> str:
        result = "["
        for fac in self.factors:
            result += repr(fac)
            result += ", "
        result += "]"
        return result

    def __mul__(self, state: MPS) -> MPS:
        out_factors = []
        # to keep track of the bond dimensions of the output mps
        # so we can do the logic in a single sweep from either left-to-right or vice-versa
        left_bond_size = 1
        if state.orth_center == 0:
            for i in range(self.num_sites):
                factor, left_bond_size = self._contract_factors(
                    self.factors[i], state.factors[i], left_bond_size
                )
                out_factors.append(factor)
            mps = MPS(out_factors)
            return mps
        elif state.orth_center == self.num_sites - 1:
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    @staticmethod
    def _contract_factors(
        mpo_factor: Any, mps_factor: Any, left_bond_size: int
    ) -> tuple[Any, int]:
        factor = torch.einsum("ijkl,mko->imjlo", mpo_factor, mps_factor)
        factor = factor.reshape(left_bond_size, 2, -1)
        left_bond_size = factor.shape[-1]
        return factor, left_bond_size

    def __rmul__(self, state: MPS) -> MPS:
        return self * state
