from typing import Any, List
from emu_ct.mps import MPS
from emu_ct.utils import assign_devices, DEVICE_COUNT
import torch
from .base_classes.state import State
from .base_classes.operator import Operator


class MPO(Operator):
    """
    Matrix Product Operator
    """

    def __init__(
        self,
        factors: List[torch.Tensor],
        num_devices_to_use: int = DEVICE_COUNT,
    ):
        self.factors = factors
        self.num_sites = len(factors)
        if not self.num_sites > 1:
            raise ValueError("For 1 qubit states, do state vector")

        assign_devices(self.factors, min(DEVICE_COUNT, num_devices_to_use))

    def __repr__(self) -> str:
        result = "["
        for fac in self.factors:
            result += repr(fac)
            result += ", "
        result += "]"
        return result

    def __mul__(self, state: State) -> MPS:
        assert isinstance(state, MPS), "MPO can only be multiplied with MPS"
        assert (
            self.num_sites == state.num_sites
        ), "MPO and MPS don't have the same number of sites"

        # Move own factors to the same devices as the MPS's factors
        self.factors = [
            self.factors[i].to(state.factors[i].device) for i in range(self.num_sites)
        ]

        out_factors = []
        # to keep track of the bond dimensions of the output mps
        # so we can do the logic in a single sweep from either left-to-right or vice-versa
        left_bond_size = 1
        for i in range(self.num_sites):
            factor, left_bond_size = self._contract_factors(
                self.factors[i], state.factors[i], left_bond_size
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

    def __rmul__(self, state: MPS) -> MPS:
        return self * state
