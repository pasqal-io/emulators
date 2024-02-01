from typing import Any
from .mps import MPS
from .config import Config
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

        self.num_devices = Config().get_num_devices_to_use()
        self.gpu_boundaries = [0]
        if self.num_devices == 0:
            self.num_devices = (
                1  # from now on we will use this to loop over factors in batches
            )
            self.gpu_boundaries = [0, self.num_sites]
            self.device = "cpu:"
        else:
            bin_size = self.num_sites / self.num_devices
            self.gpu_boundaries = [
                round(bin_size * i) for i in range(self.num_devices + 1)
            ]
            self.device = "cuda:"

        for i in range(self.num_devices):
            for j in range(self.gpu_boundaries[i], self.gpu_boundaries[i + 1]):
                self.factors[j] = self.factors[j].to(self.device + str(i))

    def __repr__(self) -> str:
        result = "["
        for fac in self.factors:
            result += repr(fac)
            result += ", "
        result += "]"
        return result

    def __mul__(self, state: MPS) -> MPS:
        assert (
            self.gpu_boundaries == state.gpu_boundaries
        ), "mpo and mps do not have the same gpu distribution"
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
