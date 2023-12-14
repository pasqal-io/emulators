from .operations import contract, qr, svd
import cupy as cp
import numpy as np
from typing import Union, List


class MPS:
    """
    Matrix Product State
    """

    def __init__(self, sites: Union[int, List]):
        if isinstance(sites, int):
            self.num_sites = sites
            self.factors = []
            for i in range(self.num_sites):
                tensor = cp.zeros((1, 2, 1), dtype=np.complex128, order="F")
                tensor[0, 0, 0] = 1.0
                self.factors.append(tensor)
        elif isinstance(sites, List):
            self.factors = sites
            self.num_sites = len(sites)
            self.truncate()
        else:
            raise NotImplementedError()
        assert self.num_sites > 1  # otherwise, do state vector
        self.orth_center = 0

    def __repr__(self) -> str:
        result = "["
        for fac in self.factors:
            result += repr(fac)
            result += ", "
        result += "]"
        return result

    @staticmethod
    def _determine_cutoff_index(singular_values: cp.array) -> int:
        index = 0
        for i in singular_values:
            if i < 1e-8:
                continue
            index += 1
        return index

    def truncate(self) -> None:
        for i in range(self.num_sites - 1):
            q, r = qr("ijk->ijl,lk", self.factors[i])
            self.factors[i] = q
            self.factors[i + 1] = contract("ij, jkl->ikl", r, self.factors[i + 1])
        for i in range(self.num_sites - 1, 0, -1):
            u, s, vh = svd("ijk->il,ljk", self.factors[i])
            index = self._determine_cutoff_index(s)
            self.factors[i] = vh[:index, :, :]
            factor = contract(
                "ij, j->ij", u[:, :index], s[:index].astype(dtype=np.complex128)
            )
            factor = contract("ijk, kl->ijl", self.factors[i - 1], factor)
            self.factors[i - 1] = factor
