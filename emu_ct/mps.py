import torch
from typing import Union, List


class MPS:
    """
    Matrix Product State
    """

    def __init__(self, sites: Union[int, List]):
        if isinstance(sites, int):
            self.num_sites = sites
            if not self.num_sites > 1:
                raise ValueError("For 1 qubit states, do state vector")
            self.factors = []
            for i in range(self.num_sites):
                tensor = torch.zeros((1, 2, 1), dtype=torch.complex128)
                tensor[0, 0, 0] = 1.0
                self.factors.append(tensor)
        elif isinstance(sites, List):
            self.factors = sites
            self.num_sites = len(sites)
            if not self.num_sites > 1:
                raise ValueError("For 1 qubit states, do state vector")
            self.truncate()
        else:
            raise NotImplementedError()
        self.orth_center = 0

    def __repr__(self) -> str:
        result = "["
        for fac in self.factors:
            result += repr(fac)
            result += ", "
        result += "]"
        return result

    @staticmethod
    def _determine_cutoff_index(singular_values: torch.Tensor) -> int:
        index = 0
        for i in singular_values:
            if i < 1e-8:
                break
            index += 1
        return index

    def orthogonalize(self, center: int = 0) -> None:
        """
        Orthogonalize the state with given orthogonality center.
        An in-place operation.

        Args:
            center (int): where to put the orthogonality center
        """
        nfactors = len(self.factors)
        for i in range(center):
            factor_shape = self.factors[i].shape
            q, r = torch.linalg.qr(self.factors[i].reshape(-1, factor_shape[2]))
            self.factors[i] = q.reshape(factor_shape[0], factor_shape[1], -1)
            self.factors[i + 1] = torch.einsum("ac,cde->ade", r, self.factors[i + 1])
        for i in range(nfactors - 1, center, -1):
            factor_shape = self.factors[i].shape
            q, r = torch.linalg.qr(
                self.factors[i].reshape(factor_shape[1], -1).transpose(1, 2)
            )
            self.factors[i] = q.transpose(1, 2).reshape(
                -1, factor_shape[1], factor_shape[2]
            )
            self.factors[i - 1] = torch.einsum("ace,fe->acf", self.factors[i - 1], r)

    def truncate(self, max_bond: int = 1024) -> None:
        """
        SVD based truncation of the state.
        An in-place operation.

        Args:
            max_bond (int): the maximum bond dimension to allow
        """
        nfactors = len(self.factors)
        self.orthogonalize(nfactors - 1)
        for i in range(nfactors - 1, 0, -1):
            factor_shape = self.factors[i].shape
            # i.e. a = u*diag(d)*vh, where vh is v conjugate transpose
            # svd.apply calls torch_ops.svd.forward
            u, d, vh = torch.linalg.svd(self.factors[i].reshape(factor_shape[0], -1))
            max_bond = min(max_bond, self._determine_cutoff_index(d))
            u = u[:, :max_bond]
            d = d[:max_bond]
            vh = vh[:max_bond, :]
            vh = vh.reshape(-1, factor_shape[1], factor_shape[2])
            self.factors[i] = vh
            self.factors[i - 1] = torch.einsum("cde,ef,f->cdf", self.factors[i - 1], u, d)
