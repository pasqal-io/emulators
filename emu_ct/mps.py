from __future__ import annotations

import torch
from typing import Union, List
from .config import Config


class MPS:
    """
    Matrix Product State
    When specifying the MPS from a list of tensors, ensure that
    the MPS is in an orthogonal gauge with center on the first qubit
    or put truncate=True (which will do it for you),
    otherwise tdvp will break!
    """

    def __init__(self, sites: Union[int, List], truncate: bool = False):
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
            assert self.num_sites > 1  # otherwise, do state vector
        else:
            raise ValueError(
                "Sites must specify a number of qubits, or a list of tensors representing the MPS"
            )

        self.num_devices = Config().get_num_devices_to_use()
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

        if truncate:
            self.truncate()

    def __repr__(self) -> str:
        result = "["
        for fac in self.factors:
            result += repr(fac)
            result += ", "
        result += "]"
        return result

    @staticmethod
    def _determine_cutoff_index(d: torch.Tensor, cutoff: float) -> int:
        acc = 0
        for i in range(d.shape[0] - 1, -1, -1):
            acc += d[i] * d[i]
            if acc > cutoff:
                return i + 1
        return d.shape[0]  # type: ignore[no-any-return]

    def orthogonalize(self) -> None:
        """
        Orthogonalize the state with given orthogonality center at the last qubit.
        An in-place operation.
        """
        for j in range(self.num_devices):
            for i in range(self.gpu_boundaries[j], self.gpu_boundaries[j + 1] - 1):
                factor_shape = self.factors[i].shape
                q, r = torch.linalg.qr(self.factors[i].reshape(-1, factor_shape[2]))
                self.factors[i] = q.reshape(factor_shape[0], factor_shape[1], -1)
                self.factors[i + 1] = torch.tensordot(r, self.factors[i + 1], dims=1)
            if j < self.num_devices - 1:
                i = self.gpu_boundaries[j + 1] - 1
                factor_shape = self.factors[i].shape
                q, r = torch.linalg.qr(self.factors[i].reshape(-1, factor_shape[2]))
                self.factors[i] = q.reshape(factor_shape[0], factor_shape[1], -1)
                self.factors[i + 1] = torch.tensordot(
                    r.to(self.device + str(j + 1)), self.factors[i + 1], dims=1
                )

    def truncate(self) -> None:
        """
        SVD based truncation of the state.
        An in-place operation.

        Args:
            max_bond (int): the maximum bond dimension to allow
        """
        precision = Config().get_bond_precision()
        self.orthogonalize()
        for j in range(self.num_devices - 1, -1, -1):
            for i in range(self.gpu_boundaries[j + 1] - 1, self.gpu_boundaries[j], -1):
                factor_shape = self.factors[i].shape
                # i.e. a = u*diag(d)*vh, where vh is v conjugate transpose
                # svd.apply calls torch_ops.svd.forward
                u, d, vh = torch.linalg.svd(self.factors[i].reshape(factor_shape[0], -1))
                max_bond = min(
                    self._determine_cutoff_index(d, precision),
                    Config().get_max_bond_dim(),
                )
                u = u[:, :max_bond]
                d = d[:max_bond]
                vh = vh[:max_bond, :]
                vh = vh.reshape(-1, factor_shape[1], factor_shape[2])
                self.factors[i] = vh
                tmp = u * d
                self.factors[i - 1] = torch.tensordot(self.factors[i - 1], tmp, dims=1)
            if j > 0:
                i = self.gpu_boundaries[j]
                factor_shape = self.factors[i].shape
                # i.e. a = u*diag(d)*vh, where vh is v conjugate transpose
                # svd.apply calls torch_ops.svd.forward
                u, d, vh = torch.linalg.svd(self.factors[i].reshape(factor_shape[0], -1))
                max_bond = min(
                    self._determine_cutoff_index(d, precision),
                    Config().get_max_bond_dim(),
                )
                u = u[:, :max_bond]
                d = d[:max_bond]
                vh = vh[:max_bond, :]
                vh = vh.reshape(-1, factor_shape[1], factor_shape[2])
                self.factors[i] = vh
                tmp = u * d
                self.factors[i - 1] = torch.tensordot(
                    self.factors[i - 1], tmp.to(self.device + str(j - 1)), dims=1
                )


def inner(left: MPS, right: MPS) -> float | complex:
    assert (
        left.gpu_boundaries == right.gpu_boundaries
    ), "states do not have the same gpu distribution"
    acc = torch.ones(1, 1, dtype=left.factors[0].dtype, device=left.factors[0].device)
    for device in range(left.num_devices):
        for i in range(left.gpu_boundaries[device], left.gpu_boundaries[device + 1]):
            # just contract into right always.
            # flop count is 2*l1*r1'*d2+2*d1'*d2*d2'
            # todo branch on this to contract into left
            acc = torch.tensordot(acc, right.factors[i], dims=1)
            acc = torch.tensordot(left.factors[i].conj(), acc, dims=([0, 1], [0, 1]))
        if device < left.num_devices - 1:
            acc = acc.to(left.device + str(device + 1))
    return acc.item()  # type: ignore[no-any-return]
