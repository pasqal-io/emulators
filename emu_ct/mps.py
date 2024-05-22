from __future__ import annotations

import torch
from typing import Union, List
from .utils import truncated_svd, assign_devices


class MPS:
    """
    Matrix Product State
    When specifying the MPS from a list of tensors, ensure that
    the MPS is in an orthogonal gauge with center on the first qubit
    or put truncate=True (which will do it for you),
    otherwise tdvp will break!
    """

    def __init__(
        self,
        sites: Union[int, List],
        truncate: bool = False,
        precision: float = 1e-5,
        max_bond_dim: int = 1024,
        num_devices_to_use: int = torch.cuda.device_count(),
    ):
        self.precision = precision
        self.max_bond_dim = max_bond_dim

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

        assign_devices(self.factors, min(torch.cuda.device_count(), num_devices_to_use))

        if truncate:
            self.truncate()

    def __repr__(self) -> str:
        result = "["
        for fac in self.factors:
            result += repr(fac)
            result += ", "
        result += "]"
        return result

    def orthogonalize(self) -> None:
        """
        Orthogonalize the state with given orthogonality center at the last qubit.
        An in-place operation.
        """
        for i in range(self.num_sites - 1):
            factor = self.factors[i]
            factor_shape = factor.shape
            q, r = torch.linalg.qr(factor.reshape(-1, factor.shape[2]))
            self.factors[i] = q.reshape(factor_shape[0], factor_shape[1], -1)
            self.factors[i + 1] = torch.tensordot(
                r.to(self.factors[i + 1].device), self.factors[i + 1], dims=1
            )

    def truncate(self) -> None:
        """
        SVD based truncation of the state.
        An in-place operation.
        """
        self.orthogonalize()

        for i in range(self.num_sites - 1, 0, -1):
            factor_shape = self.factors[i].shape

            u, d, vh = truncated_svd(
                self.factors[i].reshape(factor_shape[0], -1),
                max_error=self.precision,
                max_rank=self.max_bond_dim,
            )

            vh = vh.reshape(-1, factor_shape[1], factor_shape[2])
            self.factors[i] = vh
            tmp = u * d
            self.factors[i - 1] = torch.tensordot(
                self.factors[i - 1], tmp.to(self.factors[i - 1].device), dims=1
            )


def inner(left: MPS, right: MPS) -> float | complex:
    assert (
        left.num_sites == right.num_sites
    ), "States do not have the same number of sites"

    acc = torch.ones(1, 1, dtype=left.factors[0].dtype, device=left.factors[0].device)

    for i in range(left.num_sites):
        acc = acc.to(left.factors[i].device)
        acc = torch.tensordot(acc, right.factors[i].to(acc.device), dims=1)
        acc = torch.tensordot(left.factors[i].conj(), acc, dims=([0, 1], [0, 1]))

    return acc.item()  # type: ignore[no-any-return]
