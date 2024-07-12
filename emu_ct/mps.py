from __future__ import annotations

import torch
import math
from typing import Union, List
from emu_ct.utils import split_tensor, assign_devices, DEVICE_COUNT
from emu_ct.base_classes.state import State
from collections import Counter


class MPS(State):
    """
    Matrix Product State
    When specifying the MPS from a list of tensors, ensure that
    the MPS is in an orthogonal gauge with center on the first qubit
    or put truncate=True (which will do it for you),
    otherwise tdvp will break!
    """

    def __init__(
        self,
        sites: Union[int, List[torch.Tensor]],
        truncate: bool = False,
        precision: float = 1e-5,
        max_bond_dim: int = 1024,
        num_devices_to_use: int = DEVICE_COUNT,
        keep_devices: bool = False,
    ):
        self.precision = precision
        self.max_bond_dim = max_bond_dim
        self.factors: List[torch.Tensor] = []

        if isinstance(sites, int):
            self.num_sites = sites
            if not self.num_sites > 1:
                raise ValueError("For 1 qubit states, do state vector")

            for i in range(self.num_sites):
                tensor = torch.zeros((1, 2, 1), dtype=torch.complex128)
                tensor[0, 0, 0] = 1.0
                self.factors.append(tensor)
        elif isinstance(sites, List):
            assert all(
                sites[i - 1].shape[2] == sites[i].shape[0] for i in range(1, len(sites))
            )
            assert sites[0].shape[0] == sites[-1].shape[2] == 1

            self.factors = sites
            self.num_sites = len(sites)
            assert self.num_sites > 1  # otherwise, do state vector
        else:
            raise ValueError(
                "Sites must specify a number of qubits, or a list of tensors representing the MPS"
            )

        if not keep_devices:
            assign_devices(self.factors, min(DEVICE_COUNT, num_devices_to_use))

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

            l, r = split_tensor(
                self.factors[i].reshape(factor_shape[0], -1),
                max_error=self.precision,
                max_rank=self.max_bond_dim,
                orth_center_right=False,
            )

            r = r.reshape(-1, factor_shape[1], factor_shape[2])
            self.factors[i] = r
            self.factors[i - 1] = torch.tensordot(
                self.factors[i - 1], l.to(self.factors[i - 1].device), dims=1
            )

    def get_max_bond_dim(self) -> int:
        """Return the max bond dimension of MPS"""
        return max((x.shape[2] for x in self.factors), default=0)

    def sample(self, num_shots: int) -> Counter[str]:
        """Returns bitstring distribution for a given number of samples or shots"""
        num_qubits = len(self.factors)
        rnd_matrix = torch.rand(num_shots, num_qubits)
        sampled_bitstrings = [
            self._sample_implementation(rnd_matrix[x, :], truncate=False)
            for x in range(num_shots)
        ]
        return Counter(sampled_bitstrings)

    def _sample_implementation(
        self, rnd_vector: torch.Tensor, truncate: bool = False
    ) -> str:
        """Returns a sample in string output."""
        # the code is taken from ITensors and adapted for d=2 (qubits)
        num_qubits = len(self.factors)

        if truncate:  # moves the orthogonality center to the first qubit
            self.truncate()

        bitstring = ""
        acc_mps_j: torch.tensor = self.factors[0]

        for qubit in range(num_qubits):
            # comp_basis is a projector: 0 is for ket |0> and 1 for ket |1>
            comp_basis = 0  # check if the qubit is in |0>
            # Measure the qubit j by applying the projector onto nth comp basis state
            tensorj_projected_n = acc_mps_j[:, comp_basis, :]
            probability_n = (tensorj_projected_n.norm() ** 2).item()

            if rnd_vector[qubit] > probability_n:
                # the qubit is in |1>
                comp_basis = 1
                tensorj_projected_n = acc_mps_j[:, comp_basis, :]
                probability_n = 1 - probability_n

            bitstring += str(comp_basis)
            if qubit < num_qubits - 1:
                acc_mps_j = torch.tensordot(
                    tensorj_projected_n.to(device=self.factors[qubit + 1].device),
                    self.factors[qubit + 1],
                    dims=1,
                )
                acc_mps_j /= math.sqrt(probability_n)

        return bitstring

    def inner(self, right: State) -> float | complex:
        assert isinstance(right, MPS), "Other state also needs to be an MPS"
        assert (
            self.num_sites == right.num_sites
        ), "States do not have the same number of sites"

        acc = torch.ones(1, 1, dtype=self.factors[0].dtype, device=self.factors[0].device)

        for i in range(self.num_sites):
            acc = acc.to(self.factors[i].device)
            acc = torch.tensordot(acc, right.factors[i].to(acc.device), dims=1)
            acc = torch.tensordot(self.factors[i].conj(), acc, dims=([0, 1], [0, 1]))

        return acc.item()  # type: ignore[no-any-return]


def inner(left: MPS, right: MPS) -> float | complex:
    return left.inner(right)
