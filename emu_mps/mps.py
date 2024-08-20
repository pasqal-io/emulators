from __future__ import annotations

import math
from collections import Counter
from typing import Any, List, Union

import torch

from emu_mps.algebra import add_factors, mul_factors
from emu_mps.base_classes.state import State
from emu_mps.utils import (
    DEVICE_COUNT,
    apply_measurement_errors,
    assign_devices,
    truncate_impl,
)


class MPS(State):
    """
    Matrix Product State, aka tensor train.

    Each tensor has 3 dimensions ordered as such: (left bond, site, right bond).

    Only qubits are supported.

    When specifying the MPS from a list of tensors, ensure that
    the MPS is in an orthogonal gauge with center on the first qubit
    or put truncate=True (which will do it for you),
    otherwise tdvp will break!
    """

    def __init__(
        self,
        sites: Union[int, List[torch.Tensor]],
        /,
        *,
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
            if sites[0].shape[0] != 1 or sites[-1].shape[2] != 1:
                raise ValueError(
                    "The dimension of the left (right) link of the first (last) tensor should be 1"
                )

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
            self._truncate()

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

    def _truncate(self) -> None:
        """
        Eigenvalues based truncation of the state.
        An in-place operation.
        """
        self.orthogonalize()
        truncate_impl(
            self.factors,
            max_error=self.precision,
            max_rank=self.max_bond_dim,
        )

    def get_max_bond_dim(self) -> int:
        """
        Return the max bond dimension of this MPS.
        """
        return max((x.shape[2] for x in self.factors), default=0)

    def sample(
        self, num_shots: int, p_false_pos: float = 0.0, p_false_neg: float = 0.0
    ) -> Counter[str]:
        """
        Returns bitstring distribution for a given number of samples or shots.
        """
        num_qubits = len(self.factors)
        rnd_matrix = torch.rand(num_shots, num_qubits)
        bitstrings = Counter(
            self._sample_implementation(rnd_matrix[x, :]) for x in range(num_shots)
        )
        if p_false_neg > 0 or p_false_pos > 0:
            bitstrings = apply_measurement_errors(
                bitstrings,
                p_false_pos=p_false_pos,
                p_false_neg=p_false_neg,
            )
        return bitstrings

    def _sample_implementation(self, rnd_vector: torch.Tensor) -> str:
        """
        Samples this MPS once, returning the resulting bitstring.
        """
        assert rnd_vector.shape == (self.num_sites,)

        num_qubits = len(self.factors)

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
        """
        Computes the inner product between this MPS and the argument.
        """

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

    def get_memory_footprint(self) -> float:
        return (  # type: ignore[no-any-return]
            sum(factor.element_size() * factor.numel() for factor in self.factors) * 1e-6
        )

    def __add__(self, other: State) -> State:
        """
        Returns the sum of two MPSs, computed with a direct algorithm.
        The resulting MPS is orthogonalized on the first site and truncated
        up to `self.precision`.
        """
        assert isinstance(other, MPS), "Other state also needs to be an MPS"
        new_tt = add_factors(self.factors, other.factors)
        return MPS(
            new_tt,
            truncate=True,
            precision=self.precision,
            max_bond_dim=self.max_bond_dim,
            keep_devices=True,
        )

    def __rmul__(self, scalar: complex) -> State:
        """
        Multiply an MPS by scalar.
        Assumes the MPS is orthogonalized on the site 0.
        """
        factors = mul_factors(self.factors, scalar)
        return MPS(
            factors,
            precision=self.precision,
            max_bond_dim=self.max_bond_dim,
            keep_devices=True,
        )

    @staticmethod
    def from_state_string(
        *,
        basis: tuple[str, ...],
        nqubits: int,
        strings: dict[str, complex],
        **kwargs: Any,
    ) -> State:
        """Transforms a state given by a string into an MPS.

        For example, 1/sqrt(2)*(|000>+|111>) -> emu_mps.MPS

        Args:
            basis: A tuple containing the basis states (e.g., ('r', 'g')).
            qubits: A list of qubit ids.
            strings: A dictionary mapping state strings to complex or floats amplitudes.

        Return:
            State: The resulting MPS representation of the state.
        """

        if set(basis) != {"r", "g"}:
            raise ValueError("Only the rydberg-ground basis is currently supported")

        values = list(strings.values())
        norm_state = math.sqrt(sum((ampli * ampli.conjugate()).real for ampli in values))

        if not math.isclose(1.0, norm_state, rel_tol=1e-5, abs_tol=0.0):
            print("\nThe state is not normalized, normalizing it for you.")
            strings = {key: value / norm_state for key, value in strings.items()}

        basis_g = torch.tensor([[[1.0], [0.0]]], dtype=torch.complex128)  # ground state
        basis_r = torch.tensor([[[0.0], [1.0]]], dtype=torch.complex128)  # excited state

        accum_mps: State = MPS(
            [torch.zeros((1, 2, 1), dtype=torch.complex128)] * nqubits, **kwargs
        )

        for state, amplitude in strings.items():
            factors = [basis_r if ch == "r" else basis_g for ch in state]
            accum_mps += amplitude * MPS(factors, **kwargs)

        return accum_mps


def inner(left: MPS, right: MPS) -> float | complex:
    return left.inner(right)
