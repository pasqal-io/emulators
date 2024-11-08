from __future__ import annotations

import math
from collections import Counter
from typing import Any, List, Optional, Iterable

import torch

from emu_base import State
from emu_mps.algebra import add_factors, scale_factors
from emu_mps.utils import (
    DEVICE_COUNT,
    apply_measurement_errors,
    assign_devices,
    truncate_impl,
    tensor_trace,
    n_operator,
)


class MPS(State):
    """
    Matrix Product State, aka tensor train.

    Each tensor has 3 dimensions ordered as such: (left bond, site, right bond).

    Only qubits are supported.
    """

    DEFAULT_MAX_BOND_DIM: int = 1024
    DEFAULT_PRECISION: float = 1e-5

    def __init__(
        self,
        factors: List[torch.Tensor],
        /,
        *,
        orthogonality_center: Optional[int] = None,
        precision: float = DEFAULT_PRECISION,
        max_bond_dim: int = DEFAULT_MAX_BOND_DIM,
        num_gpus_to_use: Optional[int] = DEVICE_COUNT,
    ):
        """
        This constructor creates a MPS directly from a list of tensors. It is for internal use only.

        Args:
            factors: the tensors for each site
                WARNING: for efficiency in a lot of use cases, this list of tensors
                IS NOT DEEP-COPIED. Therefore, the new MPS object is not necessarily
                the exclusive owner of the list and its tensors. As a consequence,
                beware of potential external modifications affecting the list or the tensors.
                You are responsible for deciding whether to pass its own exclusive copy
                of the data to this constructor, or some shared objects.
            orthogonality_center: the orthogonality center of the MPS, or None (in which case
                it will be orthogonalized when needed)
            precision: the precision with which to truncate here or in tdvp
            max_bond_dim: the maximum bond dimension to allow
            num_gpus_to_use: distribute the factors over this many GPUs
                0=all factors to cpu, None=keep the existing device assignment.
        """
        self.precision = precision
        self.max_bond_dim = max_bond_dim

        assert all(
            factors[i - 1].shape[2] == factors[i].shape[0] for i in range(1, len(factors))
        ), "The dimensions of consecutive tensors should match"
        assert (
            factors[0].shape[0] == 1 and factors[-1].shape[2] == 1
        ), "The dimension of the left (right) link of the first (last) tensor should be 1"

        self.factors = factors
        self.num_sites = len(factors)
        assert self.num_sites > 1  # otherwise, do state vector

        assert (orthogonality_center is None) or (
            0 <= orthogonality_center < self.num_sites
        ), "Invalid orthogonality center provided"
        self.orthogonality_center = orthogonality_center

        if num_gpus_to_use is not None:
            assign_devices(self.factors, min(DEVICE_COUNT, num_gpus_to_use))

    @classmethod
    def make(
        cls,
        num_sites: int,
        precision: float = DEFAULT_PRECISION,
        max_bond_dim: int = DEFAULT_MAX_BOND_DIM,
        num_gpus_to_use: int = DEVICE_COUNT,
    ) -> MPS:
        """
        Returns a MPS in ground state |000..0>.

        Args:
            num_sites: the number of qubits
            precision: the precision with which to truncate here or in tdvp
            max_bond_dim: the maximum bond dimension to allow
            num_gpus_to_use: distribute the factors over this many GPUs
                0=all factors to cpu
        """
        if num_sites <= 1:
            raise ValueError("For 1 qubit states, do state vector")

        return cls(
            [
                torch.tensor([[[1.0], [0.0]]], dtype=torch.complex128)
                for _ in range(num_sites)
            ],
            precision=precision,
            max_bond_dim=max_bond_dim,
            num_gpus_to_use=num_gpus_to_use,
            orthogonality_center=0,  # Arbitrary: every qubit is an orthogonality center.
        )

    def __repr__(self) -> str:
        result = "["
        for fac in self.factors:
            result += repr(fac)
            result += ", "
        result += "]"
        return result

    def orthogonalize(self, desired_orthogonality_center: int = 0) -> int:
        """
        Orthogonalize the state on the given orthogonality center.

        Returns the new orthogonality center index as an integer,
        this is convenient for type-checking purposes.
        """
        assert (
            0 <= desired_orthogonality_center < self.num_sites
        ), f"Cannot move orthogonality center to nonexistent qubit #{desired_orthogonality_center}"

        lr_swipe_start = (
            self.orthogonality_center if self.orthogonality_center is not None else 0
        )

        for i in range(lr_swipe_start, desired_orthogonality_center):
            q, r = torch.linalg.qr(self.factors[i].reshape(-1, self.factors[i].shape[2]))
            self.factors[i] = q.reshape(self.factors[i].shape[0], 2, -1)
            self.factors[i + 1] = torch.tensordot(
                r.to(self.factors[i + 1].device), self.factors[i + 1], dims=1
            )

        rl_swipe_start = (
            self.orthogonality_center
            if self.orthogonality_center is not None
            else (self.num_sites - 1)
        )

        for i in range(rl_swipe_start, desired_orthogonality_center, -1):
            q, r = torch.linalg.qr(
                self.factors[i].reshape(self.factors[i].shape[0], -1).mT,
            )
            self.factors[i] = q.mT.reshape(-1, 2, self.factors[i].shape[2])
            self.factors[i - 1] = torch.tensordot(
                self.factors[i - 1], r.to(self.factors[i - 1].device), ([2], [1])
            )

        self.orthogonality_center = desired_orthogonality_center

        return desired_orthogonality_center

    def truncate(self) -> None:
        """
        SVD based truncation of the state. Puts the orthogonality center at the first qubit.
        Calls orthogonalize on the last qubit, and then sweeps a series of SVDs right-left.
        Uses self.precision and self.max_bond_dim for determining accuracy.
        An in-place operation.
        """
        self.orthogonalize(self.num_sites - 1)
        truncate_impl(
            self.factors,
            max_error=self.precision,
            max_rank=self.max_bond_dim,
        )
        self.orthogonality_center = 0

    def get_max_bond_dim(self) -> int:
        """
        Return the max bond dimension of this MPS.

        Returns:
            the largest bond dimension in the state
        """
        return max((x.shape[2] for x in self.factors), default=0)

    def sample(
        self, num_shots: int, p_false_pos: float = 0.0, p_false_neg: float = 0.0
    ) -> Counter[str]:
        """
        Samples bitstrings, taking into account the specified error rates.

        Args:
            num_shots: how many bitstrings to sample
            p_false_pos: the rate at which a 0 is read as a 1
            p_false_neg: teh rate at which a 1 is read as a 0

        Returns:
            the measured bitstrings, by count
        """
        self.orthogonalize(0)

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

    def norm(self) -> float:
        """Computes the norm of the MPS."""
        orthogonality_center = (
            self.orthogonality_center
            if self.orthogonality_center is not None
            else self.orthogonalize(0)
        )

        return float(
            torch.linalg.norm(self.factors[orthogonality_center].to("cpu")).item()
        )

    def _sample_implementation(self, rnd_vector: torch.Tensor) -> str:
        """
        Samples this MPS once, returning the resulting bitstring.
        """
        assert rnd_vector.shape == (self.num_sites,)
        assert self.orthogonality_center == 0

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

    def inner(self, other: State) -> float | complex:
        """
        Compute the inner product between this state and other.
        Note that self is the left state in the inner product,
        so this function is linear in other, and anti-linear in self

        Args:
            other: the other state

        Returns:
            inner product
        """
        assert isinstance(other, MPS), "Other state also needs to be an MPS"
        assert (
            self.num_sites == other.num_sites
        ), "States do not have the same number of sites"

        acc = torch.ones(1, 1, dtype=self.factors[0].dtype, device=self.factors[0].device)

        for i in range(self.num_sites):
            acc = acc.to(self.factors[i].device)
            acc = torch.tensordot(acc, other.factors[i].to(acc.device), dims=1)
            acc = torch.tensordot(self.factors[i].conj(), acc, dims=([0, 1], [0, 1]))

        return acc.item()  # type: ignore[no-any-return]

    def get_memory_footprint(self) -> float:
        """
        Returns the number of MBs of memory occupied to store the state

        Returns:
            the memory in MBs
        """
        return (  # type: ignore[no-any-return]
            sum(factor.element_size() * factor.numel() for factor in self.factors) * 1e-6
        )

    def __add__(self, other: State) -> MPS:
        """
        Returns the sum of two MPSs, computed with a direct algorithm.
        The resulting MPS is orthogonalized on the first site and truncated
        up to `self.precision`.

        Args:
            other: the other state

        Returns:
            the summed state
        """
        assert isinstance(other, MPS), "Other state also needs to be an MPS"
        new_tt = add_factors(self.factors, other.factors)
        result = MPS(
            new_tt,
            precision=self.precision,
            max_bond_dim=self.max_bond_dim,
            num_gpus_to_use=None,
            orthogonality_center=None,  # Orthogonality is lost.
        )
        result.truncate()
        return result

    def __rmul__(self, scalar: complex) -> MPS:
        """
        Multiply an MPS by a scalar.

        Args:
            scalar: the scale factor

        Returns:
            the scaled MPS
        """
        which = (
            self.orthogonality_center
            if self.orthogonality_center is not None
            else 0  # No need to orthogonalize for scaling.
        )
        factors = scale_factors(self.factors, scalar, which=which)
        return MPS(
            factors,
            precision=self.precision,
            max_bond_dim=self.max_bond_dim,
            num_gpus_to_use=None,
            orthogonality_center=self.orthogonality_center,
        )

    def __imul__(self, scalar: complex) -> MPS:
        return self.__rmul__(scalar)

    @staticmethod
    def from_state_string(
        *,
        basis: Iterable[str],
        nqubits: int,
        strings: dict[str, complex],
        **kwargs: Any,
    ) -> MPS:
        """
        See the base class.

        Args:
            basis: A tuple containing the basis states (e.g., ('r', 'g')).
            nqubits: the number of qubits.
            strings: A dictionary mapping state strings to complex or floats amplitudes.

        Returns:
            The resulting MPS representation of the state.s
        """

        basis = set(basis)
        if basis == {"r", "g"}:
            one = "r"
        elif basis == {"0", "1"}:
            one = "1"
        else:
            raise ValueError("Unsupported basis provided")

        basis_0 = torch.tensor([[[1.0], [0.0]]], dtype=torch.complex128)  # ground state
        basis_1 = torch.tensor([[[0.0], [1.0]]], dtype=torch.complex128)  # excited state

        accum_mps = MPS(
            [torch.zeros((1, 2, 1), dtype=torch.complex128)] * nqubits,
            orthogonality_center=0,
            **kwargs,
        )

        for state, amplitude in strings.items():
            factors = [basis_1 if ch == one else basis_0 for ch in state]
            accum_mps += amplitude * MPS(factors, **kwargs)
        norm = accum_mps.norm()
        if not math.isclose(1.0, norm, rel_tol=1e-5, abs_tol=0.0):
            print("\nThe state is not normalized, normalizing it for you.")
            accum_mps *= 1 / norm

        return accum_mps

    def expect_batch(self, single_qubit_operators: torch.Tensor) -> torch.Tensor:
        """
        Computes expectation values for each qubit and each single qubit operator in
        the batched input tensor.

        Returns a tensor T such that T[q, i] is the expectation value for qubit #q
        and operator single_qubit_operators[i].
        """
        orthogonality_center = (
            self.orthogonality_center
            if self.orthogonality_center is not None
            else self.orthogonalize(0)
        )

        result = torch.zeros(
            self.num_sites, single_qubit_operators.shape[0], dtype=torch.complex128
        )

        center_factor = self.factors[orthogonality_center]
        for qubit_index in range(orthogonality_center, self.num_sites):
            temp = torch.tensordot(center_factor.conj(), center_factor, ([0, 2], [0, 2]))

            result[qubit_index] = torch.tensordot(
                single_qubit_operators.to(temp.device), temp, dims=2
            )

            if qubit_index < self.num_sites - 1:
                _, r = torch.linalg.qr(center_factor.reshape(-1, center_factor.shape[2]))
                center_factor = torch.tensordot(
                    r, self.factors[qubit_index + 1].to(r.device), dims=1
                )

        center_factor = self.factors[orthogonality_center]
        for qubit_index in range(orthogonality_center - 1, -1, -1):
            _, r = torch.linalg.qr(
                center_factor.reshape(center_factor.shape[0], -1).mT,
            )
            center_factor = torch.tensordot(
                self.factors[qubit_index],
                r.to(self.factors[qubit_index].device),
                ([2], [1]),
            )

            temp = torch.tensordot(center_factor.conj(), center_factor, ([0, 2], [0, 2]))

            result[qubit_index] = torch.tensordot(
                single_qubit_operators.to(temp.device), temp, dims=2
            )

        return result

    def apply(self, qubit_index: int, single_qubit_operator: torch.Tensor) -> None:
        """
        Apply given single qubit operator to qubit qubit_index, leaving the MPS
        orthogonalized on that qubit.
        """
        self.orthogonalize(qubit_index)

        self.factors[qubit_index] = torch.tensordot(
            self.factors[qubit_index],
            single_qubit_operator.to(self.factors[qubit_index].device),
            ([1], [1]),
        ).transpose(1, 2)

    def get_correlation_matrix(
        self, *, operator: torch.Tensor = n_operator
    ) -> list[list[float]]:
        """
        Efficiently compute the symmetric correlation matrix
            C_ij = <self|operator_i operator_j|self>
        in basis ("r", "g").

        Args:
            operator: a 2x2 Torch tensor to use

        Returns:
            the corresponding correlation matrix
        """
        assert operator.shape == (2, 2)

        result = [[0.0 for _ in range(self.num_sites)] for _ in range(self.num_sites)]

        for left in range(0, self.num_sites):
            self.orthogonalize(left)
            accumulator = torch.tensordot(
                self.factors[left],
                operator.to(self.factors[left].device),
                dims=([1], [0]),
            )
            accumulator = torch.tensordot(
                accumulator, self.factors[left].conj(), dims=([0, 2], [0, 1])
            )
            result[left][left] = accumulator.trace().item().real
            for right in range(left + 1, self.num_sites):
                partial = torch.tensordot(
                    accumulator.to(self.factors[right].device),
                    self.factors[right],
                    dims=([0], [0]),
                )
                partial = torch.tensordot(
                    partial, self.factors[right].conj(), dims=([0], [0])
                )

                result[left][right] = (
                    torch.tensordot(
                        partial, operator.to(partial.device), dims=([0, 2], [0, 1])
                    )
                    .trace()
                    .item()
                    .real
                )
                result[right][left] = result[left][right]
                accumulator = tensor_trace(partial, 0, 2)

        return result


def inner(left: MPS, right: MPS) -> float | complex:
    """
    Wrapper around MPS.inner.

    Args:
        left: the anti-linear argument
        right: the linear argument

    Returns:
        the inner product
    """
    return left.inner(right)
