from __future__ import annotations
import math
from collections import Counter
from typing import List, Optional, Sequence, TypeVar, Mapping

import torch

from pulser.backend.state import State, Eigenstate
from emu_base import DEVICE_COUNT, apply_measurement_errors
from emu_mps.algebra import add_factors, scale_factors
from emu_mps.utils import (
    assign_devices,
    truncate_impl,
    tensor_trace,
)


ArgScalarType = TypeVar("ArgScalarType")
dtype = torch.complex128

DEFAULT_PRECISION = 1e-5
DEFAULT_MAX_BOND_DIM = 1024


class MPS(State[complex, torch.Tensor]):
    """
    Matrix Product State, aka tensor train.

    Each tensor has 3 dimensions ordered as such: (left bond, site, right bond).

    Only qubits are supported.
    """

    def __init__(
        self,
        factors: List[torch.Tensor],
        /,
        *,
        orthogonality_center: Optional[int] = None,
        precision: float = DEFAULT_PRECISION,
        max_bond_dim: int = DEFAULT_MAX_BOND_DIM,
        num_gpus_to_use: Optional[int] = DEVICE_COUNT,
        eigenstates: Sequence[Eigenstate] = ("r", "g"),
    ):
        """
        This constructor creates a MPS directly from a list of tensors. It is
        for internal use only.

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
            precision: the precision with which to keep this MPS
            max_bond_dim: the maximum bond dimension to allow for this MPS
            num_gpus_to_use: distribute the factors over this many GPUs
                0=all factors to cpu, None=keep the existing device assignment.
        """
        super().__init__(eigenstates=eigenstates)
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

        self.dim = len(self.eigenstates)
        assert all(factors[i].shape[1] == self.dim for i in range(self.num_sites)), (
            "All tensors should have the same physical dimension as the number "
            "of eigenstates"
        )

        self.n_operator = torch.zeros(
            self.dim, self.dim, dtype=dtype, device=self.factors[0].device
        )
        self.n_operator[1, 1] = 1.0

        assert (orthogonality_center is None) or (
            0 <= orthogonality_center < self.num_sites
        ), "Invalid orthogonality center provided"
        self.orthogonality_center = orthogonality_center

        if num_gpus_to_use is not None:
            assign_devices(self.factors, min(DEVICE_COUNT, num_gpus_to_use))

    @property
    def n_qudits(self) -> int:
        """The number of qudits in the state."""
        return self.num_sites

    @classmethod
    def make(
        cls,
        num_sites: int,
        precision: float = DEFAULT_PRECISION,
        max_bond_dim: int = DEFAULT_MAX_BOND_DIM,
        num_gpus_to_use: int = DEVICE_COUNT,
        eigenstates: Sequence[Eigenstate] = ["0", "1"],
    ) -> MPS:
        """
        Returns a MPS in ground state |000..0>.

        Args:
            num_sites: the number of qubits
            precision: the precision with which to keep this MPS
            max_bond_dim: the maximum bond dimension to allow for this MPS
            num_gpus_to_use: distribute the factors over this many GPUs
                0=all factors to cpu
        """
        if num_sites <= 1:
            raise ValueError("For 1 qubit states, do state vector")

        if len(eigenstates) == 2:
            ground_state = [
                torch.tensor([[[1.0], [0.0]]], dtype=dtype) for _ in range(num_sites)
            ]

        elif len(eigenstates) == 3:  # (g,r,x)
            ground_state = [
                torch.tensor([[[1.0], [0.0], [0.0]]], dtype=dtype)
                for _ in range(num_sites)
            ]

        else:
            raise ValueError(
                "Unsupported basis provided. The supported "
                "bases are:{('0','1'),('r','g'),('r','g','x')}"
            )

        return cls(
            ground_state,
            precision=precision,
            max_bond_dim=max_bond_dim,
            num_gpus_to_use=num_gpus_to_use,
            orthogonality_center=0,  # Arbitrary: every qubit is an orthogonality center.
            eigenstates=eigenstates,
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
            q, r = torch.linalg.qr(self.factors[i].view(-1, self.factors[i].shape[2]))

            self.factors[i] = q.view(self.factors[i].shape[0], self.dim, -1)
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
                self.factors[i].contiguous().view(self.factors[i].shape[0], -1).mT,
            )

            self.factors[i] = q.mT.view(-1, self.dim, self.factors[i].shape[2])
            self.factors[i - 1] = torch.tensordot(
                self.factors[i - 1], r.to(self.factors[i - 1].device), ([2], [1])
            )

        self.orthogonality_center = desired_orthogonality_center

        return desired_orthogonality_center

    def truncate(self) -> None:
        """
        SVD based truncation of the state. Puts the orthogonality center at the first qubit.
        Calls orthogonalize on the last qubit, and then sweeps a series of SVDs right-left.
        Uses self.config for determining accuracy.
        An in-place operation.
        """
        self.orthogonalize(self.num_sites - 1)
        truncate_impl(
            self.factors, precision=self.precision, max_bond_dim=self.max_bond_dim
        )
        self.orthogonality_center = 0

    def get_max_bond_dim(self) -> int:
        """
        Return the max bond dimension of this MPS.

        Returns:
            the largest bond dimension in the state
        """
        return max((factor.shape[2] for factor in self.factors), default=0)

    def sample(
        self,
        *,
        num_shots: int,
        one_state: Eigenstate | None = None,
        p_false_pos: float = 0.0,
        p_false_neg: float = 0.0,
    ) -> Counter[str]:
        """
        Samples bitstrings, taking into account the specified error rates.

        Args:
            num_shots: how many bitstrings to sample
            p_false_pos: the rate at which a 0 is read as a 1
            p_false_neg: the rate at which a 1 is read as a 0

        Returns:
            the measured bitstrings, by count
        """
        assert one_state in {None, "r", "1"}
        self.orthogonalize(0)

        bitstrings: Counter[str] = Counter()

        # Shots are performed in batches.
        # Larger max_batch_size is faster but uses more memory.
        max_batch_size = 32

        shots_done = 0

        while shots_done < num_shots:
            batch_size = min(max_batch_size, num_shots - shots_done)
            batched_accumulator = torch.ones(
                batch_size, 1, dtype=dtype, device=self.factors[0].device
            )

            batch_outcomes = torch.empty(batch_size, self.num_sites, dtype=torch.int)
            rangebatch = torch.arange(batch_size)
            for qubit, factor in enumerate(self.factors):
                batched_accumulator = torch.tensordot(
                    batched_accumulator.to(factor.device), factor, dims=1
                )

                # Probabilities for each state in the basis
                probn = torch.linalg.vector_norm(batched_accumulator, dim=2) ** 2

                # list of: 0,1 for |g>,|r> or 0,1,2 for |g>,|r>,|x>
                outcomes = torch.multinomial(probn, num_samples=1).reshape(-1)

                batch_outcomes[:, qubit] = outcomes

                # expected shape (batch_size, bond_dim)
                batched_accumulator = batched_accumulator[rangebatch, outcomes, :]

            shots_done += batch_size

            for outcome in batch_outcomes:
                bitstrings.update(["".join("1" if x == 1 else "0" for x in outcome)])

        if p_false_neg > 0 or p_false_pos > 0 and self.dim == 2:
            bitstrings = apply_measurement_errors(
                bitstrings,
                p_false_pos=p_false_pos,
                p_false_neg=p_false_neg,
            )
        if p_false_pos > 0 and self.dim > 2:
            raise NotImplementedError("Not implemented for qudits > 2 levels")

        return bitstrings

    def norm(self) -> torch.Tensor:
        """Computes the norm of the MPS."""
        orthogonality_center = (
            self.orthogonality_center
            if self.orthogonality_center is not None
            else self.orthogonalize(0)
        )
        # the torch.norm function is not properly typed.
        return self.factors[orthogonality_center].norm().cpu()  # type: ignore[no-any-return]

    def inner(self, other: State) -> torch.Tensor:
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

        return acc.view(1)[0].cpu()

    def overlap(self, other: State, /) -> torch.Tensor:
        """
        Compute the overlap of this state and other. This is defined as
        $|\\langle self | other \\rangle |^2$
        """
        return torch.abs(self.inner(other)) ** 2  # type: ignore[no-any-return]

    def entanglement_entropy(self, mps_site: int) -> torch.Tensor:
        """
        Returns
        the Von Neumann entanglement entropy of the state `mps` at the bond
        between sites b and b+1

        S = -Σᵢsᵢ² log(sᵢ²)),

        where sᵢ are the singular values at the chosen bond.
        """
        self.orthogonalize(mps_site)

        # perform svd on reshaped matrix at site b
        matrix = self.factors[mps_site].flatten(end_dim=1)
        s = torch.linalg.svdvals(matrix)

        s_e = torch.Tensor(torch.special.entr(s**2))
        s_e = torch.sum(s_e)

        self.orthogonalize(0)
        return s_e.cpu()

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
        up to `self.config.precision`.

        Args:
            other: the other state

        Returns:
            the summed state
        """
        assert isinstance(other, MPS), "Other state also needs to be an MPS"
        assert (
            self.eigenstates == other.eigenstates
        ), f"`Other` state has basis {other.eigenstates} != {self.eigenstates}"
        new_tt = add_factors(self.factors, other.factors)
        result = MPS(
            new_tt,
            precision=self.precision,
            max_bond_dim=self.max_bond_dim,
            num_gpus_to_use=None,
            orthogonality_center=None,  # Orthogonality is lost.
            eigenstates=self.eigenstates,
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
            eigenstates=self.eigenstates,
        )

    def __imul__(self, scalar: complex) -> MPS:
        return self.__rmul__(scalar)

    @classmethod
    def _from_state_amplitudes(
        cls,
        *,
        eigenstates: Sequence[Eigenstate],
        n_qudits: int,
        amplitudes: Mapping[str, complex],
    ) -> tuple[MPS, Mapping[str, complex]]:
        """
        See the base class.

        Args:
            basis: A tuple containing the basis states (e.g., ('r', 'g')).
            nqubits: the number of qubits.
            strings: A dictionary mapping state strings to complex or floats amplitudes.

        Returns:
            The resulting MPS representation of the state.s
        """

        leak = ""
        one = "r"
        basis = set(eigenstates)

        if basis == {"r", "g"}:
            pass
        elif basis == {"0", "1"}:
            one = "1"
        elif basis == {"g", "r", "x"}:
            leak = "x"
        else:
            raise ValueError("Unsupported basis provided")
        dim = len(eigenstates)
        if dim == 2:
            basis_0 = torch.tensor([[[1.0], [0.0]]], dtype=dtype)  # ground state
            basis_1 = torch.tensor([[[0.0], [1.0]]], dtype=dtype)  # excited state

        elif dim == 3:
            basis_0 = torch.tensor([[[1.0], [0.0], [0.0]]], dtype=dtype)  # ground state
            basis_1 = torch.tensor([[[0.0], [1.0], [0.0]]], dtype=dtype)  # excited state
            basis_x = torch.tensor([[[0.0], [0.0], [1.0]]], dtype=dtype)  # leakage state

        accum_mps = MPS(
            [torch.zeros((1, dim, 1), dtype=dtype)] * n_qudits,
            orthogonality_center=0,
            eigenstates=eigenstates,
        )
        for state, amplitude in amplitudes.items():
            factors = []
            for ch in state:
                if ch == one:
                    factors.append(basis_1)
                elif ch == leak:
                    factors.append(basis_x)
                else:
                    factors.append(basis_0)
            accum_mps += amplitude * MPS(factors, eigenstates=eigenstates)
        norm = accum_mps.norm()
        if not math.isclose(1.0, norm, rel_tol=1e-5, abs_tol=0.0):
            print("\nThe state is not normalized, normalizing it for you.")
            accum_mps *= 1 / norm

        return accum_mps, amplitudes

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

        result = torch.zeros(self.num_sites, single_qubit_operators.shape[0], dtype=dtype)

        center_factor = self.factors[orthogonality_center]
        for qubit_index in range(orthogonality_center, self.num_sites):
            temp = torch.tensordot(center_factor.conj(), center_factor, ([0, 2], [0, 2]))

            result[qubit_index] = torch.tensordot(
                single_qubit_operators.to(temp.device), temp, dims=2
            )

            if qubit_index < self.num_sites - 1:
                _, r = torch.linalg.qr(center_factor.view(-1, center_factor.shape[2]))
                center_factor = torch.tensordot(
                    r, self.factors[qubit_index + 1].to(r.device), dims=1
                )

        center_factor = self.factors[orthogonality_center]
        for qubit_index in range(orthogonality_center - 1, -1, -1):
            _, r = torch.linalg.qr(
                center_factor.view(center_factor.shape[0], -1).mT,
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

        self.factors[qubit_index] = (
            single_qubit_operator.to(self.factors[qubit_index].device)
            @ self.factors[qubit_index]
        )

    def get_correlation_matrix(
        self, operator: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Efficiently compute the symmetric correlation matrix
            C_ij = <self|operator_i operator_j|self>
        in basis ("r", "g").

        Args:
            operator: a 2x2 (or 3x3) Torch tensor to use

        Returns:
            the corresponding correlation matrix
        """

        if operator is None:
            operator = self.n_operator

        assert operator.shape == (self.dim, self.dim), "Operator has wrong shape"

        result = torch.zeros(self.num_sites, self.num_sites, dtype=dtype)

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
            result[left, left] = accumulator.trace().item().real
            for right in range(left + 1, self.num_sites):
                partial = torch.tensordot(
                    accumulator.to(self.factors[right].device),
                    self.factors[right],
                    dims=([0], [0]),
                )
                partial = torch.tensordot(
                    partial, self.factors[right].conj(), dims=([0], [0])
                )

                result[left, right] = (
                    torch.tensordot(
                        partial, operator.to(partial.device), dims=([0, 2], [0, 1])
                    )
                    .trace()
                    .item()
                    .real
                )
                result[right, left] = result[left, right]
                accumulator = tensor_trace(partial, 0, 2)

        return result


def inner(left: MPS, right: MPS) -> torch.Tensor:
    """
    Wrapper around MPS.inner.

    Args:
        left: the anti-linear argument
        right: the linear argument

    Returns:
        the inner product
    """
    return left.inner(right)
