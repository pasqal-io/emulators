from __future__ import annotations
from collections import Counter
import math
from typing import Mapping, TypeVar, Type, Sequence
import torch
from pulser.backend import State
from emu_base import DEVICE_COUNT, apply_measurement_errors
from emu_sv.state_vector import StateVector
from emu_sv.utils import index_to_bitstring
from pulser.backend.state import Eigenstate

DensityMatrixType = TypeVar("DensityMatrixType", bound="DensityMatrix")

dtype = torch.complex128


class DensityMatrix(State[complex, torch.Tensor]):
    """Represents a density matrix in a computational basis."""

    # for the moment no need to check positivity and trace 1
    def __init__(
        self,
        matrix: torch.Tensor,
        *,
        gpu: bool = True,
    ):
        # NOTE: this accepts also zero matrices.

        device = "cuda" if gpu and DEVICE_COUNT > 0 else "cpu"
        self.matrix = matrix.to(dtype=dtype, device=device)

    @property
    def n_qudits(self) -> int:
        """The number of qudits in the state."""
        nqudits = math.log2(self.matrix.shape[0])
        return int(nqudits)

    @classmethod
    def make(cls, n_atoms: int, gpu: bool = True) -> DensityMatrix:
        """Creates the density matrix of the ground state |000...0>"""
        result = torch.zeros(2**n_atoms, 2**n_atoms, dtype=dtype)
        result[0, 0] = 1.0
        return cls(result, gpu=gpu)

    def __add__(self, other: State) -> DensityMatrix:
        raise NotImplementedError("Not implemented")

    def __rmul__(self, scalar: complex) -> DensityMatrix:
        raise NotImplementedError("Not implemented")

    def _normalize(self) -> None:
        # NOTE: use this in the callbacks
        """Normalize the density matrix state"""
        matrix_trace = torch.trace(self.matrix)
        if not torch.allclose(matrix_trace, torch.tensor(1.0, dtype=torch.float64)):
            self.matrix = self.matrix / matrix_trace

    def overlap(self, other: State) -> torch.Tensor:
        """
        Compute Tr(self^† @ other). The type of other must be DensityMatrix.

        Args:
            other: the other state

        Returns:
            the inner product

        Example:
        >>> density_bell_state = (1/2* torch.tensor([[1, 0, 0, 1], [0, 0, 0, 0],
        ... [0, 0, 0, 0], [1, 0, 0, 1]],dtype=torch.complex128))
        >>> density_c = DensityMatrix(density_bell_state, gpu=False)
        >>> density_c.overlap(density_c)
        tensor(1.+0.j, dtype=torch.complex128)
        """

        assert isinstance(
            other, DensityMatrix
        ), "Other state also needs to be a DensityMatrix"
        assert (
            self.matrix.shape == other.matrix.shape
        ), "States do not have the same number of sites"

        return torch.vdot(
            self.matrix.flatten(), other.matrix.to(self.matrix.device).flatten()
        )

    @classmethod
    def from_state_vector(cls, state: StateVector) -> DensityMatrix:
        """Convert a state vector to a density matrix.
        This function takes a state vector |ψ❭ and returns the corresponding
        density matrix ρ = |ψ❭❬ψ| representing the pure state |ψ❭.
        Example:
           >>> from emu_sv import StateVector
           >>> import math
           >>> bell_state_vec = 1 / math.sqrt(2) * torch.tensor(
           ... [1.0, 0.0, 0.0, 1.0j],dtype=torch.complex128)
           >>> bell_state = StateVector(bell_state_vec, gpu=False)
           >>> density = DensityMatrix.from_state_vector(bell_state)
           >>> print(density.matrix)
           tensor([[0.5000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000-0.5000j],
                   [0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j],
                   [0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j],
                   [0.0000+0.5000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.5000+0.0000j]],
                  dtype=torch.complex128)
        """

        return cls(
            torch.outer(state.vector, state.vector.conj()), gpu=state.vector.is_cuda
        )

    @classmethod
    def _from_state_amplitudes(
        cls: Type[DensityMatrixType],
        *,
        eigenstates: Sequence[Eigenstate],
        n_qudits: int,
        amplitudes: Mapping[str, complex],
    ) -> tuple[DensityMatrix, Mapping[str, complex]]:
        """Transforms a state given by a string into a density matrix.

        Construct a state from the pulser abstract representation
        https://pulser.readthedocs.io/en/stable/conventions.html

        Args:
            basis: A tuple containing the basis states (e.g., ('r', 'g')).
            nqubits: the number of qubits.
            strings: A dictionary mapping state strings to complex or floats amplitudes.

        Returns:
            The resulting state.

        Examples:
            >>> eigenstates = ("r","g")
            >>> n = 2
            >>> dense_mat=DensityMatrix.from_state_amplitudes(eigenstates=eigenstates,
            ... amplitudes={"rr":1.0,"gg":1.0})
            >>> print(dense_mat.matrix)
            tensor([[0.5000+0.j, 0.0000+0.j, 0.0000+0.j, 0.5000+0.j],
                    [0.0000+0.j, 0.0000+0.j, 0.0000+0.j, 0.0000+0.j],
                    [0.0000+0.j, 0.0000+0.j, 0.0000+0.j, 0.0000+0.j],
                    [0.5000+0.j, 0.0000+0.j, 0.0000+0.j, 0.5000+0.j]],
                   dtype=torch.complex128)
        """

        state_vector, amplitudes = StateVector._from_state_amplitudes(
            eigenstates=eigenstates, n_qudits=n_qudits, amplitudes=amplitudes
        )

        return DensityMatrix.from_state_vector(state_vector), amplitudes

    def sample(
        self,
        num_shots: int = 1000,
        one_state: Eigenstate | None = None,
        p_false_pos: float = 0.0,
        p_false_neg: float = 0.0,
    ) -> Counter[str]:
        """
        Samples bitstrings, taking into account the specified error rates.

        Args:
            num_shots: how many bitstrings to sample
            p_false_pos: the rate at which a 0 is read as a 1
            p_false_neg: teh rate at which a 1 is read as a 0

        Returns:
            the measured bitstrings, by count

        Example:
        >>> import math
        >>> torch.manual_seed(1234)
        >>> from emu_sv import StateVector
        >>> bell_vec = 1 / math.sqrt(2) * torch.tensor(
        ... [1.0, 0.0, 0.0, 1.0j],dtype=torch.complex128)
        >>> bell_state_vec = StateVector(bell_vec)
        >>> bell_density = DensityMatrix.from_state_vector(bell_state_vec)
        >>> bell_density.sample(1000)
         Counter({'00': 517, '11': 483})
        """

        probabilities = torch.abs(self.matrix.diagonal())

        outcomes = torch.multinomial(probabilities, num_shots, replacement=True)

        # Convert outcomes to bitstrings and count occurrences
        counts = Counter(
            [index_to_bitstring(self.n_qudits, outcome) for outcome in outcomes]
        )

        if p_false_neg > 0 or p_false_pos > 0:
            counts = apply_measurement_errors(
                counts,
                p_false_pos=p_false_pos,
                p_false_neg=p_false_neg,
            )
        return counts


if __name__ == "__main__":
    import doctest

    doctest.testmod()
