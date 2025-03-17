from __future__ import annotations
from collections import Counter
from typing import Any, Iterable
import torch
from emu_base import State, DEVICE_COUNT
from emu_sv.state_vector import StateVector

dtype = torch.complex128

from emu_sv.utils import _index_to_bitstring
from emu_sv.state_vector import StateVector


class DensityMatrix(State):
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

    def __add__(self, other):

        NotImplementedError("Not implemented")

    def __rmul__(self, scalar):
        NotImplementedError("Not implemented")

    def _normalize(self) -> None:
        # NOTE: use this in the callbacks
        """Checks if the input matrix has trace equals to 1.0 or not"""
        matrix_trace = torch.trace(self.matrix)
        if not torch.allclose(matrix_trace, torch.tensor(1.0, dtype=torch.float64)):
            self.matrix = self.matrix / matrix_trace

    def inner(self, other: State) -> float | complex:
        """
        Compute Tr(self^† @ other). The type of other must be DensityMatrix.

        Args:
            other: the other state

        Returns:
            the inner product
        """

        assert isinstance(
            other, DensityMatrix
        ), "Other state also needs to be a StateVector"
        assert (
            self.matrix.shape == other.matrix.shape
        ), "States do not have the same number of sites"

        return torch.trace(self.matrix.conj().T @ other.matrix).item()

    @classmethod
    def from_state_vector(cls, state: StateVector) -> DensityMatrix:
        """convert a state vector to a density matrix"""

        return DensityMatrix(
            torch.outer(state.vector, state.vector.conj()), gpu=state.vector.is_cuda
        )

    @staticmethod
    def from_state_string(
        *,
        basis: Iterable[str],
        nqubits: int,
        strings: dict[str, complex],
        **kwargs: Any,
    ) -> DensityMatrix:
        
        state_vector = StateVector.from_state_string(
            basis=basis, nqubits=nqubits, strings=strings, **kwargs
        )

        return DensityMatrix.from_state_vector(state_vector)

    def sample(
        self, num_shots: int = 1000, p_false_pos: float = 0.0, p_false_neg: float = 0.0
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

        probabilities = torch.abs(self.matrix.diagonal())
        probabilities /= probabilities.sum()  # multinomial does not normalize the input

        outcomes = torch.multinomial(probabilities, num_shots, replacement=True)

        # Convert outcomes to bitstrings and count occurrences
        counts = Counter(
            [_index_to_bitstring(self.matrix.diagonal(), outcome) for outcome in outcomes]
        )

        # NOTE: false positives and negatives
        return counts
