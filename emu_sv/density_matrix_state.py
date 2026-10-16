from __future__ import annotations
from typing import Mapping, TypeVar, Type, Sequence
import torch
from pulser.backend import State
from emu_sv.dense_state import DenseState
from emu_sv.state_vector import StateVector
from pulser.backend.state import Eigenstate

DensityMatrixType = TypeVar("DensityMatrixType", bound="DensityMatrix")

dtype = torch.complex128


class DensityMatrix(DenseState):
    """Represents an n-qubit density matrix ρ in the computational (|g⟩, |r⟩)
    basis. The input should be a square complex tensor with shape (2ⁿ, 2ⁿ),
    where n is the number of atoms. ρ must be Hermitian, positive semidefinite,
    and has trace 1. These checks are too expensive to actually perform, so
    it's up to the user to ensure these constraints are met.

    Args:
        data: Square complex tensor of shape (2ⁿ, 2ⁿ),
            Hermitian with trace 1, that represents the state in the
            computational basis.
        gpu: If True, place the operator on a CUDA device when
            available. Default: True.
        eigenstates: sequence of eigenstates used as basis only qubit basis are
            supported (default: ('r','g'))
    """

    # for the moment no need to check positivity and trace 1

    @classmethod
    def make(cls, n_atoms: int, gpu: bool = True) -> DensityMatrix:
        """Creates the density matrix of the ground state |000...0>"""
        result = torch.zeros(2**n_atoms, 2**n_atoms, dtype=dtype)
        result[0, 0] = 1.0
        return cls(result, gpu=gpu)

    def __add__(self, other: State) -> DensityMatrix:
        raise NotImplementedError("Not implemented")

    def __rmul__(self, scalar: complex) -> DensityMatrix:
        return DensityMatrix(self.data * scalar, gpu=self.data.is_cuda)

    def _normalize(self) -> None:
        # NOTE: use this in the callbacks
        """Normalize the density matrix state"""
        matrix_trace = torch.trace(self.data).real
        if not torch.allclose(matrix_trace, torch.tensor(1.0, dtype=torch.float64)):
            self.data = self.data / matrix_trace

    def norm(self) -> torch.Tensor:
        """Returns the norm of the state

        Returns:
            the norm of the state
        """
        nrm: torch.Tensor = torch.trace(self.data).real.cpu()
        return nrm

    @classmethod
    def from_state_vector(cls, state: StateVector) -> DensityMatrix:
        """
        Convert a state vector to a density matrix.
        This function takes a state vector |ψ❭ and returns the corresponding
        density matrix ρ = |ψ❭❬ψ| representing the pure state |ψ❭.

        Examples:
            ```python
            bell_state_vec = 0.7071 * torch.tensor([1.0, 0.0, 0.0, 1.0j],
            dtype=torch.complex128)
            bell_state = StateVector(bell_state_vec, gpu=False)
            density = DensityMatrix.from_state_vector(bell_state)
            print(density.data)
            ```

            Output:
            ```
            tensor([[0.5000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000-0.5000j],
                   [0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j],
                   [0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j],
                   [0.0000+0.5000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.5000+0.0000j]],
                  dtype=torch.complex128)
            ```
        """

        return cls(torch.outer(state.data, state.data.conj()), gpu=state.data.is_cuda)

    @classmethod
    def _from_state_amplitudes(
        cls: Type[DensityMatrixType],
        *,
        eigenstates: Sequence[Eigenstate],
        n_qudits: int,
        amplitudes: Mapping[str, complex],
    ) -> tuple[DensityMatrix, Mapping[str, complex]]:
        """
        Transforms a state given by a string into a density matrix.

        Construct a state from the pulser abstract representation
        https://pulser.readthedocs.io/en/stable/conventions.html

        Args:
            basis: A tuple containing the basis states (e.g., ('r', 'g')).
            nqubits: the number of qubits.
            strings: A dictionary mapping state strings to complex or floats
            amplitudes.

        Returns:
            The resulting state.

        Example:
            ```python
            eigenstates = ("r","g")
            n = 2
            dense_mat=DensityMatrix.from_state_amplitudes(eigenstates=eigenstates,
            amplitudes={"rr":1.0,"gg":1.0})
            print(dense_mat.data)
            ```

            Output:
            ```
            tensor([[0.5000+0.j, 0.0000+0.j, 0.0000+0.j, 0.5000+0.j],
                    [0.0000+0.j, 0.0000+0.j, 0.0000+0.j, 0.0000+0.j],
                    [0.0000+0.j, 0.0000+0.j, 0.0000+0.j, 0.0000+0.j],
                    [0.5000+0.j, 0.0000+0.j, 0.0000+0.j, 0.5000+0.j]],
                   dtype=torch.complex128)
            ```
        """

        state_vector, _ = StateVector._from_state_amplitudes(
            eigenstates=eigenstates, n_qudits=n_qudits, amplitudes=amplitudes
        )

        return DensityMatrix.from_state_vector(state_vector), amplitudes
