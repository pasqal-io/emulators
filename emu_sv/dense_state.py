from __future__ import annotations
from pulser.backend import State
import torch
import math
from pulser.backend.state import Eigenstate
from typing import Sequence, TypeVar, Type, Mapping
from emu_base import DEVICE_COUNT
from emu_base import apply_measurement_errors
from emu_sv.utils import index_to_bitstring
from collections import Counter

dtype = torch.complex128
StateType = TypeVar("StateType", bound="State")
DenseStateType = TypeVar("DenseStateType", bound="DenseState")


class DenseState(State[complex, torch.Tensor]):
    def __init__(
        self,
        data: torch.Tensor,
        *,
        gpu: bool = True,
        eigenstates: Sequence[Eigenstate] = ("r", "g"),
    ):
        assert data.dim() == 1 or data.dim() == 2
        super().__init__(eigenstates=eigenstates)

        device = "cuda" if gpu and DEVICE_COUNT > 0 else "cpu"
        self.data = data.to(dtype=dtype, device=device)

    def __repr__(self) -> str:
        return repr(self.data)

    @property
    def n_qudits(self) -> int:
        """The number of qudits in the state."""
        nqudits = math.log2(self.data.shape[0])
        return int(nqudits)

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

        Examples:
            ```python
            torch.manual_seed(1234)
            from emu_sv import StateVector
            bell_vec = 0.7071 * torch.tensor([1.0, 0.0, 0.0, 1.0j],
               dtype=torch.complex128)
            bell_state_vec = StateVector(bell_vec)
            bell_density = DensityMatrix.from_state_vector(bell_state_vec)
            bell_density.sample(1000)
            ```

            Output:
            ```
            Counter({'00': 517, '11': 483})
            ```
        """

        probabilities: torch.Tensor
        if self.data.dim() == 2:
            probabilities = torch.abs(self.data.diagonal())
        elif self.data.dim() == 1:
            probabilities = torch.abs(self.data) ** 2

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

    def overlap(self, other: State) -> torch.Tensor:
        """
        Compute Tr(self^† @ other). The type of other must be DensityMatrix.

        Args:
            other: the other state

        Returns:
            the inner product

        Examples:
            ```python
            density_bell_state = 0.5 * torch.tensor([[1, 0, 0, 1], [0, 0, 0, 0],
            [0, 0, 0, 0], [1, 0, 0, 1]],dtype=torch.complex128)
            density_c = DensityMatrix(density_bell_state, gpu=False)
            density_c.overlap(density_c)
            ```

            Output:
            ```
            tensor(1.+0.j, dtype=torch.complex128)
            ```
        """
        assert isinstance(
            other, DenseState
        ), "Only overlaps between emu-sv states are implemented."
        l_data = self.data
        r_data = other.data
        if l_data.dim() < r_data.dim():
            l_data = torch.outer(l_data, l_data)
        if r_data.dim() < l_data.dim():
            r_data = torch.outer(r_data, r_data)

        assert (
            self.data.shape == other.data.shape
        ), "States do not have the same number of sites"

        overlap = torch.vdot(
            self.data.flatten(), other.data.to(self.data.device).flatten()
        )
        if l_data.dim() == 1:
            overlap = torch.abs(overlap) ** 2

        return overlap

    @staticmethod
    def _build_state_vector(
        n_qudits: int,
        amplitudes: Mapping[str, complex],
    ) -> torch.Tensor:
        """Transforms a state given by a string into a state vector.

        Construct a state from the pulser abstract representation
        https://pulser.readthedocs.io/en/stable/conventions.html

        Args:
            eigenstates: A tuple containing the basis states (e.g., ('r', 'g')).
            amplitudes: A dictionary mapping state strings to complex or floats
            amplitudes.

        Returns:
            The normalised resulting state.

        Examples:
            ```python
            basis = ("r","g")
            st = StateVector.from_state_amplitudes(
                eigenstates=basis,
                amplitudes={"rr": 1.0, "gg": 1.0}
            )
            print(st)
            ```

            Output:
            ```
            tensor([0.7071+0.j, 0.0000+0.j, 0.0000+0.j, 0.7071+0.j],
                   dtype=torch.complex128)
            ```
        """

        accum_state = torch.zeros(
            2**n_qudits,
            dtype=torch.complex128,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        for state, amplitude in amplitudes.items():
            bin_to_int = int(state.replace("r", "1").replace("g", "0"), 2)
            accum_state[bin_to_int] = amplitude  # type: ignore [assignment]

        accum_state /= accum_state.norm()

        return accum_state

    @classmethod
    def _from_state_amplitudes(
        cls: Type[DenseStateType],
        *,
        eigenstates: Sequence[Eigenstate],
        n_qudits: int,
        amplitudes: Mapping[str, complex],
    ) -> tuple[DenseStateType, Mapping[str, complex]]:
        """Transforms a state given by a string into a state vector.

        Construct a state from the pulser abstract representation
        https://pulser.readthedocs.io/en/stable/conventions.html

        Args:
            eigenstates: A tuple containing the basis states (e.g., ('r', 'g')).
            amplitudes: A dictionary mapping state strings to complex or floats
            amplitudes.

        Returns:
            The normalised resulting state.

        Examples:
            ```python
            basis = ("r","g")
            st = DenseState.from_state_amplitudes(
                eigenstates=basis,
                amplitudes={"rr": 1.0, "gg": 1.0}
            )
            print(st)
            ```

            Output:
            ```
            tensor([0.7071+0.j, 0.0000+0.j, 0.0000+0.j, 0.7071+0.j],
                   dtype=torch.complex128)
            ```
        """
        basis = set(eigenstates)
        if basis == {"r", "g"}:
            pass
        elif basis == {"0", "1"}:
            raise NotImplementedError(
                "{'0','1'} basis is related to XY Hamiltonian, which is not implemented"
            )
        else:
            raise ValueError("Unsupported basis provided")

        sv = cls._build_state_vector(n_qudits=n_qudits, amplitudes=amplitudes)

        return cls(sv.data, gpu=sv.data.is_cuda, eigenstates=eigenstates), amplitudes
