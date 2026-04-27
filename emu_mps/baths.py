import torch

from emu_mps.mps import MPS
from emu_mps.mpo import MPO
from emu_mps.solver_utils import (
    new_right_bath,
    right_baths,
)
from emu_mps.utils import (
    new_left_bath,
)


class Baths:
    """
    Helper container managing left and right environment ("bath") tensors
    used during MPS sweeps.

    Maintains stacks of left and right baths and provides utilities to
    update them during left-to-right and right-to-left sweeps.
    """

    _left: list[torch.Tensor]
    _right: list[torch.Tensor]

    def __init__(
        self,
        state: MPS,
        hamiltonian: MPO,
    ):
        """
        Initialize baths from the given MPS state and Hamiltonian.
        Left baths start with a trivial identity-like tensor.
        Right baths are initialized from the full system.
        """
        self.device = state.factors[0].device
        self.dtype = state.factors[0].dtype

        self._left = [torch.ones(1, 1, 1, dtype=self.dtype, device=self.device)]
        self._right = right_baths(state, hamiltonian, final_qubit=2)
        assert len(self._right) == len(state.factors) - 1

    def current_left(self) -> torch.Tensor:
        return self._left[-1]

    def current_right(self) -> torch.Tensor:
        return self._right[-1]

    def current(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.current_left(), self.current_right()

    def append_left(self, state: MPS, hamiltonian: MPO, sweep_index: int) -> None:
        self._left.append(
            new_left_bath(
                self.current_left(),
                state.factors[sweep_index],
                hamiltonian.factors[sweep_index],
            ).to(state.factors[sweep_index + 1].device)
        )
        return

    def append_right(self, state: MPS, hamiltonian: MPO, sweep_index: int) -> None:
        self._right.append(
            new_right_bath(
                self.current_right(),
                state.factors[sweep_index],
                hamiltonian.factors[sweep_index],
            ).to(state.factors[sweep_index - 1].device)
        )

    def pop_left(self) -> torch.Tensor:
        return self._left.pop()

    def pop_right(self) -> torch.Tensor:
        return self._right.pop()
