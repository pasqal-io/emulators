import torch

from emu_mps.mps import MPS
from emu_mps.mpo import MPO


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
        self._right = self._right_baths(state, hamiltonian, final_qubit=2)
        assert len(self._right) == len(state.factors) - 1

    def current_left(self) -> torch.Tensor:
        return self._left[-1]

    def current_right(self) -> torch.Tensor:
        return self._right[-1]

    def current(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.current_left(), self.current_right()

    def _new_left_bath(
        self,
        bath: torch.Tensor,
        state: torch.Tensor,
        op: torch.Tensor,
    ) -> torch.Tensor:
        # this order is more efficient than contracting the op first in general
        bath = torch.tensordot(bath, state.conj(), ([0], [0]))
        bath = torch.tensordot(bath, op.to(bath.device), ([0, 2], [0, 1]))
        bath = torch.tensordot(bath, state, ([0, 2], [0, 1]))
        return bath

    def append_left(self, state: MPS, hamiltonian: MPO, sweep_index: int) -> None:
        new_node = self._new_left_bath(
            self.current_left(),
            state.factors[sweep_index],
            hamiltonian.factors[sweep_index],
        )
        self._left.append(new_node.to(state.factors[sweep_index + 1].device))
        return

    def _new_right_bath(
        self, bath: torch.Tensor, state: torch.Tensor, op: torch.Tensor
    ) -> torch.Tensor:
        bath = torch.tensordot(state, bath, ([2], [2]))
        bath = torch.tensordot(op.to(bath.device), bath, ([2, 3], [1, 3]))
        bath = torch.tensordot(state.conj(), bath, ([1, 2], [1, 3]))
        return bath

    def _right_baths(
        self,
        state: MPS,
        op: MPO,
        final_qubit: int,
    ) -> list[torch.Tensor]:
        """
        function to compute the right baths. The three indices in the bath are as follows:
        (bond of state conj, bond of operator, bond of state)
        The baths have shape
        -xx
        -xx
        -xx
        with the index ordering (top, middle, bottom)
        bath tensors are put on the device of the factor to the left
        """

        state_factor = state.factors[-1]
        bath = torch.ones(1, 1, 1, device=state_factor.device, dtype=state_factor.dtype)
        baths = [bath]
        for i in range(len(state.factors) - 1, final_qubit - 1, -1):
            bath = self._new_right_bath(bath, state.factors[i], op.factors[i])
            bath = bath.to(state.factors[i - 1].device)
            baths.append(bath)
        return baths

    def append_right(self, state: MPS, hamiltonian: MPO, sweep_index: int) -> None:
        self._right.append(
            self._new_right_bath(
                self.current_right(),
                state.factors[sweep_index],
                hamiltonian.factors[sweep_index],
            ).to(state.factors[sweep_index - 1].device)
        )

    def pop_left(self) -> torch.Tensor:
        return self._left.pop()

    def pop_right(self) -> torch.Tensor:
        return self._right.pop()
