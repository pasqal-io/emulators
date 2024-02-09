import torch
from .mps import MPS
from .mpo import MPO

"""
function to compute the left baths. The three indices in the bath are as follows:
(bond of state conj, bond of operator, bond of state)
"""


def left_baths(state: MPS, op: MPO) -> list[torch.Tensor]:
    baths = []
    bath_shape = op.factors[0].shape
    # remove left bond which has dim 1, and order indices per docstring
    bath = op.factors[0].reshape(bath_shape[1:]).transpose(1, 2)
    for i in range(len(state.factors) - 1):
        state_factor = state.factors[i]
        right_bond = state_factor.shape[-1]
        state_factor = state_factor.reshape(-1, right_bond)
        bath = torch.tensordot(state_factor.conj(), bath, ([0], [0]))
        bath = (
            torch.tensordot(bath, state_factor, ([2], [0]))
            .reshape(right_bond, -1, right_bond)
            .to(state.factors[i + 1].device)
        )
        baths.append(bath)
        if i != len(state.factors) - 2:
            bath = (
                torch.tensordot(bath, op.factors[i + 1], ([1], [0]))
                .permute(0, 2, 4, 1, 3)
                .reshape(2 * right_bond, -1, 2 * right_bond)
            )
    return baths


"""
function to compute the right baths. The three indices in the bath are as follows:
(bond of state conj, bond of operator, bond of state)
"""


def right_baths(state: MPS, op: MPO) -> list[torch.Tensor]:
    baths = []
    bath_shape = op.factors[-1].shape
    # remove left bond which has dim 1, and order indices per docstring
    bath = op.factors[-1].reshape(bath_shape[:3]).transpose(0, 1)
    for i in range(len(state.factors) - 1, 0, -1):
        state_factor = state.factors[i]
        left_bond = state_factor.shape[0]
        state_factor = state_factor.reshape(left_bond, -1)
        bath = torch.tensordot(state_factor.conj(), bath, ([1], [0]))
        bath = (
            torch.tensordot(bath, state_factor, ([2], [1]))
            .reshape(left_bond, -1, left_bond)
            .to(state.factors[i - 1].device)
        )
        baths.append(bath)
        if i != 1:
            bath = (
                torch.tensordot(op.factors[i - 1], bath, ([3], [1]))
                .permute(1, 3, 0, 2, 4)
                .reshape(2 * left_bond, -1, 2 * left_bond)
            )

    return baths
