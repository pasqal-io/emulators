from emu_ct.tdvp import left_baths, right_baths
from emu_ct import MPS, MPO
import torch


def test_left_baths_bell():
    # state = (|0> + |1>)^3 / norm
    mps_factor1 = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.complex128)
    mps_factor2 = torch.tensor(
        [[[1, 0], [0, 0]], [[0, 0], [0, 1]]], dtype=torch.complex128
    )
    mps_factor3 = torch.tensor([[[1], [0]], [[0], [1]]], dtype=torch.complex128)

    # Hamiltonian X1*X2*X3
    mpo_factor = torch.tensor([[[[0], [1]], [[1], [0]]]], dtype=torch.complex128)

    state = MPS([mps_factor1, mps_factor2, mps_factor3])
    obs = MPO([mpo_factor] * 3)
    for b in left_baths(state, obs):
        # Because the Hamiltonian flips all the spins, the baths have shape
        # (2,1,2) and they're all pauli_x
        assert torch.allclose(
            b, torch.tensor([[[0, 1]], [[1, 0]]], dtype=torch.complex128)
        )


def test_left_baths_total_magnetization():
    # Hamiltonian Z1+Z2+Z3
    mpo_factor1 = torch.tensor(
        [[[[1, 1], [0, 0]], [[0, 0], [-1, 1]]]], dtype=torch.complex128
    )
    mpo_factor2 = torch.tensor(
        [[[[1, 0], [0, 0]], [[0, 0], [1, 0]]], [[[1, 1], [0, 0]], [[0, 0], [-1, 1]]]],
        dtype=torch.complex128,
    )
    mpo_factor3 = torch.tensor(
        [[[[1], [0]], [[0], [1]]], [[[1], [0]], [[0], [-1]]]], dtype=torch.complex128
    )

    # state |111>
    mps_factor = torch.tensor([[[0], [1]]], dtype=torch.complex128)

    state = MPS([mps_factor] * 3)
    obs = MPO([mpo_factor1, mpo_factor2, mpo_factor3])
    baths = left_baths(state, obs)
    # The baths carry the information of the magnetization, so the baths have shape
    # (1,2,1), and L_i = [-i,1], which basically counts how magnetized the bath is.
    assert torch.allclose(baths[0], torch.tensor([[[-1], [1]]], dtype=torch.complex128))
    assert torch.allclose(baths[1], torch.tensor([[[-2], [1]]], dtype=torch.complex128))


def test_right_baths_bell():
    # state = (|0> + |1>)^3 / norm
    mps_factor1 = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.complex128)
    mps_factor2 = torch.tensor(
        [[[1, 0], [0, 0]], [[0, 0], [0, 1]]], dtype=torch.complex128
    )
    mps_factor3 = torch.tensor([[[1], [0]], [[0], [1]]], dtype=torch.complex128)

    # Hamiltonian X1*X2*X3
    mpo_factor = torch.tensor([[[[0], [1]], [[1], [0]]]], dtype=torch.complex128)

    state = MPS([mps_factor1, mps_factor2, mps_factor3])
    obs = MPO([mpo_factor] * 3)
    for b in right_baths(state, obs):
        # Because the Hamiltonian flips all the spins, the baths have shape
        # (2,1,2) and they're all pauli_x
        assert torch.allclose(
            b, torch.tensor([[[0, 1]], [[1, 0]]], dtype=torch.complex128)
        )


def test_right_baths_total_magnetization():
    # Hamiltonian Z1+Z2+Z3
    mpo_factor1 = torch.tensor(
        [[[[1, 1], [0, 0]], [[0, 0], [-1, 1]]]], dtype=torch.complex128
    )
    mpo_factor2 = torch.tensor(
        [[[[1, 0], [0, 0]], [[0, 0], [1, 0]]], [[[1, 1], [0, 0]], [[0, 0], [-1, 1]]]],
        dtype=torch.complex128,
    )
    mpo_factor3 = torch.tensor(
        [[[[1], [0]], [[0], [1]]], [[[1], [0]], [[0], [-1]]]], dtype=torch.complex128
    )

    # state |111>
    mps_factor = torch.tensor([[[0], [1]]], dtype=torch.complex128)

    state = MPS([mps_factor] * 3)
    obs = MPO([mpo_factor1, mpo_factor2, mpo_factor3])
    baths = right_baths(state, obs)
    # The baths carry the information of the magnetization, so the baths have shape
    # (1,2,1), and R_i = [1,-i], which basically counts how magnetized the bath is.
    assert torch.allclose(baths[0], torch.tensor([[[1], [-1]]], dtype=torch.complex128))
    assert torch.allclose(baths[1], torch.tensor([[[1], [-2]]], dtype=torch.complex128))
