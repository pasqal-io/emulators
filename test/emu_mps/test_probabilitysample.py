import torch

from emu_mps import MPS

from utils_testing import cpu_multinomial_wrapper, ghz_state_factors

from unittest.mock import patch

seed = 1337  # any number will do
device = "cpu"  # 'cuda'

dtype = torch.complex128


def test_sampling_ghz5_mps():
    torch.manual_seed(seed)
    num_qubits = 5
    shots = 1000
    ghz_mps = MPS(
        ghz_state_factors(num_qubits, device=device),
        eigenstates=("0", "1"),
    )
    # Mock torch.multinomial inside the MPS.sample
    with patch("emu_mps.mps.torch.multinomial", side_effect=cpu_multinomial_wrapper):
        bitstrings = ghz_mps.sample(num_shots=shots)

    assert bitstrings.get("11111") == 463
    assert bitstrings.get("00000") == 537


def test_not_orthogonalized_state():
    torch.manual_seed(seed)
    shots = 1000
    # right orthogonalized mps
    l_factor1 = torch.tensor([[[1, 0], [0, 1j]]], dtype=torch.complex128, device=device)
    l_factor2 = torch.tensor(
        [[[1, 0], [0, 0]], [[0, 0], [0, 1]]], dtype=torch.complex128, device=device
    )
    # the orthogonality center is at the end
    l_factor3 = (
        1
        / torch.sqrt(torch.tensor(2))
        * torch.tensor([[[1], [0]], [[0], [1]]], dtype=torch.complex128, device=device)
    )
    bell = MPS(
        [l_factor1, l_factor2, l_factor3],
        eigenstates=("0", "1"),
    )
    # Mock torch.multinomial inside the MPS.sample
    with patch("emu_mps.mps.torch.multinomial", side_effect=cpu_multinomial_wrapper):
        bitstrings = bell.sample(num_shots=shots)

    assert bitstrings.get("111") == 499
    assert bitstrings.get("000") == 501


def test_sampling_wall():
    torch.manual_seed(seed)
    shots = 1000
    natoms = 4  # half the number of qubits

    factor1 = torch.tensor([[[0], [1]]], dtype=torch.complex128, device=device)
    factor2 = torch.tensor([[[1.0], [0.0]]], dtype=torch.complex128, device=device)
    wall_state = [factor1] * natoms + [factor2] * natoms

    state_wall = MPS(wall_state, eigenstates=("0", "1"))
    bitstrings = state_wall.sample(num_shots=shots)
    state_bit = "1" * natoms + "0" * natoms

    assert bitstrings.get(state_bit) == 1000


def test_with_leakage():
    torch.manual_seed(seed)
    shots = 1000
    natoms = 4  # half the number of qubits

    ket1 = torch.tensor([[[0], [1], [0]]], dtype=torch.complex128, device=device)  # |r>
    ket0 = torch.tensor([[[1], [0], [0]]], dtype=torch.complex128, device=device)  # |g>
    wall_state = [ket1] * natoms + [ket0] * natoms  # wall state in rydberg basis

    # check manually check that x works with rydberg basis
    # decision: x, g, r order
    state_wall = MPS(wall_state, eigenstates=("g", "r", "x"))
    bitstrings = state_wall.sample(num_shots=shots)
    state_bit = "1" * natoms + "0" * natoms

    assert bitstrings.get(state_bit) == shots


def test_with_leakage_ghz_3level(mocked_results=None):
    torch.manual_seed(seed)

    shots = 1000
    natoms = 3  # half the number of qubits

    factorg = torch.tensor([[[1], [0], [0]]], dtype=torch.complex128, device=device)
    factorr = torch.tensor([[[0], [1], [0]]], dtype=torch.complex128, device=device)
    factorx = torch.tensor([[[0], [0], [1]]], dtype=torch.complex128, device=device)
    wall_state = (
        MPS([factorx] * natoms, eigenstates=("x", "g", "r"))
        + MPS([factorg] * natoms, eigenstates=("x", "g", "r"))
        + MPS([factorr] * natoms, eigenstates=("x", "g", "r"))
    )
    wall_state /= wall_state.norm()

    # Mock torch.multinomial inside the MPS.sample
    with patch("torch.multinomial", side_effect=cpu_multinomial_wrapper):
        bitstrings = wall_state.sample(num_shots=shots)

    state_bit0 = "0" * natoms
    state_bit1 = "1" * natoms

    assert bitstrings.get(state_bit0) == 653
    assert bitstrings.get(state_bit1) == 347


def test_with_leakage_edge_case_3level():
    "The output of state xr + xg + gg - gr"
    torch.manual_seed(seed)

    shots = 1000

    factorx = torch.tensor([[[0], [0], [1]]], dtype=torch.complex128, device=device)
    factorr = torch.tensor([[[0], [1], [0]]], dtype=torch.complex128, device=device)
    factorg = torch.tensor([[[1], [0], [0]]], dtype=torch.complex128, device=device)
    mpsxr = MPS([factorx, factorr], eigenstates=("x", "g", "r"))
    mpsxg = MPS([factorx, factorg], eigenstates=("x", "g", "r"))
    mpsgg = MPS([factorg, factorg], eigenstates=("x", "g", "r"))
    mpsgr = MPS([factorg, factorr], eigenstates=("x", "g", "r"))
    state = mpsxr + mpsxg + mpsgg + (-1.0) * mpsgr
    state /= state.norm()

    # torch.multinomial to use the CPU wrapper
    with patch("torch.multinomial", new=cpu_multinomial_wrapper):
        bitstrings = state.sample(num_shots=shots)

    assert bitstrings.get("00") == 502
    assert bitstrings.get("01") == 498
