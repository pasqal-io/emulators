import torch
from emu_ct import MPS
from .utils_testing import (
    ghz_state_factors,
    pulser_afm_sequence_ring,
    simulate_pulser_sequence,
)
import numpy as np

seed = 1337  # any number will do
device = "cpu"
dtype = torch.complex128


def test_sampling_ghz5_mps():
    device = "cpu"
    torch.manual_seed(seed)
    num_qubits = 5
    shots = 1000
    ghz_mps = MPS(ghz_state_factors(num_qubits, device=device))
    bitstring = ghz_mps.sample_mps(shots)

    assert bitstring.get("11111") == 505
    assert bitstring.get("00000") == 495


def test_not_orthogonalized_state():
    torch.manual_seed(seed)
    device = "cpu"
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
    bell = MPS([l_factor1, l_factor2, l_factor3])
    bitstring = bell.sample_mps(shots, truncate=True)

    assert bitstring.get("111") == 489
    assert bitstring.get("000") == 511


def simulate_afm_ring_state(num_qubits: int):
    Omega_max = 4 * 2 * np.pi
    U = Omega_max / 2
    delta_0 = -6 * U
    delta_f = 2 * U
    t_rise = 500
    t_fall = 1000

    puls_discre, reg = pulser_afm_sequence_ring(
        num_qubits=num_qubits,
        Omega_max=Omega_max,
        U=U,
        delta_0=delta_0,
        delta_f=delta_f,
        t_rise=t_rise,
        t_fall=t_fall,
    )

    return simulate_pulser_sequence(puls_discre, reg)


def test_afm_ring_mps_tdvp_sampling():
    torch.manual_seed(seed)

    num_qubits = 10
    state = simulate_afm_ring_state(num_qubits)

    shots = 1000
    bitstrings = state.sample_mps(shots)

    assert bitstrings["1010101010"] == 148
    assert bitstrings["0101010101"] == 151
