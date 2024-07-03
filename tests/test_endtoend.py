import torch
from emu_ct.MPSbackend import MPSBackend

from .utils_testing import pulser_afm_sequence_ring

import numpy as np

seed = 1337


def simulate_afm_ring_state(num_qubits: int):
    Omega_max = 4 * 2 * np.pi
    U = Omega_max / 2
    delta_0 = -6 * U
    delta_f = 2 * U
    t_rise = 500
    t_fall = 1000

    seq = pulser_afm_sequence_ring(
        num_qubits=num_qubits,
        Omega_max=Omega_max,
        U=U,
        delta_0=delta_0,
        delta_f=delta_f,
        t_rise=t_rise,
        t_fall=t_fall,
    )

    sim = MPSBackend(seq)

    return sim.run(dt=100)


def test_afm_ring_mps_tdvp_sampling():
    torch.manual_seed(seed)

    num_qubits = 10
    state = simulate_afm_ring_state(num_qubits)

    shots = 1000
    bitstrings = state.get_samples(shots)

    assert bitstrings["1010101010"] == 148
    assert bitstrings["0101010101"] == 151


test_afm_ring_mps_tdvp_sampling()
