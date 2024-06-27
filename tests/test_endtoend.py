import torch
from emu_ct import MPS, MPSBackend, MPSConfig, BitStrings, StateResult

from .utils_testing import pulser_afm_sequence_ring

import numpy as np
from pytest import approx


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
    final_time = seq.get_duration()
    mpsConfig = MPSConfig(
        dt=100,
        precision=1e-5,
        observables=[
            StateResult(times=[final_time]),
            BitStrings(times=[final_time], num_shots=1000),
        ],
    )
    sim = MPSBackend()

    return sim.run(seq, mpsConfig), seq


def test_afm_ring_mps_tdvp_sampling():
    torch.manual_seed(seed)

    num_qubits = 10
    result, seq = simulate_afm_ring_state(num_qubits)
    final_time = seq.get_duration()
    bitstrings = result[BitStrings.name()][final_time]
    final_state = result[StateResult.name()][final_time]
    max_bond_dim = final_state.get_max_bond_dim()

    assert bitstrings["1010101010"] == 148
    assert bitstrings["0101010101"] == 151
    assert max_bond_dim == 29

    one = torch.tensor([[[0], [1]]], dtype=torch.complex128)
    zero = torch.tensor([[[1], [0]]], dtype=torch.complex128)

    expected_state1 = MPS([one, zero] * (num_qubits // 2))
    inner_sol1 = expected_state1.inner(final_state)

    expected_state2 = MPS([zero, one] * (num_qubits // 2))
    inner_sol2 = expected_state2.inner(final_state)

    assert abs(inner_sol1) ** 2 == approx(0.15, abs=1e-2)

    assert abs(inner_sol2) ** 2 == approx(0.15, abs=1e-2)
