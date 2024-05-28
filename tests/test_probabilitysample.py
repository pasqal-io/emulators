import torch
import emu_ct
from emu_ct.pulser_adapter import registers_to_pyemunt, extract_values_from_sequence
from emu_ct import MPS
from utils_testing import ghz_state_factors

import pulser

from pulser.waveforms import RampWaveform
from pulser.devices import MockDevice
from pulser.sampler.sampler import sample
import numpy as np

seed = 1337  # any number will do
device = "cpu"
dtype = torch.complex128


def test_sampling_bell5_mps():
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


def pulser_afm_sequence(num_qubits: int):
    # Setup

    Omega_max = 4 * 2 * np.pi
    U = Omega_max / 2
    delta_0 = -6 * U
    delta_f = 2 * U
    t_rise = 500
    t_fall = 1000
    t_sweep = (delta_f - delta_0) / (2 * np.pi * 10) * 1000

    # Define a ring of atoms distanced by a blockade radius distance:
    R_interatomic = MockDevice.rydberg_blockade_radius(U)
    coords = (
        R_interatomic
        / (2 * np.tan(np.pi / num_qubits))
        * np.array(
            [
                (
                    np.cos(theta * 2 * np.pi / num_qubits),
                    np.sin(theta * 2 * np.pi / num_qubits),
                )
                for theta in range(num_qubits)
            ]
        )
    )

    reg = pulser.Register.from_coordinates(coords, prefix="q")
    rise = pulser.Pulse.ConstantDetuning(
        RampWaveform(t_rise, 0.0, Omega_max), delta_0, 0.0
    )
    sweep = pulser.Pulse.ConstantAmplitude(
        Omega_max, RampWaveform(t_sweep, delta_0, delta_f), 0.0
    )
    fall = pulser.Pulse.ConstantDetuning(
        RampWaveform(t_fall, Omega_max, 0.0), delta_f, 0.0
    )

    seq = pulser.Sequence(reg, pulser.devices.MockDevice)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(rise, "ising_global")
    seq.add(sweep, "ising_global")
    seq.add(fall, "ising_global")
    puls_discre = sample(seq)

    return puls_discre, reg


def test_afm_mps_tdvp_sampling():

    num_qubits = 10
    shots = 1000
    torch.manual_seed(seed)
    puls_discre, reg = pulser_afm_sequence(num_qubits)
    state = MPS(
        [(torch.tensor([1.0, 0.0]).reshape(1, 2, 1).to(dtype=torch.complex128))]
        * len(reg.qubits)
    )
    reg_test = registers_to_pyemunt(reg)
    dt: int = 100
    coeff = 0.001

    i = 0
    while (
        i < puls_discre.max_duration
    ):  # TODO: this while should be converted into a run function as in Pulser
        ampli_test, detu_test = extract_values_from_sequence(
            puls_discre.channel_samples, reg, i
        )
        mpo_t0 = emu_ct.make_H(reg_test, ampli_test, detu_test)
        emu_ct.tdvp(-dt * coeff * 1j, state, mpo_t0)
        i += dt

    bitstrings = state.sample_mps(shots)

    assert bitstrings["1010101010"] == 148
    assert bitstrings["0101010101"] == 151
