from emu_ct.pulser_adapter import (
    extract_values_from_channel,
    extract_values_from_sequence,
    slot_target_to_positions,
    registers_to_pyemunt,
)
import torch
from pulser import Register, Pulse, Sequence
from pulser.devices import MockDevice
from pulser.sampler.sampler import sample
from pulser.waveforms import RampWaveform
import numpy as np


def discretize_sequence(L: int):
    """Sequence creation and discretization. It is a short duration where
    L is the number of atoms"""
    Omega_max = 2.3 * 2 * np.pi
    U = Omega_max / 2.3
    delta_0 = -3 * U
    delta_f = 1 * U
    t_rise = 5
    t_fall = 5
    t_sweep = (delta_f - delta_0) / (2 * np.pi * 10) * 4
    # Define a ring of atoms distanced by a blockade radius distance:
    R_interatomic = MockDevice.rydberg_blockade_radius(U)
    # define the coordinates in order to locate the atoms
    coords = (
        R_interatomic
        / (2 * np.tan(np.pi / L))
        * np.array(
            [
                (np.cos(theta * 2 * np.pi / L), np.sin(theta * 2 * np.pi / L))
                for theta in range(L)
            ]
        )
    )

    # create the register according to positions a of atoms
    reg = Register.from_coordinates(coords, prefix="q")

    # create pulses with/out waveforms
    rise = Pulse.ConstantDetuning(RampWaveform(t_rise, 0.0, Omega_max), delta_0, 0.0)
    sweep = Pulse.ConstantAmplitude(
        Omega_max, RampWaveform(t_sweep, delta_0, delta_f), 0.0
    )
    fall = Pulse.ConstantDetuning(RampWaveform(t_fall, Omega_max, 0.0), delta_f, 0.0)
    conste_amp = Pulse.ConstantAmplitude(
        Omega_max, RampWaveform(t_rise, Omega_max, Omega_max), 0.0
    )

    # create sequences with local and global pulses
    seq = Sequence(reg, MockDevice)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.declare_channel("ch_local1", "rydberg_local", initial_target="q0")
    seq.declare_channel("ch_local2", "rydberg_local", initial_target="q1")
    seq.add(conste_amp, "ch_local1")
    seq.add(rise, "ch_local2")
    seq.add(rise, "ising_global")
    seq.add(sweep, "ising_global")
    seq.add(fall, "ising_global")
    seq.target("q2", "ch_local1")
    seq.add(rise, "ch_local1")
    seq.add(sweep, "ch_local2", protocol="wait-for-all")
    return sample(seq), reg


def test_register_pyemutn():
    L = 5
    _, reg = discretize_sequence(L)

    reg_pyemutn = registers_to_pyemunt(reg)

    reg_solution = [
        [6.714509878096849, 0.0],
        [2.0748976612303833, 6.3858783732921856],
        [-5.4321526002788065, 3.94668988271746],
        [-5.432152600278808, -3.946689882717459],
        [2.0748976612303816, -6.385878373292186],
    ]
    for num in range(len(reg_pyemutn)):
        assert reg_pyemutn[num].x == reg_solution[num][0]
        assert reg_pyemutn[num].y == reg_solution[num][1]


def test_slot_to_target():

    L = 4  # number the atoms to test
    puls_discre, reg = discretize_sequence(L)

    test_position = slot_target_to_positions(
        puls_discre.channel_samples["ch_local2"].slots[0].targets, reg.qubit_ids
    )

    assert test_position == [1]

    test_position = slot_target_to_positions(
        puls_discre.channel_samples["ch_local2"].slots[1].targets, reg.qubit_ids
    )

    assert test_position == [1]

    test_position = slot_target_to_positions(
        puls_discre.channel_samples["ch_local1"].slots[0].targets, reg.qubit_ids
    )
    assert test_position == [0]

    test_position = slot_target_to_positions(
        puls_discre.channel_samples["ch_local1"].slots[1].targets, reg.qubit_ids
    )
    assert test_position == [2]

    test_position = slot_target_to_positions(
        puls_discre.channel_samples["ising_global"].slots[0].targets, reg.qubit_ids
    )
    assert test_position == [0, 1, 2, 3]

    test_position = slot_target_to_positions(
        puls_discre.channel_samples["ising_global"].slots[1].targets, reg.qubit_ids
    )
    assert test_position == [0, 1, 2, 3]


def test_channel_amp_det_t_0():
    # local channel test
    L = 3
    puls_discre, reg = discretize_sequence(L)
    # test at  t= 0
    t = 0
    ret_amp: list[float] = torch.zeros(
        len(reg.qubit_ids), dtype=torch.complex128
    )  # initialize amplitude values
    ret_det: list[float] = torch.zeros(
        len(reg.qubit_ids), dtype=torch.complex128
    )  # initialize detuning values
    amp_solution = torch.tensor(
        [14.4513, 0.0, 0.0], dtype=torch.complex128
    )  # comming from pulser local channel
    det_solution = torch.tensor(
        [14.4513, 0.0, 0.0], dtype=torch.complex128
    )  # comming from pulser local channel
    extract_values_from_channel(
        puls_discre.channel_samples["ch_local1"], reg, ret_amp, ret_det, t
    )

    torch.testing.assert_close(ret_amp, amp_solution, atol=1e-3, rtol=1e-05)

    torch.testing.assert_close(ret_det, det_solution, atol=1e-3, rtol=1e-05)


def test_channel_amp_det_t_14():
    # global channel test
    L = 4
    puls_discre, reg = discretize_sequence(L)
    t = 14
    ret_amp: list[float] = torch.zeros(
        len(reg.qubit_ids), dtype=torch.complex128
    )  # initialize amp
    ret_det: list[float] = torch.zeros(
        len(reg.qubit_ids), dtype=torch.complex128
    )  # initialize det
    amp_solution = torch.tensor(
        [3.61283155, 3.61283155, 3.61283155, 3.61283155],
        dtype=torch.complex128,
    )  # comming from pulser global channel
    det_solution = torch.tensor(
        [6.28318531, 6.28318531, 6.28318531, 6.28318531],
        dtype=torch.complex128,
    )  # comming from pulser global channel
    extract_values_from_channel(
        puls_discre.channel_samples["ising_global"], reg, ret_amp, ret_det, t
    )

    torch.testing.assert_close(ret_amp, amp_solution, atol=0.0, rtol=1e-05)

    torch.testing.assert_close(ret_det, det_solution, atol=1e-3, rtol=1e-05)


def test_channel_amp_det_t_21():
    # local channel test
    L = 3
    puls_discre, reg = discretize_sequence(L)
    t = 21
    ret_amp: list[float] = torch.zeros(
        len(reg.qubit_ids), dtype=torch.complex128
    )  # initialize amplitude values
    ret_det: list[float] = torch.zeros(
        len(reg.qubit_ids), dtype=torch.complex128
    )  # initialize detuning values
    amp_solution = torch.tensor(
        [0.0, 14.45132621, 0.0], dtype=torch.complex128
    )  # comming from pulser local channel
    det_solution = torch.tensor(
        [0.0, -18.84955592, 0.0], dtype=torch.complex128
    )  # comming from pulser local channel
    extract_values_from_channel(
        puls_discre.channel_samples["ch_local2"], reg, ret_amp, ret_det, t
    )

    torch.testing.assert_close(ret_amp, amp_solution, atol=1e-3, rtol=1e-05)

    torch.testing.assert_close(ret_det, det_solution, atol=1e-3, rtol=1e-05)


def test_channel_amp_det_t_21_different_channel():
    # local channel test
    L = 3
    puls_discre, reg = discretize_sequence(L)
    t = 21
    ret_amp: list[float] = torch.zeros(
        len(reg.qubit_ids), dtype=torch.complex128
    )  # initialize amplitude values
    ret_det: list[float] = torch.zeros(
        len(reg.qubit_ids), dtype=torch.complex128
    )  # initialize detuning values
    amp_solution = torch.tensor(
        [0.0, 0.0, 0.0], dtype=torch.complex128
    )  # comming from pulser local channel
    det_solution = torch.tensor(
        [0.0, 0.0, 0.0], dtype=torch.complex128
    )  # comming from pulser local channel
    extract_values_from_channel(
        puls_discre.channel_samples["ch_local1"], reg, ret_amp, ret_det, t
    )

    torch.testing.assert_close(ret_amp, amp_solution, atol=1e-3, rtol=1e-05)

    torch.testing.assert_close(ret_det, det_solution, atol=1e-3, rtol=1e-05)


def test_sequence_amp_det_t_1():
    """ " Extracting amplitude and detuning from the whole sequence given a certain time"""
    L = 4
    puls_discre, reg = discretize_sequence(L)  # test for t= 1
    t = 1
    amp_solution = torch.tensor(
        [14.45132621, 3.61283155, 0.0, 0.0], dtype=torch.complex128
    )
    det_solution = torch.tensor(
        [14.45132621, -18.84955592, 0.0, 0.0], dtype=torch.complex128
    )
    dis_amp, dis_det = extract_values_from_sequence(puls_discre.channel_samples, reg, t)

    torch.testing.assert_close(dis_amp, amp_solution, atol=1e-3, rtol=1e-05)

    torch.testing.assert_close(dis_det, det_solution, atol=1e-3, rtol=1e-05)


def test_sequence_amp_det_t_5():
    # testing the whole sequence
    L = 3
    puls_discre, reg = discretize_sequence(L)
    # test for some values at t= 5
    t = 5
    amp_solution = torch.tensor([0.0, 0.0, 0.0], dtype=torch.complex128)
    det_solution = torch.tensor(
        [-18.84955592, -18.84955592, -18.84955592], dtype=torch.complex128
    )
    dis_amp, dis_det = extract_values_from_sequence(puls_discre.channel_samples, reg, t)

    torch.testing.assert_close(dis_amp, amp_solution, atol=1e-3, rtol=1e-05)

    torch.testing.assert_close(dis_det, det_solution, atol=1e-3, rtol=1e-05)


def test_sequence_amp_det_t_6():
    # local channel test
    L = 3
    puls_discre, reg = discretize_sequence(L)
    # test for some values at t= 6
    t = 6
    amp_solution = torch.tensor(
        [3.61283155, 3.61283155, 3.61283155], dtype=torch.complex128
    )
    det_solution = torch.tensor(
        [-18.84955592, -18.84955592, -18.84955592], dtype=torch.complex128
    )
    dis_amp, dis_det = extract_values_from_sequence(puls_discre.channel_samples, reg, t)

    torch.testing.assert_close(dis_amp, amp_solution, atol=1e-3, rtol=1e-05)

    torch.testing.assert_close(dis_det, det_solution, atol=1e-3, rtol=1e-05)


def test_sequence_amp_det_t_21():
    # local channel test
    L = 4
    puls_discre, reg = discretize_sequence(L)
    # test for some values at t= 21
    t = 21
    amp_solution = torch.tensor([0.0, 14.45132621, 0.0, 0.0], dtype=torch.complex128)
    det_solution = torch.tensor([0.0, -18.84955592, 0.0, 0.0], dtype=torch.complex128)
    dis_amp, dis_det = extract_values_from_sequence(puls_discre.channel_samples, reg, t)

    torch.testing.assert_close(dis_amp, amp_solution, atol=1e-3, rtol=1e-05)

    torch.testing.assert_close(dis_det, det_solution, atol=1e-3, rtol=1e-05)
