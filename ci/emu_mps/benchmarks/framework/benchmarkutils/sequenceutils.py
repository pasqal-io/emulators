import math
from pulser import Pulse, Sequence, Register
from pulser.waveforms import ConstantWaveform, RampWaveform, CompositeWaveform
from pulser.devices import MockDevice, AnalogDevice


# All sequences use MockDevice to be free from device-related spacing constraints.


def make_adiabatic_afm_state_2d_seq(
    rows: int, columns: int, perm_map: list = None
) -> Sequence:
    # from https://pulser.readthedocs.io/en/stable/tutorials/afm_prep.html
    # parameters in rad/µs and ns
    Omega_max = 2.0 * 2 * math.pi
    U = Omega_max / 2.0

    delta_0 = -6 * U
    delta_f = 2 * U

    t_rise = 500
    t_fall = 1000
    t_sweep = (delta_f - delta_0) / (2 * math.pi * 10) * 3000

    R_interatomic = MockDevice.rydberg_blockade_radius(U)
    reg = Register.rectangle(rows, columns, R_interatomic, prefix="q")
    if perm_map:
        reg_coords = reg._coords
        reg = Register.from_coordinates([reg_coords[i] for i in perm_map])

    rise = Pulse.ConstantDetuning(RampWaveform(t_rise, 0.0, Omega_max), delta_0, 0.0)
    sweep = Pulse.ConstantAmplitude(
        Omega_max, RampWaveform(t_sweep, delta_0, delta_f), 0.0
    )
    fall = Pulse.ConstantDetuning(RampWaveform(t_fall, Omega_max, 0.0), delta_f, 0.0)

    seq = Sequence(reg, MockDevice)
    seq.declare_channel("ising", "rydberg_global")

    seq.add(rise, "ising")
    seq.add(sweep, "ising")
    seq.add(fall, "ising")

    return seq


def make_quench_2d_seq(nx: int, ny: int) -> Sequence:
    # Hamiltonian parameters as ratios of J_max
    hx = 1.5  # hx/J_max
    hz = 0  # hz/J_max
    t = 1.5  # t/J_max

    # Set up pulser simulations
    R = 7  # microm
    reg = Register.rectangle(nx, ny, R, prefix="q")

    # Conversion from Rydberg Hamiltonian to Ising model
    U = AnalogDevice.interaction_coeff / R**6  # U_ij
    NN_coeff = U / 4
    omega = 2 * hx * NN_coeff
    delta = -2 * hz * NN_coeff + 2 * U
    T = round(1000 * t / NN_coeff)

    seq = Sequence(reg, MockDevice)
    seq.declare_channel("ising", "rydberg_global")
    seq.target([1, 2, 3, 4, 5], "ising")

    simple_pulse = Pulse.ConstantPulse(T, omega, delta, 0)
    seq.add(simple_pulse, "ising")
    return seq


def make_adiabatic_afm_state_1d_seq(N: int, perm_map=None) -> Sequence:
    # from https://pulser.readthedocs.io/en/stable/tutorials/1D_crystals.html
    # parameters in rad/µs and ns
    Omega_max = 2 * 2 * math.pi
    delta_0 = -6 * 2 * math.pi
    delta_f = 10 * 2 * math.pi
    t_rise = 500
    t_stop = 4500

    R_blockade = MockDevice.rydberg_blockade_radius(Omega_max)

    reg = Register.rectangle(1, N, spacing=R_blockade, prefix="q")
    if perm_map:
        reg = Register.from_coordinates([reg._coords[i] for i in perm_map])

    hold = ConstantWaveform(t_rise, delta_0)
    excite = RampWaveform(t_stop - t_rise, delta_0, delta_f)
    sweep = Pulse.ConstantAmplitude(Omega_max, CompositeWaveform(hold, excite), 0.0)

    seq = Sequence(reg, MockDevice)
    seq.declare_channel("ising", "rydberg_global")
    seq.add(sweep, "ising")

    return seq
