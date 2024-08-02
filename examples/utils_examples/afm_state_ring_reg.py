import pulser
import numpy as np


def afm_sequence_from_register(
    reg: pulser.Register,
    Omega_max: float,
    delta_0: float,
    delta_f: float,
    t_rise: float,
    t_fall: float,
    factor_sweep: int,
    device: pulser.devices = pulser.devices.MockDevice,
):
    """Sequence that creates AntiFerromagnetic State (AFM) using pulser
    This function constructs a sequence of pulses to transition a system of qubits
    (represented by `reg`) into an AFM state using a specified device. The sequence
    consists of three phases: a rise, a sweep, and a fall."""

    t_sweep = (delta_f - delta_0) / (2 * np.pi * 10) * 1000 * factor_sweep
    print(t_sweep)
    rise = pulser.Pulse.ConstantDetuning(
        pulser.waveforms.RampWaveform(t_rise, 0.0, Omega_max), delta_0, 0.0
    )
    sweep = pulser.Pulse.ConstantAmplitude(
        Omega_max, pulser.waveforms.RampWaveform(t_sweep, delta_0, delta_f), 0.0
    )
    fall = pulser.Pulse.ConstantDetuning(
        pulser.waveforms.RampWaveform(t_fall, Omega_max, 0.0), delta_f, 0.0
    )

    seq = pulser.Sequence(reg, device)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(rise, "ising_global")
    seq.add(sweep, "ising_global")
    seq.add(fall, "ising_global")

    return seq
