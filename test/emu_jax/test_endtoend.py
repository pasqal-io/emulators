from emu_jax import JaxSVBackend

import pulser
from pulser.backend.default_observables import Energy, StateResult
import math
from emu_sv import SVConfig, SVBackend
import torch
import numpy as np


Omega_max = 4 * 2 * math.pi
U = Omega_max / 2
delta_0 = -6 * U
delta_f = 2 * U
t_rise = 500
t_fall = 1000
t_sweep = (delta_f - delta_0) / (2 * math.pi * 10) * 1000


rows = 2
cols = 3


def sequence():
    R_interatomic = pulser.devices.MockDevice.rydberg_blockade_radius(U)
    reg = pulser.Register.rectangle(rows, cols, R_interatomic, prefix="q")

    rise = pulser.Pulse.ConstantDetuning(
        pulser.waveforms.RampWaveform(t_rise, 0.0, Omega_max), delta_0, 0.0
    )
    sweep = pulser.Pulse.ConstantAmplitude(
        Omega_max,
        pulser.waveforms.RampWaveform(t_sweep, delta_0, delta_f),
        0.0,
    )
    fall = pulser.Pulse.ConstantDetuning(
        pulser.waveforms.RampWaveform(t_fall, Omega_max, 0.0), delta_f, 0.0
    )

    seq = pulser.Sequence(reg, pulser.devices.MockDevice)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(rise, "ising_global")
    seq.add(sweep, "ising_global")
    seq.add(fall, "ising_global")

    return seq



config = SVConfig(
    default_evaluation_times=(1.0,),
    observables=[
        Energy(evaluation_times=[0.1 * n for n in range(11)]),
    ],
    dt=40,
    krylov_tolerance=1e-7
)


def test_grid_energy():
    jax_backend = JaxSVBackend(sequence(), config=config)
    jax_results = jax_backend.run()

    sv_backend = SVBackend(sequence(), config=config)

    expected_results = sv_backend.run()

    assert torch.allclose(torch.tensor(jax_results.get_result_times("energy")), torch.tensor(expected_results.get_result_times("energy")))

    jax_energy = [torch.from_numpy(np.array(x)).to(torch.double) for x in jax_results.energy]

    assert torch.allclose(torch.tensor(jax_energy), torch.tensor(expected_results.energy))
