# script that uses the interaction matrix instead of atmos positions
import numpy as np

import pulser
from pulser.devices import AnalogDevice


# emu_mps backend, confing and observables
from emu_mps import MPS, MPSConfig, MPSBackend, StateResult, BitStrings, Fidelity, Occupation, CorrelationMatrix

# interaction matrix for 5 atoms
interaction_matrix = [
    [0.0000, 5.4202, 5.4202, 0.6775, 43.3613],
    [5.4202, 0.0000, 0.6775, 5.4202, 43.3613],
    [5.4202, 0.6775, 0.0000, 5.4202, 43.3613],
    [0.6775, 5.4202, 5.4202, 0.0000, 43.3613],
    [43.3613, 43.3613, 43.3613, 43.3613, 0.0000],
]

# Pulser setup for AFM state

Omega_max = 2 * 2 * np.pi
U = Omega_max / 2

delta_0 = -6 * U
delta_f = 1 * U

t_rise = 500
t_fall = 1500
sweep_factor = 2  # time for the sweep

R_interatomic = AnalogDevice.rydberg_blockade_radius(U)  # separation between atoms

# create a ficticious register, this register is not important
# because we are going to use the above interaction matrix
# NOTE: the number of atoms should be the same as in the interaction matrix
reg = pulser.Register.rectangle(5, 1, spacing=9.0)

t_sweep = (delta_f - delta_0) / (2 * np.pi * 10) * 1000 * 2
rise = pulser.Pulse.ConstantDetuning(
    pulser.waveforms.RampWaveform(t_rise, 0.0, Omega_max), delta_0, 0.0
)
sweep = pulser.Pulse.ConstantAmplitude(
    Omega_max, pulser.waveforms.RampWaveform(t_sweep, delta_0, delta_f), 0.0
)
fall = pulser.Pulse.ConstantDetuning(
    pulser.waveforms.RampWaveform(t_fall, Omega_max, 0.0), delta_f, 0.0
)
seq = pulser.Sequence(reg, AnalogDevice)
seq.declare_channel("ising_global", "rydberg_global")
seq.add(rise, "ising_global")
seq.add(sweep, "ising_global")
seq.add(fall, "ising_global")


# Configuration for the Backend and for observables
dt = 100  # time step for discretization, by the default: dt =10

# information for the observables
times = [1.0]  # final step for an observable to be measure

# As an example, we are going to create an antiferromagnetic (afm) state (mps) and
# a suporpostion of afm states in order to calculate the
# fidelity against the evolved state

basis = (
    "r",
    "g",
)  # basis that the observables are going to be measured.
# At the moment, we are only accepting the rydberg basis

nqubits = len(seq.register.qubit_ids)  # qubit_ids, for all observables

afm_string_state = {"rgrgr": 1.0}

another_afm = {
    "rgrgr": 1.0 / np.sqrt(2),
    "grgrg": 1.0 / np.sqrt(2),
}

afm_mps_state = MPS.from_state_amplitudes(eigenstates=basis, amplitudes=afm_string_state)
another_afm_mps = MPS.from_state_amplitudes(eigenstates=basis, amplitudes=another_afm)


state_result = StateResult(evaluation_times=times)
bitstrings = BitStrings(evaluation_times=times, num_shots=1000)
fidelity = Fidelity(
    evaluation_times=times, state=afm_mps_state, tag_suffix="afm"
)  # fidelity number 1
fidelity_another_state = Fidelity(
    evaluation_times=times, state=another_afm_mps, tag_suffix="bell_afm"
)  # fidelity number 2

occup = Occupation(evaluation_times=times)
corr = CorrelationMatrix(evaluation_times=times)

# we give the configuration of the backend and the observables
mpsconfig = MPSConfig(
    dt=dt,
    observables=[bitstrings, fidelity, fidelity_another_state, occup, corr],
    interaction_matrix=interaction_matrix,
)

# we initialize a Backend instance
sim = MPSBackend(seq, config=mpsconfig)

results = sim.run()

# all the observables computed
results.get_result_tags()

# bitstrings at a given time
bitstrings_final = results.get_result(bitstrings, 1.0)  # get the bitstring

max_val = max(bitstrings_final.values())  # max number of counts in the bitstring
print("Max count value: ", max_val)
max_string = [key for key, value in bitstrings_final.items() if value == max_val]
print("Bitstring with the max number of counts: ", max_string)
