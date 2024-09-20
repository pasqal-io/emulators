# script that uses the interaction matrix instead of atmos positions

import torch

import numpy as np

import pulser
from pulser.devices import AnalogDevice


# emu_mps backend, confing and observables
from emu_mps import MPS, MPSConfig, MPSBackend, StateResult, BitStrings, Fidelity

# interaction matrix for 5 atoms
interaction_matrix = torch.tensor(
    [
        [0.0000, 5.4202, 5.4202, 0.6775, 43.3613],
        [5.4202, 0.0000, 0.6775, 5.4202, 43.3613],
        [5.4202, 0.6775, 0.0000, 5.4202, 43.3613],
        [0.6775, 5.4202, 5.4202, 0.0000, 43.3613],
        [43.3613, 43.3613, 43.3613, 43.3613, 0.0000],
    ]
)

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


# we initialize a Backend instance
sim = MPSBackend()

# Configuration for the Backend and for observables
dt = 100  # time step for discretization, by the default: dt =10

# information for the observables

final_time = seq.get_duration() // dt * dt
# Calculate the final time of the sequence. Some observables will be measured at time steps
# that are multiples of dt.  Using integer division // ensures time steps align with dt.
# NOTE: The sequence is discretized by dt, so the state evolves at these time steps.

times = {final_time}  # final step for an observable to be measure
# all the times have to be a multiple of dt


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

afm_mps_state = MPS.from_state_string(
    basis=basis, nqubits=nqubits, strings=afm_string_state
)
another_afm_mps = MPS.from_state_string(basis=basis, nqubits=nqubits, strings=another_afm)


state_result = StateResult(evaluation_times=times)
bitstrings = BitStrings(evaluation_times=times, num_shots=1000)
fidelity = Fidelity(evaluation_times=times, state=afm_mps_state)  # fidelity number 1
fidelity_another_state = Fidelity(
    evaluation_times=times, state=another_afm_mps
)  # fidelity number 2


# we give the configuration of the backend and the observables
mpsconfig = MPSConfig(
    dt=dt,
    observables=[state_result, bitstrings, fidelity, fidelity_another_state],
    interaction_matrix=interaction_matrix,
)

results = sim.run(seq, mpsconfig)

# all the observables computed
results.get_result_names()

# bitstrings at a given time
bitstrings_final = results[bitstrings.name()][final_time]  # get the bitstring

max_val = max(bitstrings_final.values())  # max number of counts in the bitstring
print("Max count value: ", max_val)
max_string = [key for key, value in bitstrings_final.items() if value == max_val]
print("Bitstring with the max number of counts: ", max_string)
