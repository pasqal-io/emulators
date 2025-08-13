import matplotlib.pyplot as plt
from emu_mps import (
    MPS,
    MPSConfig,
    MPSBackend,
    BitStrings,
    Fidelity,
    Occupation,
)
from utils_examples import afm_sequence_from_register, square_perimeter_points

import pulser
from pulser.devices import AnalogDevice
import numpy as np


Omega_max = 2 * 2 * np.pi
delta_0 = -6 * Omega_max / 2
delta_f = 1 * Omega_max / 2
t_rise = 500
t_fall = 1500
sweep_factor = 2

square_length = 3
R_interatomic = AnalogDevice.rydberg_blockade_radius(Omega_max / 2)

coords = R_interatomic * square_perimeter_points(square_length)
reg = pulser.Register.from_coordinates(coords)

seq = afm_sequence_from_register(
    reg, Omega_max, delta_0, delta_f, t_rise, t_fall, sweep_factor, AnalogDevice
)

dt = 100
eval_times = [1.0]

basis = ("r", "g")

sampling_times = 1000
bitstrings = BitStrings(evaluation_times=eval_times, num_shots=sampling_times)

nqubits = len(seq.register.qubit_ids)

afm_string_pure = {"rgrgrgrg": 1.0}

afm_mps_state = MPS.from_state_amplitudes(eigenstates=basis, amplitudes=afm_string_pure)
fidelity_mps_pure = Fidelity(evaluation_times=eval_times, state=afm_mps_state)

density = Occupation(
    evaluation_times=[x / seq.get_duration() for x in range(0, seq.get_duration(), dt)]
)

mpsconfig = MPSConfig(
    dt=dt,
    observables=[
        bitstrings,
        fidelity_mps_pure,
        density,
    ],
    adiabatic_evolution=True,
)

sim = MPSBackend(seq, config=mpsconfig)
results = sim.run()
results.get_result_times(bitstrings)
bitstrings_final = results.get_result(bitstrings, 1.0)

max_val = max(bitstrings_final.values())  # max number of counts in the bitstring
max_string = [key for key, value in bitstrings_final.items() if value == max_val]
print(
    "The most frequent bitstring is {} which was sampled {} times".format(
        max_string, max_val
    )
)


filtered_counts = [count for count in bitstrings_final.values() if count > 20]
filtered_bitstrings = [
    bitstring for bitstring, count in bitstrings_final.items() if count > 20
]
fidelity_pure = results.get_result(fidelity_mps_pure, 1.0)

# print(
#     "The fidelity computed for the system final state against the pure state |rgrgrgr> is {}.\n
# The probability of the system being in that sate is equal to {} ".format(
#         fidelity_pure, abs(fidelity_pure) ** 2
#     )
# )

magnetization_values = np.array(list(results.occupation))
magnetization_times = results.get_result_times(density)

fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")

num_time_points, positions = magnetization_values.shape
x, y = np.meshgrid(np.arange(num_time_points), np.arange(1, positions + 1))
im = plt.pcolormesh(magnetization_times, y, magnetization_values.T, shading="auto")
ax.set_xlabel("Time [ns]")
ax.set_ylabel("Qubit")
ax.set_title("State Density")
ax.set_yticks(np.arange(1, positions + 1))
cb = fig.colorbar(im, ax=ax)

# fig.savefig(f"TDVP.png", dpi=300, bbox_inches="tight")
