import math

import torch
from emu_mps import (
    MPS,
    MPSConfig,
    MPSBackend,
    BitStrings,
    Fidelity,
    Occupation,
    StateResult,
)

import pulser
from pulser.devices import MockDevice

from typing import Any

import numpy as np

dtype = torch.complex128


def afm_sequence_from_register(
    reg: pulser.Register,
    Omega_max: float,
    delta_0: float,
    delta_f: float,
    t_rise: int,
    t_fall: int,
    factor_sweep: int,
    device: Any = pulser.devices.MockDevice,
) -> pulser.Sequence:
    """Sequence that creates AntiFerromagnetic State (AFM) for 1d chain of atoms using pulser.
    This function constructs a sequence of pulses to transition a system of qubits
    distributed in a 1d chain (represented by `reg`) into an AFM state using a specified device.
    The sequence consists of three phases: a rise, a sweep, and a fall.
    For more information, check Pulser
    [tutorial](https://pulser.readthedocs.io/en/stable/tutorials/afm_prep.html)."""

    t_sweep = (delta_f - delta_0) / (2 * np.pi * 10) * 1000 * factor_sweep
    rise = pulser.Pulse.ConstantDetuning(
        pulser.waveforms.RampWaveform(t_rise, 0.0, Omega_max), delta_0, 0.0
    )
    sweep = pulser.Pulse.ConstantAmplitude(
        Omega_max, pulser.waveforms.RampWaveform(int(t_sweep), delta_0, delta_f), 0.0
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


def square_perimeter_points(L: int) -> np.ndarray:
    """
    Calculate the coordinates of the points located on the perimeter of a square of size L.
    The square is centered at the origin (0, 0) with sides parallel to the axes.
    The points are ordered starting from the bottom-left corner and moving
    counter-clockwise around the square. The order is important when measuare the bitstrings

    Args:
        L (int): The length of the side of the square. L should be a positive integer.

    Returns:
        np.ndarray: An array of shape (4*L-4, 2) containing the coordinates of the perimeter points.

    Example:
        >>> square_perimeter_points(3)
        array([[-1, -1],
               [-1,  0],
               [-1,  1],
               [ 0,  1],
               [ 1,  1],
               [ 1,  0],
               [ 1, -1],
               [ 0, -1]])
    """
    pairOrodd = L % 2
    toGrid = int(math.floor(L / 2))
    if pairOrodd == 0:
        axis = list(range(-toGrid, toGrid, 1))
    else:
        axis = list(range(-toGrid, toGrid + 1, 1))
    coord = []
    for i in axis:  # from left, first column of the perimeter
        coord.append([axis[0], i])

    for i in axis[1:-1]:
        coord.append([i, axis[-1]])

    for i in reversed(axis):
        coord.append([axis[-1], i])

    for i in reversed(axis[1:-1]):
        coord.append([i, axis[0]])

    return np.array(coord)


Omega_max = 2 * 2 * np.pi
delta_0 = -6 * Omega_max / 2
delta_f = 1 * Omega_max / 2
t_rise = 50
t_fall = 150
sweep_factor = 2

square_length = 2
R_interatomic = MockDevice.rydberg_blockade_radius(Omega_max / 2)

coords = R_interatomic * square_perimeter_points(square_length)
reg = pulser.Register.from_coordinates(coords)
# reg.draw(blockade_radius=R_interatomic, draw_graph=True, draw_half_radius=True)

seq = afm_sequence_from_register(
    reg, Omega_max, delta_0, delta_f, t_rise, t_fall, sweep_factor, MockDevice
)
# seq.draw("input")

dt = 10
eval_times = [1.0]

basis = ("r", "g")

sampling_times = 1000
bitstrings = BitStrings(evaluation_times=eval_times, num_shots=sampling_times)


nqubits = len(seq.register.qubit_ids)

afm_string_pure = {
    "rg" * int((2 * square_length + 2 * (square_length - 2)) / 2): 1.0
}  # |10101010> in ground-rydberg basis

afm_mps_state = MPS.from_state_amplitudes(eigenstates=basis, amplitudes=afm_string_pure)
fidelity_mps_pure = Fidelity(evaluation_times=eval_times, state=afm_mps_state)
density = Occupation(
    evaluation_times=[x / seq.get_duration() for x in range(0, seq.get_duration(), dt)]
)
state = StateResult(evaluation_times=eval_times)

random_xy = torch.rand((nqubits, nqubits))
# lindbladians are using the leakage state |x> as |r1>.
# the basis will be |g>, |r>, |x>=|r1>
l1 = torch.zeros(3, 3, dtype=dtype)
l1[1, 2] = 1.0  # l1 = |r><x| = |r><r1| for xy model
l2 = l1.mT  # l2 = |x><r|= |r1><r| for xy model
lindbladians = [l1, l2]

noise_model = pulser.NoiseModel(
    eff_noise_opers=lindbladians,
    eff_noise_rates=(0.00, 0.00),  # decay from |r> to |r1> and from |r1> to |r>
    with_leakage=True,  # the leakage will be the |r1> for the xy model (|r> is |0>, |r1> is |1>)
)


mpsconfig = MPSConfig(
    dt=dt,
    precision=1.0e-9,
    observables=[bitstrings, fidelity_mps_pure, density, state],
    log_level=100000,
    interaction_matrix_xy=random_xy,
    noise_model=noise_model,
)

sim = MPSBackend(seq, config=mpsconfig)
results = sim.run()

print(results.get_result_tags())

# print statistics of step 1
statistics = results.statistics
print(f"Number of discretized steps:\n {len(statistics)}")
print("Step 1 statistics\n", statistics[1])

results.get_result_times(bitstrings)
bitstrings_final = results.get_result(bitstrings, 1.0)
# print("bitstrings:\n",bitstrings_final)


statefi = (results.get_result(state, 1.0)).factors
print("factor 0 of final state,\n", statefi[0])  # MPS at final time

fidelity_pure = results.get_result(fidelity_mps_pure, 1.0)

print(
    "The probability of the system being in the sate |rgrgrgr> is equal to {} ".format(
        fidelity_pure,
    )
)
